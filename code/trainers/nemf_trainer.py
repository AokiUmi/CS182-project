import torch
import torch.nn.functional as F
import random
import time

import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from collections import OrderedDict
from utils.utils import print_current_loss
from os.path import join as pjoin
import codecs as cs
import torch.distributed as dist
import numpy as np
from utils.plot_script import *
from utils.motion_process import *
import utils.paramUtil as paramUtil
from tqdm import tqdm

from utils.skeleton import Skeleton


from mmcv.runner import get_dist_info
from models.gaussian_diffusion import (
    GaussianDiffusion,
    get_named_beta_schedule,
    create_named_schedule_sampler,
    UniformSamplerGeneric,
    ModelMeanType,
    ModelVarType,
    LossType
)

from datasets import build_dataloader


class MFTrainer(object):

    def __init__(self, args, encoder):
        self.opt = args
        self.device = args.device
        self.encoder = encoder
        self.diffusion_steps = args.diffusion_steps
        sampler = 'uniform'
        beta_scheduler = 'linear'
        betas = get_named_beta_schedule(beta_scheduler, self.diffusion_steps)
        self.diffusion = GaussianDiffusion(
            betas=betas,
            model_mean_type=ModelMeanType.EPSILON,
            model_var_type=ModelVarType.FIXED_SMALL,
            loss_type=LossType.MSE
        )
        self.sampler = create_named_schedule_sampler(sampler, self.diffusion)
        self.sampler_name = sampler

        self.frame_sampler = UniformSamplerGeneric()

        if args.is_train:
            self.mse_criterion = torch.nn.MSELoss(reduction='none')
        self.to(self.device)

    @staticmethod
    def zero_grad(opt_list):
        for opt in opt_list:
            opt.zero_grad()

    @staticmethod
    def clip_norm(network_list):
        for network in network_list:
            clip_grad_norm_(network.parameters(), 0.5)

    @staticmethod
    def step(opt_list):
        for opt in opt_list:
            opt.step()

    def forward(self, batch_data, eval_mode=False):
        caption, motions, m_lens = batch_data
        motions = motions.detach().to(self.device).float()

        self.caption = caption
        self.motions = motions

        B, T = motions.shape[:2]
        cur_len = torch.LongTensor([min(T, m_len) for m_len in  m_lens]).to(self.device)
        t, _ = self.sampler.sample(B, motions.device)

        f = self.frame_sampler.sample(cur_len-40, motions.device)

        x = []
        context = []
        length = 0
        for i in range(len(f)):
            x.append(motions[[i],f[i].item():f[i].item()+40])
            ctx = motions[[i], :f[i].item()]
            if length < ctx.shape[1]:
                length = ctx.shape[1]
            context.append(motions[[i], :f[i].item()])

        pads = []
        for i in range(len(f)):
            l = length - context[i].shape[1]
            pads.append(l)
            context[i] = torch.cat([torch.zeros([1, l, 263], device=context[i].device), context[i]], dim=1)

        x_start = torch.cat(x, dim=0)
        context = torch.cat(context, dim=0)

        # x_start = motions[:, f[0]:f[0]+10]
        # context = motions[:, :f[0]]
        # self.encoder.context_embed.init_hidden()

        output = self.diffusion.training_losses(
            model=self.encoder,
            x_start=x_start,
            t=t,
            model_kwargs={"text": caption, "frames": f, "context": context, "pads":pads}
        )

        self.real_noise = output['target']
        self.fake_noise = output['pred']
        try:
            self.src_mask = self.encoder.module.generate_src_mask(B, T, cur_len).to(x_start.device)
        except:
            self.src_mask = self.encoder.generate_src_mask(B, T, cur_len).to(x_start.device)


    def backward_G(self):
        loss_mot_rec = self.mse_criterion(self.fake_noise, self.real_noise).mean()
        # loss_mot_rec = (loss_mot_rec * self.src_mask[:,1:]).sum() / self.src_mask.sum()
        self.loss_mot_rec = loss_mot_rec
        loss_logs = OrderedDict({})
        loss_logs['loss_mot_rec'] = self.loss_mot_rec.item()
        return loss_logs

    def update(self):
        self.zero_grad([self.opt_encoder])
        loss_logs = self.backward_G()
        self.loss_mot_rec.backward()
        self.clip_norm([self.encoder])
        self.step([self.opt_encoder])

        return loss_logs

    def to(self, device):
        if self.opt.is_train:
            self.mse_criterion.to(device)
        self.encoder = self.encoder.to(device)

    def train_mode(self):
        self.encoder.train()

    def eval_mode(self):
        self.encoder.eval()

    def save(self, file_name, ep, total_it):
        state = {
            'opt_encoder': self.opt_encoder.state_dict(),
            'ep': ep,
            'total_it': total_it
        }
        try:
            state['encoder'] = self.encoder.module.state_dict()
        except:
            state['encoder'] = self.encoder.state_dict()
        torch.save(state, file_name, _use_new_zipfile_serialization=False)
        return

    def load(self, model_dir):
        checkpoint = torch.load(model_dir, map_location=self.device)
        # if self.opt.is_train:
        #     self.opt_encoder.load_state_dict(checkpoint['opt_encoder'])
        self.encoder.load_state_dict(checkpoint['encoder'], strict=True)
        return checkpoint['ep'], checkpoint.get('total_it', 0)

    def train(self, train_dataset):
        rank, world_size = get_dist_info()
        self.to(self.device)
        self.opt_encoder = optim.AdamW(self.encoder.parameters(), lr=self.opt.lr)
        it = 0
        cur_epoch = 0
        if self.opt.is_continue:
            model_dir = pjoin(self.opt.model_dir, 'latest.tar')
            print(model_dir)
            cur_epoch, it = self.load(model_dir)

        start_time = time.time()

        train_loader = build_dataloader(
            train_dataset,
            samples_per_gpu=self.opt.batch_size,
            drop_last=True,
            dist=False,
            workers_per_gpu=4,
            shuffle=True)

        logs = OrderedDict()
        for epoch in range(cur_epoch, self.opt.num_epochs):
            self.train_mode()
            for i, batch_data in enumerate(train_loader):
                self.forward(batch_data)
                log_dict = self.update()
                for k, v in log_dict.items():
                    if k not in logs:
                        logs[k] = v
                    else:
                        logs[k] += v
                it += 1
                if it % self.opt.log_every == 0 and rank == 0:
                    mean_loss = OrderedDict({})
                    for tag, value in logs.items():
                        mean_loss[tag] = value / self.opt.log_every
                    logs = OrderedDict()
                    print_current_loss(start_time, it, mean_loss, epoch, inner_iter=i)

                if it % self.opt.save_latest == 0 and rank == 0:
                    self.save(pjoin(self.opt.model_dir, 'latest.tar'), epoch, it)

            if rank == 0:
            # if 1:
                self.save(pjoin(self.opt.model_dir, 'latest.tar'), epoch, it)

            if epoch % self.opt.save_every_e == 0 and rank == 0:
                self.save(pjoin(self.opt.model_dir, 'ckpt_e%03d.tar'%(epoch)),
                            epoch, total_it=it)

    def plot_t2m(self, mp_data, result_path, caption):
        mp_joint = []
        for data in mp_data:
            n_raw_offsets = torch.from_numpy(paramUtil.t2m_raw_offsets)
            skel = Skeleton(n_raw_offsets, paramUtil.t2m_kinematic_chain, 'cpu')
            # joint = recover_from_ric(data.float(), 22)
            # skel.get_offsets_joints(joint[0])
            joint = recover_from_ric(data.float(), 22)
            # joint = recover_from_rot(data.float(), 22, skel)
            mp_joint.append(joint.numpy())
        # joint = motion_temporal_filter(joint, sigma=1)
        plot_3d_motion(result_path, paramUtil.t2m_kinematic_chain, mp_joint, title=caption, fps=20)

    def gen(self, train_dataset):
        self.to(self.device)

        cur_epoch = 0
        if self.opt.is_continue:
            model_dir = pjoin(self.opt.model_dir, 'latest.tar')
            print(model_dir)
            cur_epoch, it = self.load(model_dir)

        train_loader = build_dataloader(
            train_dataset,
            samples_per_gpu=self.opt.batch_size,
            drop_last=True,
            dist=False,
            workers_per_gpu=1,
            shuffle=True)

        for epoch in range(cur_epoch, self.opt.num_epochs):
            self.eval_mode()
            for i, batch_data in enumerate(train_loader):

                caption, motions, m_lens = batch_data
                motions = motions.float().to(self.device)
                context = motions[:, :40]

                #caption = ("a person is punching the opponent","another one dodges the attack",)

                out_batch = self.generate_batch(caption, context, 263, sample_length=200).cpu().detach()
                # out_batch = motions.cpu().detach()

                out_batch *= train_dataset.std
                out_batch += train_dataset.mean

                self.plot_t2m([out_batch[0],out_batch[1]], "./1.mp4", '. '.join(caption))
                np.savetxt("1.txt", out_batch[0])
                np.savetxt("2.txt", out_batch[1])
                # return

    def generate_batch(self, caption, context, dim_pose, sample_length=200):
        xf_proj, xf_out = self.encoder.encode_text(caption, self.device)

        B = len(caption)
        frames = torch.ones([B]).to(self.device)
        output = []
        # context = torch.zeros([B, 0, dim_pose], device=xf_proj.device)
        for i in tqdm(range(sample_length//40-2)):
            cur_state = self.diffusion.p_sample_loop(
                self.encoder,
                (B, 40, dim_pose),
                clip_denoised=False,
                progress=False,
                model_kwargs={
                    'xf_proj': xf_proj,
                    'xf_out': xf_out,
                    'frames': frames,
                    'context': context,
                })
            output.append(cur_state)
            context = torch.cat([context, cur_state.clone()],dim=1)
            frames += 1
        output = context
        return output

    def generate(self, caption, m_lens, dim_pose, batch_size=16):
        N = len(caption)
        cur_idx = 0
        self.encoder.eval()
        all_output = []
        while cur_idx < N:
            if cur_idx + batch_size >= N:
                batch_caption = caption[cur_idx:]
                batch_m_lens = m_lens[cur_idx:]
            else:
                batch_caption = caption[cur_idx: cur_idx + batch_size]
                batch_m_lens = m_lens[cur_idx: cur_idx + batch_size]
            output = self.generate_batch(batch_caption, batch_m_lens, dim_pose)
            B = output.shape[0]

            for i in range(B):
                all_output.append(output[i])
            cur_idx += batch_size
        return all_output