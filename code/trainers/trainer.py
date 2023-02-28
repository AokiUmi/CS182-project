import torch
import torch.nn.functional as F
import random
import time

import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from collections import OrderedDict
from utils.utils import *
from os.path import join as pjoin
import codecs as cs
import torch.distributed as dist
import numpy as np
from utils.plot_script import *
from utils.motion_process import *
import utils.paramUtil as paramUtil
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from utils.skeleton import Skeleton


from mmcv.runner import get_dist_info


from datasets import build_dataloader

mean_lagacy = torch.from_numpy(np.load('mean1.npy'))
std_lagacy = torch.from_numpy(np.load('std1.npy'))

joints_num = 22
std_lagacy[0:1] = std_lagacy[0:1] / 5
# root_linear_velocity (B, seq_len, 2)
std_lagacy[1:3] = std_lagacy[1:3] / 5
# root_y (B, seq_len, 1)
std_lagacy[3:4] = std_lagacy[3:4] / 5
# ric_data (B, seq_len, (joint_num - 1)*3)
std_lagacy[4: 4 + (joints_num - 1) * 3] = std_lagacy[4: 4 + (joints_num - 1) * 3] / 1.0
# rot_data (B, seq_len, (joint_num - 1)*6)
std_lagacy[4 + (joints_num - 1) * 3: 4 + (joints_num - 1) * 9] = std_lagacy[4 + (joints_num - 1) * 3: 4 + (
        joints_num - 1) * 9] / 1.0
# local_velocity (B, seq_len, joint_num*3)
std_lagacy[4 + (joints_num - 1) * 9: 4 + (joints_num - 1) * 9 + joints_num * 3] = std_lagacy[
                                                                           4 + (joints_num - 1) * 9: 4 + (
                                                                                   joints_num - 1) * 9 + joints_num * 3] / 1.0
# foot contact (B, seq_len, 4)
std_lagacy[4 + (joints_num - 1) * 9 + joints_num * 3:] = std_lagacy[
                                                  4 + (joints_num - 1) * 9 + joints_num * 3:] / 5

mean = torch.from_numpy(np.load('mean.npy'))
std = torch.from_numpy(np.load('std.npy'))

joints_num = 22
std[0:1] = std[0:1] / 3
# root_linear_velocity (B, seq_len, 2)
std[1:3] = std[1:3] / 3
# root_y (B, seq_len, 1)
std[3:4] = std[3:4] / 3
# ric_data (B, seq_len, (joint_num - 1)*3)
std[4: 4 + (joints_num - 1) * 3] = std[4: 4 + (joints_num - 1) * 3] / 1.0
# rot_data (B, seq_len, (joint_num - 1)*6)
std[4 + (joints_num - 1) * 3: 4 + (joints_num - 1) * 9] = std[4 + (joints_num - 1) * 3: 4 + (
        joints_num - 1) * 9] / 1.0
# local_velocity (B, seq_len, joint_num*3)
std[4 + (joints_num - 1) * 9: 4 + (joints_num - 1) * 9 + joints_num * 3] = std[
                                                                           4 + (joints_num - 1) * 9: 4 + (
                                                                                   joints_num - 1) * 9 + joints_num * 3] / 1.0
# foot contact (B, seq_len, 4)
std[4 + (joints_num - 1) * 9 + joints_num * 3:] = std[
                                                  4 + (joints_num - 1) * 9 + joints_num * 3:] / 3

class Trainer(object):

    def __init__(self, args, model):
        self.opt = args
        self.device = args.device
        self.model = model
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.opt.lr, weight_decay=0.00001)
        self.normalizer = MotionNormalizer(dataset=self.opt.dataset_name)

        if args.is_train:
            self.mse_criterion = torch.nn.MSELoss(reduction='none')
        self.to(self.device)
        self.writer = SummaryWriter(self.opt.log_dir)

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
        root, caption, motions, m_lens, relative = batch_data
        motions = motions.detach().to(self.device).float()

        B, T = motions.shape[:2]
        cur_len = torch.LongTensor([min(T, m_len[0]) for m_len in m_lens]).to(self.device)
        padded_len = cur_len.max()

        batch = OrderedDict({})
        batch["caption"] = caption
        batch["motions"] = motions.reshape(B,T,-1)[:,:padded_len]
        batch["m_lens"] = m_lens
        batch["relative"] = relative

        # batch = self.model.encode_motion(batch)
        self.loss, self.loss_logs = self.model.compute_loss(batch)



    def update(self):

        self.zero_grad([self.optimizer])
        self.loss.backward()
        self.clip_norm([self.model])
        self.step([self.optimizer])


    def to(self, device):
        if self.opt.is_train:
            self.mse_criterion.to(device)
        self.model = self.model.to(device)

    def train_mode(self):
        self.model.train()

    def eval_mode(self):
        self.model.eval()

    def save(self, file_name, ep, total_it):
        state = {
            'optimizer': self.optimizer.state_dict(),
            'ep': ep,
            'total_it': total_it
        }
        try:
            state['model'] = self.model.module.state_dict()
        except:
            state['model'] = self.model.state_dict()
        torch.save(state, file_name, _use_new_zipfile_serialization=False)
        return

    def load(self, model_dir):
        checkpoint = torch.load(model_dir, map_location=self.device)
        # if self.opt.is_train:
        #     self.optimizer.load_state_dict(checkpoint['optimizer'])
        # print(checkpoint['model'].keys())

        # weights = {}
        # for key in checkpoint['model'].keys():
        #     if "encoder" in key.split(".")[0]:
        #         inner_key = '.'.join(key.split(".")[1:])
        #         weights[inner_key] = checkpoint['model'][key]
        # self.model.encoder.load_state_dict(weights, strict=True)

        weights = {}
        for key in checkpoint['model'].keys():
            if "decoder" in key.split(".")[0]:
                inner_key = '.'.join(key.split(".")[1:])
                weights[inner_key] = checkpoint['model'][key]
        self.model.decoder.load_state_dict(weights, strict=True)
        # #
        self.model.load_state_dict(checkpoint['model'], strict=False)

        # weights = {}
        # for key in checkpoint['model'].keys():
        #     if "diffusion_prior" in key.split(".")[0]:
        #         inner_key = '.'.join(key.split(".")[1:])
        #         weights[inner_key] = checkpoint['model'][key]
        # self.model.diffusion_prior.load_state_dict(weights, strict=True)



        print("Model loaded~~")
        return checkpoint['ep'], checkpoint.get('total_it', 0)

    def train(self, train_dataset):
        rank, world_size = get_dist_info()
        self.to(self.device)
        it = 0
        cur_epoch = 0
        if self.opt.rewake:
            model_dir = pjoin(self.opt.model_dir, 'latest.tar')
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
            self.eval_mode()
            for i, batch_data in tqdm(enumerate(train_loader)):
                root, caption, motions, m_lens, relative = batch_data
                if torch.isnan(motions).sum() > 0.5:
                    print("has nan")
                    continue
                self.forward(batch_data)
                # if self.loss.item() > 0.3:
                #     print(caption)
                #     fail_list.append(caption[0]+"\n")
                    # continue
                self.update()
                for k, v in self.loss_logs.items():
                    self.writer.add_scalar(k, v, it)
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
                    print_current_loss(start_time, it, mean_loss, epoch, inner_iter=i, lr=self.opt.lr)


                if it % self.opt.save_latest == 0 and rank == 0:
                    self.save(pjoin(self.opt.model_dir, 'latest.tar'), epoch, it)

            if rank == 0:
            # if 1:
                self.save(pjoin(self.opt.model_dir, 'latest.tar'), epoch, it)

            if epoch % self.opt.save_every_e == 0 and rank == 0:
                self.save(pjoin(self.opt.model_dir, 'ckpt_e%03d.tar'%(epoch)),
                            epoch, total_it=it)
            # f.writelines(fail_list)
            # break

    def generate_batch(self, caption, m_lens, dim_pose):

        batch = OrderedDict({})
        batch["caption"] = caption
        batch["m_lens"] = m_lens

        B = len(caption)

        out_batch = self.model(batch)
        motion_output = out_batch["output"].reshape(-1, out_batch["output"].shape[1], 263)
        motion_output = motion_output * std.to(motion_output.device) + mean.to(motion_output.device)
        motion_output = (motion_output - mean_lagacy.to(motion_output.device))/std_lagacy.to(motion_output.device)
        # motion_output = torch.zeros([B, 196 ,263]).float()

        return motion_output

    def generate(self, caption, m_lens, dim_pose, batch_size=1024):
        N = len(caption)
        cur_idx = 0
        self.model.eval()
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

    def gen(self, train_dataset):
        self.to(self.device)

        cur_epoch = 0
        if self.opt.rewake:
            model_dir = pjoin(self.opt.model_dir, 'latest.tar')
            cur_epoch, it = self.load(model_dir)

        train_loader = build_dataloader(
            train_dataset,
            samples_per_gpu=self.opt.batch_size,
            drop_last=True,
            dist=False,
            workers_per_gpu=1,
            shuffle=True)

        self.eval_mode()
        for i, batch_data in enumerate(train_loader):
            root, caption, motions, m_lens, relative = batch_data
            print(relative)
            motions = motions.detach().to(self.device).float()

            B, T = motions.shape[:2]
            cur_len = torch.LongTensor([min(T, m_len[0]) for m_len in m_lens]).to(self.device)
            padded_len = cur_len.max()

            batch = OrderedDict({})
            # caption = ("the first person greets to the second person and then they hug each other",)
            batch["caption"] = caption
            batch["motions"] = motions.reshape(B,T,-1)
            batch["m_lens"] = m_lens
            batch["relative"] = relative
            motions = motions.cpu().detach().numpy().reshape(B, T, -1, 263)[:, :padded_len]


            #
            # # caption = ("two people are practicing fencing",)
            #
            for i in range(1):

                # self.plot_t2m([motion_output[0],motions[0]], "./1.mp4", '. '.join(caption))

                if self.opt.dataset_name == "interaction":
                    self.plot_t2m([motions[0,:,0], motions[0,:,1]], "results/"+root[0].replace("/", "_").replace("\\", "_")+".mp4", caption[0], relative[0])
                    # self.plot_t2m([motion_output[0, :, 0], motion_output[0, :, 1]], "results/" + root[0].replace("/", "_").replace("\\", "_") + f"_{i}.mp4", caption[0], relative[0])
                else:
                    self.plot_t2m([motion_output[0,:,0], motions[0,:,0]], "results/"+caption[0][:20]+".mp4", caption[0])
                    # self.plot_t2m([motion_output[0,:,0], motions[0,:,0]], "results/"+root[0].replace("/", "_").replace("\\", "_")+".mp4", caption[0])

            for i in range(1,2):
                out_batch = self.model(batch)
                motion_output = out_batch["output"].cpu().detach().numpy().reshape(B,out_batch["output"].shape[1],-1,263)
                motion_output = self.normalizer.backward(motion_output, global_rt=True)

                # self.plot_t2m([motion_output[0],motions[0]], "./1.mp4", '. '.join(caption))

                if self.opt.dataset_name == "interaction":
                    # self.plot_t2m([motions[0,:,0], motions[0,:,1]], "results/"+root[0].replace("/", "_").replace("\\", "_")+".mp4", caption[0], relative[0])
                    self.plot_t2m([motion_output[0, :, 0], motion_output[0, :, 1]], "results/" + root[0].replace("/", "_").replace("\\", "_") + f"_{i}.mp4", caption[0], relative[0])
                else:
                    self.plot_t2m([motion_output[0,:,0], motions[0,:,0]], "results/"+caption[0][:20]+".mp4", caption[0])
                    # self.plot_t2m([motion_output[0,:,0], motions[0,:,0]], "results/"+root[0].replace("/", "_").replace("\\", "_")+".mp4", caption[0])





    def plot_t2m(self, mp_data, result_path, caption, relative=None):
        mp_joint = []
        for i,data in enumerate(mp_data):
            if i==0:
                joint = recover_from_ric(torch.from_numpy(data).float(), 22, global_rt=True).numpy()
            else:
                joint = recover_from_ric(torch.from_numpy(data).float(), 22, relative, global_rt=True).numpy()
            mp_joint.append(joint)
            np.save(f"results/joints/{os.path.basename(result_path)[7:]}_{i}.npy", joint)

        # joint = motion_temporal_filter(joint, sigma=1)
        plot_3d_motion(result_path, paramUtil.t2m_kinematic_chain, mp_joint, title=caption, fps=20)