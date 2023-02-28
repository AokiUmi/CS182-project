"""
Copyright 2021 S-Lab
"""

from cv2 import norm
import torch
import torch.nn.functional as F
from torch import layer_norm, nn
import numpy as np
import clip
import models.mlp as mlp
from models.gru import TemporalEncoder
import math
from models.utils import *
from utils.utils import *

from models.gaussian_diffusion import (
    SpacedDiffusion,
    MotionDiffusion,
    space_timesteps,
    get_named_beta_schedule,
    create_named_schedule_sampler,
    ModelMeanType,
    ModelVarType,
    LossType
)




class MotionEncoder(nn.Module):
    def __init__(self, input_feats, latent_dim=512, num_frames=5000,
                 ff_size=1024, num_layers=8, num_heads=16, dropout=0.0,
                 ablation=None, activation="gelu"):
        super().__init__()


        self.latent_dim = latent_dim

        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout

        self.ablation = ablation
        self.activation = activation

        self.input_feats = input_feats

        self.skelEmbedding = nn.Linear(self.input_feats, self.latent_dim)

        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout, max_len=num_frames)

        seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                          nhead=self.num_heads,
                                                          dim_feedforward=self.ff_size,
                                                          dropout=self.dropout,
                                                          activation=self.activation,
                                                          batch_first=True)
        self.backbone = nn.TransformerEncoder(seqTransEncoderLayer, num_layers=self.num_layers)
        self.out_mu = zero_module(nn.Linear(self.latent_dim, self.latent_dim//2))
        self.out_logvar = zero_module(nn.Linear(self.latent_dim, self.latent_dim//2))

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, batch):
        x, mask = batch["motions"], batch["mask"]
        bs, nframes, nfeats  = x.shape

        xseq = self.skelEmbedding(x)

        xseq = self.sequence_pos_encoder(xseq)

        mu = self.backbone(xseq)[:,0:2]
        batch["z_m_mu"] = self.out_mu(mu[:,0])
        batch["z_m_logvar"] = self.out_logvar(mu[:,1])

        batch["z_m"] = self.reparameterize(batch["z_m_mu"], batch["z_m_logvar"])

        return batch


class MVAE(nn.Module):
    def __init__(self, input_feats, latent_dim=512, num_frames=5000,
                 ff_size=1024, num_layers=8, num_heads=16, dropout=0.0,
                 ablation=None, activation="gelu"):
        super().__init__()


        self.latent_dim = latent_dim

        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout

        self.ablation = ablation
        self.activation = activation

        self.input_feats = input_feats

        self.encoder = MotionEncoder(input_feats)
        seqTransDecoderLayer = nn.TransformerDecoderLayer(d_model=self.latent_dim,
                                                          nhead=self.num_heads,
                                                          dim_feedforward=self.ff_size,
                                                          dropout=self.dropout,
                                                          activation=self.activation,
                                                          batch_first=True)

        self.decoder = nn.Sequential(*[nn.TransformerDecoder(seqTransDecoderLayer,
                                                     num_layers=self.num_layers), zero_module(nn.Linear(self.latent_dim, self.input_feats))])
        # self.prior_logvar = nn.Parameter(torch.zeros(self.latent_dim))

        self.mse_criterion = torch.nn.MSELoss(reduction='none')

    def recover_root_rot_pos(self, data):

        rot_vel = data[..., 0:1]
        r_rot_ang = torch.zeros_like(rot_vel).to(data.device)
        '''Get Y-axis rotation from rotation velocity'''
        r_rot_ang[:,1:] = rot_vel[:, :-1]
        r_rot_ang = torch.cumsum(r_rot_ang, dim=1)

        r_rot_quat = torch.zeros(data.shape[:-1] + (4,)).to(data.device)
        r_rot_quat[..., 0:1] = torch.cos(r_rot_ang)
        r_rot_quat[..., 2:3] = torch.sin(r_rot_ang)

        r_pos = torch.zeros(data.shape[:-1] + (3,)).to(data.device)
        r_pos[:, 1:, [0, 2]] = data[:, :-1, 1:3]
        '''Add Y-axis rotation to root position'''

        r_pos = qrot(qinv(r_rot_quat), r_pos)

        r_pos = torch.cumsum(r_pos, dim=1)

        r_pos[..., 1] = data[..., 3]
        return r_rot_ang, r_rot_quat, r_pos

    def recover_from_ric(self, data, joints_num):
        r_rot_ang, r_rot_quat, r_pos = self.recover_root_rot_pos(data)
        positions = data[..., 4:(joints_num - 1) * 3 + 4]
        positions = positions.view(positions.shape[:-1] + (-1, 3))

        '''Add Y-axis rotation to local joints'''
        positions = qrot(qinv(r_rot_quat[..., None, :]).expand(positions.shape[:-1] + (4,)), positions)

        '''Add root XZ to joints'''
        positions[..., 0] += r_pos[..., 0:1]
        positions[..., 2] += r_pos[..., 2:3]

        '''Concate root and joints'''
        positions = torch.cat([r_pos.unsqueeze(-2), positions], dim=-2)

        return positions, r_rot_ang, r_pos

    def masked_mse(self, prediction, target, mask, mask2=None):
        # mask: B,T,1,1
        # mask2:B,T,1,N
        # data: B,T,1,D
        # data2:B,T,1,N,D
        loss = self.mse_criterion(prediction, target).mean(dim=-1, keepdim=True)
        if mask2 is not None:
            loss = (loss[...,0] * mask2).sum(dim=-1, keepdim=True) / (mask2.sum(dim=-1, keepdim=True)+1.e-7)
        loss = (loss * mask).sum() / (mask.sum()+1.e-7)
        return loss

    def forward(self, batch):
        x, mask = batch["motions"], batch["mask"]
        bs, nframes, nfeats  = x.shape


        # joints, r_rot_angle, r_pos = self.recover_from_ric(x, joints_num=22)
        # local_rt = x[...,:3].reshape(bs,nframes,-1)
        # lowerbody_idx = np.array([1,2,4,5,7,8,10,11])
        # lowerbody_joints = joints[...,lowerbody_idx,:].reshape(bs,nframes,-1)
        # lowerbody_pose = x[...,4+21*3:4+21*3+21*6].reshape(bs,nframes,21,-1)[...,lowerbody_idx-1,:].reshape(bs,nframes,-1)
        # contact = x[...,4+21*3+21*6+22*3:4+21*3+21*6+22*3+4]
        #
        # lowerbody_global_motion = torch.cat([r_rot_angle, r_pos, local_rt, lowerbody_joints, lowerbody_pose, contact], dim=-1)
        # print(lowerbody_global_motion.shape)

        # y = self.lowerSkelEmbedding(lowerbody_global_motion)
        batch = self.encoder(batch)

        # B, T, latent_dim
        h = torch.zeros([bs,nframes,self.latent_dim]).float().to(batch["z_m"].device)
        h = self.sequence_pos_encoder(h)


        # h = self.seqTransEncoder(h)#, src_key_padding_mask=~(src_mask>0.5))
        h = self.seqTransDecoder(h, memory=batch["z_m"].unsqueeze(1))#, tgt_key_padding_mask=~(mask>0.5))


        batch["rec_x"] = self.out_decoder(h).contiguous()

        return batch



    def compute_loss(self, batch):
        # z_m = batch["z_m"]
        z_m_mu = batch["z_m_mu"]
        z_m_logvar = batch["z_m_logvar"]
        x = batch["x"]
        rec_x = batch["rec_x"]
        mask = batch["mask"]

        # prior_var = self.prior_logvar.exp()
        # kld_loss = torch.mean(-0.5 * (1 + z_m_logvar - self.prior_logvar - ((z_m_mu ** 2 + z_m_logvar.exp())/prior_var)))
        kld_loss = torch.mean(-0.5 * (1 + z_m_logvar - ((z_m_mu ** 2 + z_m_logvar.exp()))))
        # shrink_loss = torch.mean(prior_var)

        rec_loss = self.masked_mse(x, rec_x, mask.unsqueeze(-1))
        losses = {}
        losses["rec"] = rec_loss
        losses["kld"] = kld_loss
        # losses["shrink"] = shrink_loss*0.1
        return losses


class TextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = clip.load("ViT-B/32", device="cpu", jit=False)
        set_requires_grad(self.clip_model, False)




    def forward(self, batch):
        x, mask = batch["motions"], batch["mask"]
        bs, nframes, nfeats  = x.shape

        xseq = self.skelEmbedding(x)

        xseq = self.sequence_pos_encoder(xseq)

        mu = self.backbone(xseq)[:,0:2]
        batch["z_m_mu"] = self.out_mu(mu[:,0])
        batch["z_m_logvar"] = self.out_logvar(mu[:,1])

        batch["z_m"] = self.reparameterize(batch["z_m_mu"], batch["z_m_logvar"])

        return batch


class TVAE(nn.Module):
    def __init__(self, input_feats, latent_dim=512, num_frames=5000,
                 ff_size=1024, num_layers=8, num_heads=16, dropout=0.0,
                 ablation=None, activation="gelu"):
        super().__init__()


        self.latent_dim = latent_dim

        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout

        self.ablation = ablation
        self.activation = activation

        self.input_feats = input_feats



        seqTransDecoderLayer = nn.TransformerDecoderLayer(d_model=self.latent_dim,
                                                          nhead=self.num_heads,
                                                          dim_feedforward=self.ff_size,
                                                          dropout=self.dropout,
                                                          activation=self.activation,
                                                          batch_first=True)

        self.decoder = nn.Sequential(*[nn.TransformerDecoder(seqTransDecoderLayer,
                                                     num_layers=self.num_layers), zero_module(nn.Linear(self.latent_dim, self.input_feats))])
        # self.prior_logvar = nn.Parameter(torch.zeros(self.latent_dim))

        self.mse_criterion = torch.nn.MSELoss(reduction='none')

    def recover_root_rot_pos(self, data):

        rot_vel = data[..., 0:1]
        r_rot_ang = torch.zeros_like(rot_vel).to(data.device)
        '''Get Y-axis rotation from rotation velocity'''
        r_rot_ang[:,1:] = rot_vel[:, :-1]
        r_rot_ang = torch.cumsum(r_rot_ang, dim=1)

        r_rot_quat = torch.zeros(data.shape[:-1] + (4,)).to(data.device)
        r_rot_quat[..., 0:1] = torch.cos(r_rot_ang)
        r_rot_quat[..., 2:3] = torch.sin(r_rot_ang)

        r_pos = torch.zeros(data.shape[:-1] + (3,)).to(data.device)
        r_pos[:, 1:, [0, 2]] = data[:, :-1, 1:3]
        '''Add Y-axis rotation to root position'''

        r_pos = qrot(qinv(r_rot_quat), r_pos)

        r_pos = torch.cumsum(r_pos, dim=1)

        r_pos[..., 1] = data[..., 3]
        return r_rot_ang, r_rot_quat, r_pos

    def recover_from_ric(self, data, joints_num):
        r_rot_ang, r_rot_quat, r_pos = self.recover_root_rot_pos(data)
        positions = data[..., 4:(joints_num - 1) * 3 + 4]
        positions = positions.view(positions.shape[:-1] + (-1, 3))

        '''Add Y-axis rotation to local joints'''
        positions = qrot(qinv(r_rot_quat[..., None, :]).expand(positions.shape[:-1] + (4,)), positions)

        '''Add root XZ to joints'''
        positions[..., 0] += r_pos[..., 0:1]
        positions[..., 2] += r_pos[..., 2:3]

        '''Concate root and joints'''
        positions = torch.cat([r_pos.unsqueeze(-2), positions], dim=-2)

        return positions, r_rot_ang, r_pos

    def masked_mse(self, prediction, target, mask, mask2=None):
        # mask: B,T,1,1
        # mask2:B,T,1,N
        # data: B,T,1,D
        # data2:B,T,1,N,D
        loss = self.mse_criterion(prediction, target).mean(dim=-1, keepdim=True)
        if mask2 is not None:
            loss = (loss[...,0] * mask2).sum(dim=-1, keepdim=True) / (mask2.sum(dim=-1, keepdim=True)+1.e-7)
        loss = (loss * mask).sum() / (mask.sum()+1.e-7)
        return loss

    def forward(self, batch):
        x, mask = batch["motions"], batch["mask"]
        bs, nframes, nfeats  = x.shape

        batch = self.encoder(batch)

        # B, T, latent_dim
        h = torch.zeros([bs,nframes,self.latent_dim]).float().to(batch["z_m"].device)
        h = self.sequence_pos_encoder(h)


        # h = self.seqTransEncoder(h)#, src_key_padding_mask=~(src_mask>0.5))
        h = self.seqTransDecoder(h, memory=batch["z_m"].unsqueeze(1))#, tgt_key_padding_mask=~(mask>0.5))


        batch["rec_x"] = self.out_decoder(h).contiguous()

        return batch



    def compute_loss(self, batch):
        # z_m = batch["z_m"]
        z_m_mu = batch["z_m_mu"]
        z_m_logvar = batch["z_m_logvar"]
        x = batch["x"]
        rec_x = batch["rec_x"]
        mask = batch["mask"]

        # prior_var = self.prior_logvar.exp()
        # kld_loss = torch.mean(-0.5 * (1 + z_m_logvar - self.prior_logvar - ((z_m_mu ** 2 + z_m_logvar.exp())/prior_var)))
        kld_loss = torch.mean(-0.5 * (1 + z_m_logvar - ((z_m_mu ** 2 + z_m_logvar.exp()))))
        # shrink_loss = torch.mean(prior_var)

        rec_loss = self.masked_mse(x, rec_x, mask.unsqueeze(-1))
        losses = {}
        losses["rec"] = rec_loss
        losses["kld"] = kld_loss
        # losses["shrink"] = shrink_loss*0.1
        return losses



class MotionDiffusionDecoderModel(nn.Module):
    def __init__(self,
                 input_feats,
                 latent_dim=512,
                 num_frames=240,
                 ff_size=1024,
                 num_layers=8,
                 num_heads=16,
                 dropout=0,
                 activation="gelu",
                 emb_type="clip",
                 **kargs):
        super().__init__()

        self.num_frames = num_frames
        self.latent_dim = latent_dim
        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.activation = activation
        self.input_feats = input_feats
        self.time_embed_dim = latent_dim
        self.emb_type = emb_type
        if emb_type == "bert":
            self.text_emb_dim = 768
        elif emb_type == "clip":
            self.text_emb_dim = 512

        # Input Embedding
        self.joint_embed = nn.Linear(self.input_feats, self.latent_dim)
        self.motion_embed = nn.Linear(self.latent_dim, self.latent_dim)
        self.text_embed = nn.Linear(self.text_emb_dim, self.latent_dim)
        self.relative_embed = nn.Linear(4, self.latent_dim)

        self.time_embed = nn.Sequential(
            nn.Linear(self.latent_dim, self.time_embed_dim),
            nn.SiLU(),
            nn.Linear(self.time_embed_dim, self.time_embed_dim),
        )


        # seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
        #                                                   nhead=self.num_heads,
        #                                                   dim_feedforward=self.ff_size,
        #                                                   dropout=self.dropout,
        #                                                   activation=self.activation,
        #                                                   batch_first=True)
        # self.seqTransEncoder = nn.TransformerEncoder(seqTransEncoderLayer,
        #                                              num_layers=self.num_layers)

        seqTransDecoderLayer = nn.TransformerDecoderLayer(d_model=self.latent_dim,
                                                          nhead=self.num_heads,
                                                          dim_feedforward=self.ff_size,
                                                          dropout=self.dropout,
                                                          activation=self.activation,
                                                          batch_first=True)
        self.seqTransDecoder = nn.TransformerDecoder(seqTransDecoderLayer,
                                                     num_layers=self.num_layers)

        # Output Module
        self.out = zero_module(nn.Linear(self.latent_dim, self.input_feats))

        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, dropout=0)

        self.embed_timestep = TimestepEmbedder(self.latent_dim, self.sequence_pos_encoder)



    def forward(self, x, timesteps, z_x=None, z_m=None, relative=None):
        """
        x: B, T, D
        """
        B, T = x.shape[0], x.shape[1]

        emb = self.embed_timestep(timesteps) #+ self.text_embed(z_x)
        # time_emb = self.time_embed(timestep_embedding(timesteps, self.latent_dim)).unsqueeze(1)

        # relative = torch.cat([torch.zeros_like(relative)[...,0:1], relative], dim=-1)
        # relative[...,1] = torch.sin(relative[...,0])
        # relative[...,0] = torch.cos(relative[...,0])



        h = self.joint_embed(x)
        text_emb = self.text_embed(z_x)
        # relative_emb = self.relative_embed(relative.to(z_x.device)).reshape(B,1,-1)


        h = torch.cat([emb, h], dim=1)
        h = self.sequence_pos_encoder(h)

        # h = h * src_mask.unsqueeze(-1)
        # h = self.seqTransEncoder(h)#, src_key_padding_mask=~(src_mask>0.5))
        h = self.seqTransDecoder(h, memory=text_emb)#, src_key_padding_mask=~(src_mask>0.5))


        output = self.out(h).contiguous()[:,1:]
        return output




class MotionDiffusionDecoder(nn.Module):
    def __init__(self, nfeats, diffusion_steps, beta_scheduler, sampler, emb_type):
        super().__init__()
        self.nfeats = nfeats
        self.net = MotionDiffusionDecoderModel(nfeats, emb_type=emb_type)
        self.diffusion_steps = diffusion_steps
        betas = get_named_beta_schedule(beta_scheduler, self.diffusion_steps)
        steps=1000
        timestep_respacing=[steps]
        # timestep_respacing="ddim500"
        self.diffusion = MotionDiffusion(
            use_timesteps=space_timesteps(steps, timestep_respacing),
            betas=betas,
            # model_mean_type=ModelMeanType.EPSILON,
            model_mean_type=ModelMeanType.START_X,
            model_var_type=ModelVarType.FIXED_SMALL,
            loss_type=LossType.MSE,
            rescale_timesteps = False,
        )
        self.sampler = create_named_schedule_sampler(sampler, self.diffusion)

    def mask_cond(self, cond, cond_mask_prob = 0.1, force_mask=False):
        bs, t, d = cond.shape
        if force_mask:
            return torch.zeros_like(cond)
        elif cond_mask_prob > 0.:
            mask = torch.bernoulli(torch.ones(bs, device=cond.device) * cond_mask_prob).view(bs, 1, 1)  # 1-> use null_cond, 0-> use real cond
            return cond * (1. - mask)
        else:
            return cond

    def generate_src_mask(self, T, length):
        B, P = length.shape
        src_mask = torch.ones(B, T, P)
        for p in range(P):
            for i in range(B):
                for j in range(length[i][p], T):
                    src_mask[i, j, p] = 0
        return src_mask

    def compute_loss(self, batch):
        x_start = batch["motions"]
        z_m = batch["z_m"]
        z_x = batch["z_x"]
        relative = batch["relative"]
        # xf_out = batch["xf_out"]
        B,T = batch["motions"].shape[:2]

        # z_m = self.mask_cond(z_m, 0.1)
        # z_x = self.mask_cond(z_x, 0.1)
        # relative = self.mask_cond(relative.reshape(B,1,-1), 0.1)

        mask = self.generate_src_mask(T, batch["m_lens"]).to(x_start.device)

        t, _ = self.sampler.sample(B, x_start.device)
        # t[:] = 900
        output = self.diffusion.training_losses(
            model=self.net,
            x_start=x_start,
            t=t,
            mask=mask,
            model_kwargs={"z_m": z_m, "z_x": z_x, "relative":relative}
        )

        return output

    def forward(self, batch):
        z_m = batch["z_m"]
        z_x = batch["z_x"]
        relative = batch["relative"]
        # xf_out = batch["xf_out"]
        m_lens = batch["m_lens"]
        B = len(z_x)
        T = torch.max(m_lens)
        output = self.diffusion.p_sample_loop(
            self.net,
            (B, T, self.nfeats),
            clip_denoised=False,
            progress=True,
            model_kwargs={
                "z_m": z_m,
                "relative": relative,
                # 'length': m_lens,
                "z_x": z_x
            })
        return {"output":output}


class MotionDecoderModel(nn.Module):
    def __init__(self,
                 input_feats,
                 latent_dim=512,
                 num_frames=240,
                 ff_size=1024,
                 num_layers=8,
                 num_heads=16,
                 dropout=0,
                 activation="gelu",
                 emb_type="text",
                 **kargs):
        super().__init__()

        self.num_frames = num_frames
        self.latent_dim = latent_dim
        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.activation = activation
        self.input_feats = input_feats
        self.time_embed_dim = latent_dim
        self.emb_type = emb_type

        # Input Embedding
        self.semantic_embed = nn.Linear(self.latent_dim, self.latent_dim)
        self.motion_embed = nn.Linear(self.latent_dim, self.latent_dim)
        seqTransDecoderLayer = nn.TransformerDecoderLayer(d_model=self.latent_dim,
                                                          nhead=self.num_heads,
                                                          dim_feedforward=self.ff_size,
                                                          dropout=self.dropout,
                                                          activation=self.activation,
                                                          batch_first=True)
        self.seqTransDecoder = nn.TransformerDecoder(seqTransDecoderLayer,
                                                     num_layers=self.num_layers)

        # Output Module
        self.out = zero_module(nn.Linear(self.latent_dim, self.input_feats))

        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, dropout=0)

    def forward(self, z_m, T=600):
        """
        x: B, T, D
        """
        B = z_m.shape[0]

        #emb = torch.cat([z_x, z_m], dim=1)#self.semantic_embed(z_x).reshape(B,1,-1) + self.detail_embed(z_m).reshape(B,1,-1)

        # B, T, latent_dim
        h = torch.zeros([B,T,self.latent_dim]).float().to(z_m.device)
        # h = torch.cat([emb, h], dim=1)
        h = self.sequence_pos_encoder(h)


        # h = self.seqTransEncoder(h)#, src_key_padding_mask=~(src_mask>0.5))
        h = self.seqTransDecoder(h, memory=z_m)#, src_key_padding_mask=~(src_mask>0.5))


        output = self.out(h).contiguous()
        return output


class MotionDecoder(nn.Module):
    def __init__(self, nfeats, emb_type):
        super().__init__()
        self.nfeats = nfeats
        self.net = MotionDecoderModel(nfeats, emb_type=emb_type)
        self.mse_criterion = torch.nn.MSELoss(reduction='none')
        self.normalizer = MotionNormalizerTorch(dataset="single")


    def masked_mse(self, prediction, target, mask, mask2=None):
        # mask: B,T,1,1
        # mask2:B,T,1,N
        # data: B,T,1,D
        # data2:B,T,1,N,D
        loss = self.mse_criterion(prediction, target).mean(dim=-1, keepdim=True)
        if mask2 is not None:
            loss = (loss[...,0] * mask2).sum(dim=-1, keepdim=True) / (mask2.sum(dim=-1, keepdim=True)+1.e-7)
        loss = (loss * mask).sum() / (mask.sum()+1.e-7)
        return loss

    def mask_cond(self, cond, cond_mask_prob = 0.1, force_mask=False):
        bs, t, d = cond.shape
        if force_mask:
            return torch.zeros_like(cond)
        elif cond_mask_prob > 0.:
            mask = torch.bernoulli(torch.ones(bs, device=cond.device) * cond_mask_prob).view(bs, 1, 1)  # 1-> use null_cond, 0-> use real cond
            return cond * (1. - mask)
        else:
            return cond

    def recover_root_rot_pos(self, data):

        rot_vel = data[..., 0:1]
        r_rot_ang = torch.zeros_like(rot_vel).to(data.device)
        '''Get Y-axis rotation from rotation velocity'''
        r_rot_ang[:,1:] = rot_vel[:, :-1]
        r_rot_ang = torch.cumsum(r_rot_ang, dim=1)

        r_rot_quat = torch.zeros(data.shape[:-1] + (4,)).to(data.device)
        r_rot_quat[..., 0:1] = torch.cos(r_rot_ang)
        r_rot_quat[..., 2:3] = torch.sin(r_rot_ang)

        r_pos = torch.zeros(data.shape[:-1] + (3,)).to(data.device)
        r_pos[:, 1:, :, [0, 2]] = data[:, :-1, :, 1:3]
        '''Add Y-axis rotation to root position'''

        r_pos = qrot(qinv(r_rot_quat), r_pos)

        r_pos = torch.cumsum(r_pos, dim=1)

        r_pos[..., 1] = data[..., 3]
        return r_rot_ang, r_rot_quat, r_pos

    def recover_from_ric(self, data, joints_num, global_rt=None):
        _, r_rot_quat, r_pos = self.recover_root_rot_pos(data)
        positions = data[..., 4:(joints_num - 1) * 3 + 4]
        positions = positions.view(positions.shape[:-1] + (-1, 3))

        '''Add Y-axis rotation to local joints'''
        positions = qrot(qinv(r_rot_quat[..., None, :]).expand(positions.shape[:-1] + (4,)), positions)

        '''Add root XZ to joints'''
        positions[..., 0] += r_pos[..., 0:1]
        positions[..., 2] += r_pos[..., 2:3]

        '''Concate root and joints'''
        positions = torch.cat([r_pos.unsqueeze(-2), positions], dim=-2)

        if global_rt is not None:
            global_rot = global_rt[0][None]
            global_t = global_rt[1:3][None]

            global_r_rot_quat = torch.zeros(positions.shape[:-1] + (4,))
            global_r_rot_quat[..., 0] = torch.cos(global_rot)
            global_r_rot_quat[..., 2] = torch.sin(global_rot)

            positions = qrot(qinv(global_r_rot_quat), positions)
            positions[..., [0, 2]] += global_t

        return positions


    def compute_loss(self, batch):
        target = batch["motions"]
        z_m = batch["z_m"].unsqueeze(1)
        z_x = batch["z_x"]
        mask = batch["mask"]
        B, T = target.shape[:-1]

        # z_d = self.mask_cond(z_d, 0.1)

        prediction = self.net(z_m, T)

        target = target.reshape(B,T,-1,263)
        prediction = prediction.reshape(B,T,-1,263)
        mask = mask.reshape(B,T,-1,1)

        loss = 0
        losses = {}
        weights = {}
        weights["y_root_rvel"] = 1
        weights["xz_root_vel"] = 1
        weights["y_height"] = 1
        weights["joints"] = 1
        weights["poses"] = 1
        weights["vel"] = 1
        weights["contact"] = 1
        weights["g_y_rot"] = 20.
        weights["g_xz_trans"] = 5.
        weights["g_joints"] = 5.
        weights["contact_feet"] = 10.

        weights["relative_y_rot"] = 1
        weights["relative_xz_trans"] = 1




        target = self.normalizer.forward(target)
        # prediction = self.normalizer.forward(prediction)


        # losses["y_root_rvel"] = self.masked_mse(prediction[:,:,...,0:1], target[:,:,...,0:1], mask[:,:])*weights["y_root_rvel"]
        # losses["xz_root_vel"] = self.masked_mse(prediction[:,:,...,1:3], target[:,:,...,1:3], mask[:,:])*weights["xz_root_vel"]
        # losses["y_height"] = self.masked_mse(prediction[...,3:4], target[...,3:4], mask)*weights["y_height"]
        # losses["joints"] = self.masked_mse(prediction[...,4:4+21*3], target[...,4:4+21*3], mask)*weights["joints"]
        # losses["poses"] = self.masked_mse(prediction[...,4+21*3:4+21*3+21*6], target[...,4+21*3:4+21*3+21*6], mask)*weights["poses"]
        # losses["vel"] = self.masked_mse(prediction[...,4+21*3+21*6:4+21*3+21*6+22*3], target[...,4+21*3+21*6:4+21*3+21*6+22*3], mask)*weights["vel"]
        # losses["contact"] = self.masked_mse(prediction[...,4+21*3+21*6+22*3:4+21*3+21*6+22*3+4], target[...,4+21*3+21*6+22*3:4+21*3+21*6+22*3+4], mask)*weights["contact"]

        losses["motion"] = self.masked_mse(prediction, target,mask)
        # losses["relative_y_rot"] = self.masked_mse(prediction[:,0:1,...,0:1], target[:,0:1,...,0:1], mask[:,0:1])*weights["relative_y_rot"]
        # losses["relative_xz_trans"] = self.masked_mse(prediction[:,0:1,...,1:3], target[:,0:1,...,1:3], mask[:,0:1])*weights["relative_xz_trans"]

        # target = self.normalizer.backward(target)
        # prediction = self.normalizer.backward(prediction)
        #
        # pred_r_rot_angle, pred_r_rot_quat, pred_r_pos = self.recover_root_rot_pos(prediction)
        # tgt_r_rot_angle, tgt_r_rot_quat, tgt_r_pos = self.recover_root_rot_pos(target)
        # losses["g_y_rot"] = self.masked_mse(pred_r_rot_quat[...,[0,2]], tgt_r_rot_quat[...,[0,2]], mask) * weights["g_y_rot"]
        # losses["g_xz_trans"] = self.masked_mse(pred_r_pos[...,[0,2]], tgt_r_pos[...,[0,2]], mask) * weights["g_xz_trans"]
        #
        # pred_g_joints = self.recover_from_ric(prediction,joints_num=22).reshape(mask.shape[:-1]+(-1,))
        # tgt_g_joints = self.recover_from_ric(target,joints_num=22).reshape(mask.shape[:-1]+(-1,))
        # losses["g_joints"] = self.masked_mse(pred_g_joints, tgt_g_joints, mask) * weights["g_joints"]
        #
        # fids = [7, 10, 8, 11]
        # feet = []
        # for fid in fids:
        #     feet.append(pred_g_joints[..., 4+(fid-1)*3:4+fid*3][...,None,:])
        # feet = torch.cat(feet, dim=-2)
        #
        # feet_vel = feet[:,1:] - feet[:,:-1]
        # contact = target[:,:-1,:,-4:]
        # # print(contact)
        # losses["contact_feet"] = self.masked_mse(feet_vel, torch.zeros_like(feet_vel), mask[:,:-1], contact) * weights["contact_feet"]



        for term in losses.keys():
            loss += losses[term]
        losses["total"] = loss
        return losses


    def forward(self, batch):
        z_m = batch["z_m"]
        z_d = batch["z_d"]
        m_lens = batch["m_lens"]
        B = len(z_m)
        T = torch.max(m_lens)
        output = self.net(z_m, z_d, T)
        print(output)

        return {"output":output}

