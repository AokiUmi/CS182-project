import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models import *
import clip

from models.gaussian_diffusion import (
    MotionDiffusion,
    SpacedDiffusion,
    space_timesteps,
    get_named_beta_schedule,
    create_named_schedule_sampler,
    ModelMeanType,
    ModelVarType,
    LossType
)

loss_ce = nn.CrossEntropyLoss()
loss_mse = nn.MSELoss(reduction='none')
loss_l1 = nn.L1Loss(reduction='none')

cosine_sim = nn.CosineSimilarity(dim=1, eps=1e-6)

def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad for all the networks.

    Args:
        nets (nn.Module | list[nn.Module]): A list of networks or a single
            network.
        requires_grad (bool): Whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad



from models.gaussian_diffusion import (
    SpacedDiffusion,)


class TransSampler(nn.Module):
    def __init__(self, latent_dim=512, num_frames=5000,
                 ff_size=1024, num_layers=8, num_heads=16, dropout=0.0,
                 ablation=None, activation="gelu"):
        super().__init__()

        self.latent_dim = latent_dim

        self.joint_embed = nn.Linear(self.latent_dim*2, self.latent_dim)
        self.z_x_embed = nn.Linear(self.latent_dim, self.latent_dim)
        self.text_embed = nn.Linear(self.latent_dim, self.latent_dim)

        # self.net = MLP(layers, skip_input_idx=0)

        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, dropout=0)

        self.embed_timestep = TimestepEmbedder(self.latent_dim, self.sequence_pos_encoder)

        seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                          nhead=num_heads,
                                                          dim_feedforward=ff_size,
                                                          dropout=0.1,
                                                          activation=activation,
                                                          batch_first=True)
        self.seqTransEncoder = nn.TransformerEncoder(seqTransEncoderLayer,
                                                     num_layers=num_layers)

    def forward(self, x, timesteps, z_x, text):
        """
        x: B, T, D
        """
        B, T = x.shape[0], x.shape[1]

        emb = self.embed_timestep(timesteps)
        emb_text = self.text_embed(text)
        emb_z_x = self.z_x_embed(z_x)
        x = self.joint_embed(x)

        h = torch.cat([x, emb, emb_z_x, emb_text], dim=1)
        h = self.sequence_pos_encoder(h)
        output = self.seqTransEncoder(h)[:,0:2]
        return output.reshape(B,1,-1)


class SemanticMotionSampler(nn.Module):
    def __init__(self, latent_dim, diffusion_steps, beta_scheduler, sampler):
        super().__init__()
        self.latent_dim = latent_dim
        self.net = TransSampler(512)
        self.diffusion_steps = diffusion_steps
        betas = get_named_beta_schedule(beta_scheduler, self.diffusion_steps)
        steps=1000
        timestep_respacing=[steps]
        # timestep_respacing="ddim500"
        self.diffusion = SpacedDiffusion(
            use_timesteps=space_timesteps(steps, timestep_respacing),
            betas=betas,
            model_mean_type=ModelMeanType.START_X,
            model_var_type=ModelVarType.FIXED_SMALL,
            loss_type=LossType.MSE,
            rescale_timesteps = False,
        )
        self.sampler = create_named_schedule_sampler(sampler, self.diffusion)

    def mask_cond(self, conds, cond_mask_prob = 0.1, force_mask=False):
        bs, t = conds[0].shape[:-1]
        return_conds = []
        mask = torch.bernoulli(torch.ones(bs, device=conds[0].device) * cond_mask_prob).view(bs, 1, 1)  # 1-> use null_cond, 0-> use real cond
        for cond in conds:
            if cond_mask_prob > 0.:
                return_conds.append(cond * (1. - mask))
            else:
                return_conds.append(cond)
        return return_conds

    def compute_loss(self, batch):
        B = len(batch["z_m"])
        z_m = batch["z_m"]
        z_d = batch["z_d"]*5
        x_start = torch.cat([z_m, z_d], dim=-1).unsqueeze(1)

        z_x = batch["z_x"]
        text = batch["text"]

        z_x, text = self.mask_cond([z_x, text], 0.1)

        t, _ = self.sampler.sample(B, z_m.device)
        output = self.diffusion.training_losses(
            model=self.net,
            x_start=x_start,
            t=t,
            model_kwargs={"z_x": z_x, "text":text},
        )

        return output["mse"].mean()

    def forward(self, batch):
        z_x = batch["z_x"]
        text = batch["text"]
        B = len(z_x)
        output = self.diffusion.p_sample_loop(
            self.net,
            (B, 1,self.latent_dim*2),
            clip_denoised=False,
            progress=False,
            model_kwargs={
                "z_x": z_x,
                "text": text,
            })
        z_m = output[..., :self.latent_dim]
        z_d = output[..., self.latent_dim:]/5
        return {"z_m":z_m, "z_d":z_d}



class MDAE(nn.Module):
    def __init__(self, input_feats, latent_dim, diffusion_steps=1000, sampler = 'uniform', beta_scheduler = 'cosine', frozen_encoder=False):
        super().__init__()

        # self.encoder = VariationalMotionEncoder(input_feats, latent_dim)
        self.encoder = MotionEncoder(input_feats, latent_dim)
        # self.decoder = MotionDiffusionDecoder(input_feats, diffusion_steps, beta_scheduler, sampler, emb_type="motion")
        self.decoder = MotionDecoder(input_feats, emb_type="motion")
        if frozen_encoder:
            set_requires_grad(self.encoder, False)
            set_requires_grad(self.decoder, False)

        self.diffusion_prior = SemanticMotionSampler(latent_dim, diffusion_steps, "cosine", sampler)

        self.latent_dim = latent_dim


        self.clip_model, _ = clip.load("ViT-B/32", device="cpu", jit=False)
        set_requires_grad(self.clip_model, False)
        # self.clip_model = kwargs['clip_model']
        self.clip_training = "text_"
        if self.clip_training and self.clip_model:
            self.clip_model.training = True
        else:
            if self.clip_model:
                assert self.clip_model.training == False  # make sure clip is frozen

        self.l1_criterion = torch.nn.L1Loss(reduction='mean')

    def compute_loss(self, batch):
        losses = {}
        losses["total"] = 0

        # compute clip losses
        batch = self.encode_text(batch)
        batch = self.encode_motion(batch)

        losses = self.decoder.compute_loss(batch)

        # losses.update(self.encoder.compute_loss(batch))
        # losses["total"] += losses["kld"]
        # losses["total"] += losses["rec"]
        # losses["total"] += losses["shrink"]

        mixed_clip_loss, clip_losses = self.compute_clip_losses(batch)
        losses.update(clip_losses)
        losses["total"] += mixed_clip_loss

        # loss = self.diffusion_prior.compute_loss(batch)
        # losses.update({"prior":loss})
        # losses["total"] += losses["prior"]


        return losses["total"], losses

    def compute_clip_losses(self, batch):
        mixed_clip_loss = 0.
        clip_losses = {}

        if 1:
            for d in self.clip_training.split('_')[:1]:
                if d == 'image':
                    features = self.clip_model.encode_image(
                        batch['clip_images']).float()  # preprocess is done in dataloader
                elif d == 'text':
                    features = batch['z_x'].squeeze(1)
                    z_m = batch['z_m'].squeeze(1)
                # normalized features
                features_norm = features / features.norm(dim=-1, keepdim=True)
                seq_motion_features_norm = z_m / z_m.norm(dim=-1, keepdim=True)
                # features_norm = features
                # seq_motion_features_norm = z_m


                cos = cosine_sim(features_norm, seq_motion_features_norm)
                cosine_loss = (1 - cos).mean()
                clip_losses[f'{d}_cosine'] = cosine_loss.item()
                mixed_clip_loss += cosine_loss

                # mse_clip_loss = loss_mse(features, batch["z_m"]).mean()
                # clip_losses[f'{d}_mse'] = mse_clip_loss.item()
                # mixed_clip_loss += mse_clip_loss


        return mixed_clip_loss, clip_losses




    @staticmethod
    def lengths_to_mask(lengths):
        max_len = max(lengths)
        if isinstance(max_len, torch.Tensor):
            max_len = max_len.item()

        index = torch.arange(max_len, device=lengths.device).expand(len(lengths), max_len)
        mask = index < lengths.unsqueeze(1)
        return mask

    def generate_src_mask(self, T, length):
        B = len(length)
        src_mask = torch.ones(B, T)
        for i in range(B):
            for j in range(length[i], T):
                src_mask[i, j] = 0
        return src_mask>0.5

    def encode_motion(self, batch):
        batch["mask"] = self.generate_src_mask(batch["motions"].shape[1], batch["m_lens"]).to(batch["motions"].device)
        # encode
        batch.update(self.encoder(batch))

        batch["z_m"] = batch["z_m"] / batch["z_m"].norm(dim=-1, keepdim=True) * 10


        # decode
        # batch.update(self.decoder(batch))

        return batch

    def encode_text(self, batch):
        raw_text = batch["caption"]
        device = next(self.parameters()).device

        # default_context_length = 77
        # context_length = 20 + 2 # start_token + 20 + end_token
        # assert context_length < default_context_length
        # texts = clip.tokenize(raw_text, context_length=context_length, truncate=True).to(device) # [bs, context_length] # if n_tokens > context_length -> will truncate
        #
        # zero_pad = torch.zeros([texts.shape[0], default_context_length-context_length], dtype=texts.dtype, device=texts.device)
        # texts = torch.cat([texts, zero_pad], dim=1)
        # batch["xf_out"] = self.clip_model.encode_text(texts).float().unsqueeze(1)

        texts = clip.tokenize(raw_text, truncate=True).to(device)
        tokens = self.clip_model.token_embedding(texts).type(self.clip_model.dtype)
        tokens = tokens + self.clip_model.positional_embedding.type(self.clip_model.dtype)
        batch["text"] = tokens
        # x = self.clip_model.token_embedding(texts).type(self.clip_model.dtype)  # [batch_size, n_ctx, d_model]
        # x = x + self.clip_model.positional_embedding.type(self.clip_model.dtype)
        # x = x.permute(1, 0, 2)  # NLD -> LND
        # batch['xf_out'] = self.clip_model.transformer(x).permute(1, 0, 2)
        batch["z_x"] = self.clip_model.encode_text(texts).float().unsqueeze(1)
        batch["z_x"] = batch["z_x"] / batch["z_x"].norm(dim=-1, keepdim=True) * 10

        # batch["z_m"] = batch["z_x"]
        # batch["z_m"] = batch["xf_out"]

        return batch

    def decode_motion(self, batch):
        # batch["mask"] = self.generate_src_mask(batch["motions"].shape[1], batch["m_lens"]).to(batch["motions"].device)
        # encode
        batch.update(self.decoder(batch))

        # batch["z_m"] = batch["mu"].unsqueeze(1)

        # decode
        # batch.update(self.decoder(batch))

        return batch

    def forward(self, batch):
        # batch["mask"] = self.generate_src_mask(batch["motions"].shape[1], batch["m_lens"]).to(batch["motions"].device)
        # encode
        batch.update(self.encode_text(batch))
        # batch.update(self.encode_motion(batch))
        batch["z_m"] = batch["z_x"]
        n_sample = 1
        z_m_list = []
        max_norm = 0
        for i in range(n_sample):
            batch.update(self.diffusion_prior(batch))
            batch["z_m"] = batch["z_m"] / batch["z_m"].norm(dim=-1, keepdim=True) * 10
            z_m_list.append(batch["z_m"])


        for i in range(n_sample):
            print((z_m_list[i] * batch["z_x"]).norm())
            if (z_m_list[i]*batch["z_x"]).norm() > max_norm:
                batch["z_m"] = z_m_list[i]

        batch["relative"] = batch["z_m"]
        print(batch["z_d"].norm(dim=-1, keepdim=True))
        batch.update(self.decode_motion(batch))


        return batch