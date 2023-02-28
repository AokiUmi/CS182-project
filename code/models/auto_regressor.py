from .transformer import *

from cv2 import norm
import torch
import torch.nn.functional as F
from torch import layer_norm, nn
import numpy as np
import clip
import models.mlp as mlp
from models.gru import TemporalEncoder
import math





class MotionAutoRegressor(nn.Module):
    def __init__(self,
                 input_feats,
                 num_frames=620,
                 latent_dim=512,
                 ff_size=1024,
                 num_layers=8,
                 num_heads=8,
                 dropout=0,
                 activation="gelu",
                 num_text_layers=4,
                 text_latent_dim=512,
                 text_ff_size=2048,
                 text_num_heads=4,
                 no_clip=False,
                 no_eff=False,
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
        self.time_embed_dim = latent_dim * 4
        self.sequence_embedding = nn.Parameter(torch.randn(num_frames, latent_dim))

        # Text Transformer
        self.clip, _ = clip.load('ViT-B/32', "cpu")
        if no_clip:
            self.clip.initialize_parameters()
        else:
            set_requires_grad(self.clip, False)
        if text_latent_dim != 512:
            self.text_pre_proj = nn.Linear(512, text_latent_dim)
        else:
            self.text_pre_proj = nn.Identity()
        textTransEncoderLayer = nn.TransformerEncoderLayer(
            d_model=text_latent_dim,
            nhead=text_num_heads,
            dim_feedforward=text_ff_size,
            dropout=dropout,
            activation=activation)
        self.textTransEncoder = nn.TransformerEncoder(
            textTransEncoderLayer,
            num_layers=num_text_layers)
        self.text_ln = nn.LayerNorm(text_latent_dim)
        self.text_proj = nn.Sequential(
            nn.Linear(text_latent_dim, self.time_embed_dim)
        )

        # Input Embedding
        self.joint_embed = nn.Linear(self.input_feats, self.latent_dim)
        # self.context_embed = nn.Linear(self.input_feats, self.latent_dim)

        self.time_embed = nn.Sequential(
            nn.Linear(self.latent_dim, self.time_embed_dim),
            nn.SiLU(),
            nn.Linear(self.time_embed_dim, self.time_embed_dim),
        )

        self.temporal_decoder_blocks = nn.ModuleList()
        for i in range(num_layers):
            if no_eff:
                self.temporal_decoder_blocks.append(
                    TemporalDiffusionTransformerDecoderLayer(
                        seq_len=num_frames,
                        latent_dim=latent_dim,
                        text_latent_dim=text_latent_dim,
                        time_embed_dim=self.time_embed_dim,
                        ffn_dim=ff_size,
                        num_head=num_heads,
                        dropout=dropout
                    )
                )
            else:
                self.temporal_decoder_blocks.append(
                    LinearTemporalDiffusionTransformerDecoderLayer(
                        seq_len=num_frames,
                        latent_dim=latent_dim,
                        text_latent_dim=text_latent_dim,
                        time_embed_dim=self.time_embed_dim,
                        ffn_dim=ff_size,
                        num_head=num_heads,
                        dropout=dropout
                    )
                )

        # Output Module
        self.out = zero_module(nn.Linear(self.latent_dim, self.input_feats))

    def encode_text(self, text, device):
        with torch.no_grad():
            text = clip.tokenize(text, truncate=True).to(device)
            x = self.clip.token_embedding(text).type(self.clip.dtype)  # [batch_size, n_ctx, d_model]

            x = x + self.clip.positional_embedding.type(self.clip.dtype)
            x = x.permute(1, 0, 2)  # NLD -> LND
            x = self.clip.transformer(x)
            x = self.clip.ln_final(x).type(self.clip.dtype)

        # T, B, D
        x = self.text_pre_proj(x)
        xf_out = self.textTransEncoder(x)
        xf_out = self.text_ln(xf_out)
        xf_proj = self.text_proj(xf_out[text.argmax(dim=-1), torch.arange(xf_out.shape[1])])
        # B, T, D
        xf_out = xf_out.permute(1, 0, 2)
        return xf_proj, xf_out

    def generate_src_mask(self, B, T, pads=None):

        src_mask = torch.ones(B, T)
        if pads is None:
            return src_mask
        for i in range(B):
            for j in range(0, pads[i]):
                src_mask[i, j] = 0
        return src_mask

    def forward(self, x, timesteps, context, frames, pads=None, length=None, text=None, xf_proj=None, xf_out=None):
        """
        x: B, T, D
        """
        B, cur_len = x.shape[0], x.shape[1]
        prev_len = context.shape[1]
        T = cur_len + prev_len

        if xf_proj is None or xf_out is None:
            xf_proj, xf_out = self.encode_text(text, x.device)

        emb = self.time_embed(timestep_embedding(timesteps, self.latent_dim)) + xf_proj

        if pads:
            seq_emb_list = []
            for pad in pads:
                seq_emb = self.sequence_embedding.unsqueeze(0)[:, :T - pad, :]
                seq_emb = torch.cat([torch.zeros([1, pad, seq_emb.shape[-1]], device=seq_emb.device), seq_emb], dim=1)
                seq_emb_list.append(seq_emb)
            seq_emb = torch.cat(seq_emb_list, dim=0)
        else:
            seq_emb = self.sequence_embedding.unsqueeze(0)[:, :T, :]

        # B, T, latent_dim
        h = torch.cat([context, x], dim=1)

        # ctx = self.context_embed(context)

        h = self.joint_embed(h)

        h = h + seq_emb

        src_mask = self.generate_src_mask(B, T, pads).to(x.device).unsqueeze(-1)

        for module in self.temporal_decoder_blocks:
            h = module(h, xf_out, emb, src_mask)

        output = self.out(h).view(B, T, -1).contiguous()[:, prev_len:]
        return output


class CrossContextTransformerDecoderLayer(nn.Module):

    def __init__(self,
                 seq_len=60,
                 latent_dim=32,
                 text_latent_dim=512,
                 time_embed_dim=128,
                 ffn_dim=256,
                 num_head=4,
                 dropout=0.1):
        super().__init__()
        self.sa_block = LinearTemporalSelfAttention(
            seq_len, latent_dim, num_head, dropout, time_embed_dim)
        self.ca_block1 = LinearTemporalCrossAttention(
            seq_len, latent_dim, latent_dim, num_head, dropout, time_embed_dim)
        self.ca_block2 = LinearTemporalCrossAttention(
            seq_len, latent_dim, text_latent_dim, num_head, dropout, time_embed_dim)
        self.ffn = FFN(latent_dim, ffn_dim, dropout, time_embed_dim)

    def forward(self, x, xf, ctx, emb, src_mask=None):
        x = self.sa_block(x, emb, src_mask)

        x = self.ca_block1(x, ctx, emb)

        x = self.ca_block2(x, xf, emb)

        x = self.ffn(x, emb)

        return x


class MotionAutoRegressorCrossContext(nn.Module):
    def __init__(self,
                 input_feats,
                 num_frames=620,
                 latent_dim=512,
                 ff_size=1024,
                 num_layers=8,
                 num_heads=8,
                 dropout=0,
                 activation="gelu",
                 num_text_layers=4,
                 text_latent_dim=512,
                 text_ff_size=2048,
                 text_num_heads=4,
                 no_clip=False,
                 no_eff=False,
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
        self.time_embed_dim = latent_dim * 4
        self.sequence_embedding = nn.Parameter(torch.randn(num_frames, latent_dim))

        # Text Transformer
        self.clip, _ = clip.load('ViT-B/32', "cpu")
        if no_clip:
            self.clip.initialize_parameters()
        else:
            set_requires_grad(self.clip, False)
        if text_latent_dim != 512:
            self.text_pre_proj = nn.Linear(512, text_latent_dim)
        else:
            self.text_pre_proj = nn.Identity()
        textTransEncoderLayer = nn.TransformerEncoderLayer(
            d_model=text_latent_dim,
            nhead=text_num_heads,
            dim_feedforward=text_ff_size,
            dropout=dropout,
            activation=activation)
        self.textTransEncoder = nn.TransformerEncoder(
            textTransEncoderLayer,
            num_layers=num_text_layers)
        self.text_ln = nn.LayerNorm(text_latent_dim)
        self.text_proj = nn.Sequential(
            nn.Linear(text_latent_dim, self.time_embed_dim)
        )

        # Input Embedding
        self.joint_embed = nn.Linear(self.input_feats, self.latent_dim)
        self.context_embed = nn.Linear(self.input_feats, self.latent_dim)

        self.time_embed = nn.Sequential(
            nn.Linear(self.latent_dim, self.time_embed_dim),
            nn.SiLU(),
            nn.Linear(self.time_embed_dim, self.time_embed_dim),
        )

        self.temporal_decoder_blocks = nn.ModuleList()
        for i in range(num_layers):
            if no_eff:
                self.temporal_decoder_blocks.append(
                    TemporalDiffusionTransformerDecoderLayer(
                        seq_len=num_frames,
                        latent_dim=latent_dim,
                        text_latent_dim=text_latent_dim,
                        time_embed_dim=self.time_embed_dim,
                        ffn_dim=ff_size,
                        num_head=num_heads,
                        dropout=dropout
                    )
                )
            else:
                self.temporal_decoder_blocks.append(
                    CrossContextTransformerDecoderLayer(
                        seq_len=num_frames,
                        latent_dim=latent_dim,
                        text_latent_dim=text_latent_dim,
                        time_embed_dim=self.time_embed_dim,
                        ffn_dim=ff_size,
                        num_head=num_heads,
                        dropout=dropout
                    )
                )

        # Output Module
        self.out = zero_module(nn.Linear(self.latent_dim, self.input_feats))

    def encode_text(self, text, device):
        with torch.no_grad():
            text = clip.tokenize(text, truncate=True).to(device)
            x = self.clip.token_embedding(text).type(self.clip.dtype)  # [batch_size, n_ctx, d_model]

            x = x + self.clip.positional_embedding.type(self.clip.dtype)
            x = x.permute(1, 0, 2)  # NLD -> LND
            x = self.clip.transformer(x)
            x = self.clip.ln_final(x).type(self.clip.dtype)

        # T, B, D
        x = self.text_pre_proj(x)
        xf_out = self.textTransEncoder(x)
        xf_out = self.text_ln(xf_out)
        xf_proj = self.text_proj(xf_out[text.argmax(dim=-1), torch.arange(xf_out.shape[1])])
        # B, T, D
        xf_out = xf_out.permute(1, 0, 2)
        return xf_proj, xf_out

    def generate_src_mask(self, B, T, pads=None):

        src_mask = torch.ones(B, T)
        if pads is None:
            return src_mask
        for i in range(B):
            for j in range(0, pads[i]):
                src_mask[i, j] = 0
        return src_mask

    def forward(self, x, timesteps, context, frames, pads=None, length=None, text=None, xf_proj=None, xf_out=None):
        """
        x: B, T, D
        """
        B, cur_len = x.shape[0], x.shape[1]
        prev_len = context.shape[1]
        T = cur_len + prev_len

        if xf_proj is None or xf_out is None:
            xf_proj, xf_out = self.encode_text(text, x.device)

        emb = self.time_embed(timestep_embedding(timesteps, self.latent_dim)) + xf_proj

        if pads:
            seq_emb_list = []
            for pad in pads:
                seq_emb = self.sequence_embedding.unsqueeze(0)[:, :T - pad, :]
                seq_emb = torch.cat([torch.zeros([1, pad, seq_emb.shape[-1]], device=seq_emb.device), seq_emb], dim=1)
                seq_emb_list.append(seq_emb)
            seq_emb = torch.cat(seq_emb_list, dim=0)
        else:
            seq_emb = self.sequence_embedding.unsqueeze(0)[:, :T, :]

        # B, T, latent_dim
        # h = torch.cat([context, x], dim=1)
        src_mask = self.generate_src_mask(B, prev_len, pads).to(x.device).unsqueeze(-1)

        ctx = self.context_embed(context) * src_mask

        ctx = ctx + seq_emb[:, :-cur_len, :]

        h = self.joint_embed(x)

        h = h + seq_emb[:, -cur_len:, :]

        # src_mask = self.generate_src_mask(B, prev_len, pads).to(x.device).unsqueeze(-1)

        for module in self.temporal_decoder_blocks:
            h = module(h, xf_out, ctx, emb)

        output = self.out(h).view(B, cur_len, -1).contiguous()
        return output