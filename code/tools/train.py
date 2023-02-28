import sys
sys.path.append(sys.path[0]+r"/../")

import os
from os.path import join as pjoin
import numpy as np

import utils.paramUtil as paramUtil
from options.train_options import TrainCompOptions

from models import *
from trainers import Trainer

from datasets import Text2MotionDataset
from datasets import InterHumanDataset

from mmcv.runner import get_dist_info, init_dist
from mmcv.parallel import MMDistributedDataParallel
import torch
import torch.distributed as dist


def build_models(opt, dim_pose):
    if opt.model=="motionclip":
        model = MOTIONCLIP(
            input_feats=dim_pose,
            latent_dim=opt.latent_dim,
            emb_type=opt.emb_type,
            finetune=opt.finetune)
    elif opt.model=="MDAE":
        model = MDAE(
            input_feats=dim_pose,
            latent_dim=opt.latent_dim,)
        print(model)
    return model



if __name__ == '__main__':
    parser = TrainCompOptions()
    opt = parser.parse()
    rank, world_size = get_dist_info()

    opt.device = torch.device("cuda")
    torch.autograd.set_detect_anomaly(True)

    opt.save_root = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name)
    opt.model_dir = pjoin(opt.save_root, opt.emb_type, 'model')
    opt.meta_dir = pjoin(opt.save_root, opt.emb_type, 'meta')
    opt.log_dir = pjoin(opt.save_root, opt.emb_type, 'log')
    opt.mode = "train"

    if rank == 0:
        os.makedirs(opt.model_dir, exist_ok=True)
        os.makedirs(opt.meta_dir, exist_ok=True)
    if world_size > 1:
        dist.barrier()

    if opt.dataset_name == 't2m':
        opt.data_root = '../data/HumanML3D'
        opt.motion_dir = pjoin(opt.data_root, 'new_joint_vecs')
        opt.text_dir = pjoin(opt.data_root, 'texts')
        opt.joints_num = 22
        radius = 4
        fps = 20
        opt.max_motion_length = 196
        dim_pose = 263
        kinematic_chain = paramUtil.t2m_kinematic_chain
    elif opt.dataset_name == 'interaction':
        opt.data_root = r'../data/interhuman'
        opt.motion_dir = pjoin(opt.data_root, 'new_joint_vecs')
        opt.text_dir = pjoin(opt.data_root, 'texts')
        opt.joints_num = 22
        radius = 4
        fps = 20
        opt.max_motion_length = 196
        dim_pose = 263*2
        kinematic_chain = paramUtil.t2m_kinematic_chain
    elif opt.dataset_name == 'single':
        opt.data_root = r'../data/interhuman'
        opt.motion_dir = pjoin(opt.data_root, 'new_joint_vecs')
        opt.text_dir = pjoin(opt.data_root, 'texts')
        opt.joints_num = 22
        radius = 4
        fps = 20
        opt.max_motion_length = 196
        dim_pose = 263
        kinematic_chain = paramUtil.t2m_kinematic_chain
    elif opt.dataset_name == 'kit':
        opt.data_root = './data/KIT-ML'
        opt.motion_dir = pjoin(opt.data_root, 'new_joint_vecs')
        opt.text_dir = pjoin(opt.data_root, 'texts')
        opt.joints_num = 21
        radius = 240 * 8
        fps = 12.5
        dim_pose = 251
        opt.max_motion_length = 196
        kinematic_chain = paramUtil.kit_kinematic_chain

    else:
        raise KeyError('Dataset Does Not Exist')

    dim_word = 300


    train_split_file = pjoin(opt.data_root, 'train.txt')

    model = build_models(opt, dim_pose)
    if world_size > 1:
        encoder = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=True)
    else:
        model = model.cuda()

    trainer = Trainer(opt, model)
    if opt.dataset_name == "interaction":
        train_dataset = InterHumanDataset(opt)
    elif opt.dataset_name == "single":
        train_dataset = InterHumanDataset(opt)
    else:
        mean = np.load(pjoin(opt.data_root, 'mean.npy'))
        std = np.load(pjoin(opt.data_root, 'std.npy'))
        train_dataset = Text2MotionDataset(opt, mean, std, train_split_file)
    trainer.train(train_dataset)
