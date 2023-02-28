import os
import numpy as np
import torch
# import cv2
from PIL import Image
from utils import paramUtil
import math
import time
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from common.quaternion import *
from common.skeleton import Skeleton

face_joint_indx = [2,1,17,16]
fid_l = [7,10]
fid_r = [8,11]


kinematic_chain = [[0, 2, 5, 8, 11],
                 [0, 1, 4, 7, 10],
                 [0, 3, 6, 9, 12, 15],
                 [9, 14, 17, 19, 21],
                 [9, 13, 16, 18, 20]]

n_raw_offsets = torch.Tensor([[0,0,0],
                           [1,0,0],
                           [-1,0,0],
                           [0,1,0],
                           [0,-1,0],
                           [0,-1,0],
                           [0,1,0],
                           [0,-1,0],
                           [0,-1,0],
                           [0,1,0],
                           [0,0,1],
                           [0,0,1],
                           [0,1,0],
                           [1,0,0],
                           [-1,0,0],
                           [0,0,1],
                           [0,-1,0],
                           [0,-1,0],
                           [0,-1,0],
                           [0,-1,0],
                           [0,-1,0],
                           [0,-1,0]])

def swap_left_right(data):
    assert len(data.shape) == 3 and data.shape[-1] == 3
    data = data.copy()
    data[..., 0] *= -1
    right_chain = [2, 5, 8, 11, 14, 17, 19, 21]
    left_chain = [1, 4, 7, 10, 13, 16, 18, 20]
    left_hand_chain = [22, 23, 24, 34, 35, 36, 25, 26, 27, 31, 32, 33, 28, 29, 30]
    right_hand_chain = [43, 44, 45, 46, 47, 48, 40, 41, 42, 37, 38, 39, 49, 50, 51]
    tmp = data[:, right_chain]
    data[:, right_chain] = data[:, left_chain]
    data[:, left_chain] = tmp
    if data.shape[1] > 24:
        tmp = data[:, right_hand_chain]
        data[:, right_hand_chain] = data[:, left_hand_chain]
        data[:, left_hand_chain] = tmp
    return data

class MotionNormalizer():
    def __init__(self, dataset="single"):
        if dataset == "single" or dataset == "t2m":
            mean = np.load("mean1.npy")
            std = np.load("std1.npy")
        elif dataset == "interaction":
            mean = np.load("mean2.npy")
            std = np.load("std2.npy")

        feat_bias = 5

        joints_num = 22
        std[0:1] = std[0:1] / feat_bias
        # root_linear_velocity (B, seq_len, 2)
        std[1:3] = std[1:3] / feat_bias
        # root_y (B, seq_len, 1)
        std[3:4] = std[3:4] / feat_bias
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
                                                          4 + (joints_num - 1) * 9 + joints_num * 3:] / feat_bias

        self.local_motion_mean = mean
        self.local_motion_std = std

        self.rtvel_mean = mean[:3]
        self.rtvel_std = std[:3]

        inter_mean1 = np.load("../data/interhuman/inter_mean.npy")
        inter_std1 = np.load("../data/interhuman/inter_std.npy")
        self.inter_mean1 = inter_mean1[:3]
        self.inter_std1 = inter_std1[:3] / feat_bias

        inter_mean2 = np.load("../data/interhuman/inter_mean.npy")
        inter_std2 = np.load("../data/interhuman/inter_std.npy")
        self.inter_mean2 = inter_mean2[:3]
        self.inter_std2 = inter_std2[:3] / feat_bias

    def forward(self, x, global_rt=False):
        if not global_rt:
            x = (x - self.local_motion_mean) / self.local_motion_std
        else:
            x[...,3:] = (x[...,3:] - self.local_motion_mean[3:]) / self.local_motion_std[3:]
            x[..., 0, 0:3] = (x[..., 0, 0:3] - self.inter_mean1) / self.inter_std1
            x[..., 1, 0:3] = (x[..., 1, 0:3] - self.inter_mean2) / self.inter_std2
        # x[...,1:, 0:3] = (x[...,1:, 0:3] - self.rtvel_mean) / self.rtvel_std
        # if global_rt:
        #     x[...,0, 0:3] = (x[...,0, 0:3] - self.rt_mean) / self.rt_std
        # else:
        #     x[...,0, 0:3] = (x[..., 0, 0:3] - self.rtvel_mean) / self.rtvel_std
        return x

    def backward(self, x, global_rt=False):
        if not global_rt:
            x = x * self.local_motion_std + self.local_motion_mean
        else:
            x[...,3:] = x[...,3:]* self.local_motion_std[3:] + self.local_motion_mean[3:]
            x[..., 0, 0:3] = x[..., 0, 0:3] * self.inter_std1 + self.inter_mean1
            x[..., 1, 0:3] = x[..., 1, 0:3] * self.inter_std2 + self.inter_mean2

        # x[...,1:, 0:3] = x[...,1:, 0:3] * self.rtvel_std + self.rtvel_mean
        # if global_rt:
        #     x[...,0, 0:3] = x[...,0, 0:3]* self.rt_std + self.rt_mean
        # else:
        #     x[...,0, 0:3] = x[..., 0, 0:3]* self.rtvel_std + self.rtvel_mean
        return x



class MotionNormalizerTorch():
    def __init__(self, dataset="single"):
        if dataset == "single" or dataset == "t2m":
            mean = np.load("mean1.npy")
            std = np.load("std1.npy")
        elif dataset == "interaction":
            mean = np.load("mean2.npy")
            std = np.load("std2.npy")

        feat_bias = 5

        joints_num = 22
        std[0:1] = std[0:1] / feat_bias
        # root_linear_velocity (B, seq_len, 2)
        std[1:3] = std[1:3] / feat_bias
        # root_y (B, seq_len, 1)
        std[3:4] = std[3:4] / feat_bias
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
                                                          4 + (joints_num - 1) * 9 + joints_num * 3:] / feat_bias

        self.local_motion_mean = torch.from_numpy(mean).float()
        self.local_motion_std = torch.from_numpy(std).float()

        # self.rtvel_mean = torch.from_numpy(mean[:3])
        # self.rtvel_std =torch.from_numpy(std[:3])

        # self.rt_mean = torch.from_numpy(np.array([0, 0, 0]))
        # self.rt_std = torch.from_numpy(np.array([1.1921, 3.2690, 3.0617])) / feat_bias
        inter_mean1 = np.load("../data/interhuman/inter_mean.npy")
        inter_std1 = np.load("../data/interhuman/inter_std.npy")
        self.inter_mean1 = torch.from_numpy(inter_mean1)[:3]
        self.inter_std1 = torch.from_numpy(inter_std1)[:3] / feat_bias

        inter_mean2 = np.load("../data/interhuman/inter_mean.npy")
        inter_std2 = np.load("../data/interhuman/inter_std.npy")
        self.inter_mean2 = torch.from_numpy(inter_mean2)[:3]
        self.inter_std2 = torch.from_numpy(inter_std2)[:3] / feat_bias


    def forward(self, x, global_rt=False):
        device = x.device
        x = x.clone()
        if not global_rt:
            x = (x - self.local_motion_mean.to(device)) / self.local_motion_std.to(device)
        else:
            x[..., 3:] = (x[..., 3:] - self.local_motion_mean.to(device)[3:]) / self.local_motion_std.to(device)[3:]
            x[..., 0, 0:3] = (x[..., 0, 0:3] - self.inter_mean1.to(device)) / self.inter_std1.to(device)
            x[..., 1, 0:3] = (x[..., 1, 0:3] - self.inter_mean2.to(device)) / self.inter_std2.to(device)
        # y[...,1:, 0:3] = (x[...,1:, 0:3] - self.rtvel_mean.to(device)) / self.rtvel_std.to(device)
        # if global_rt:
        #     y[...,0, 0:3] = (x[...,0, 0:3] - self.rt_mean.to(device)) / self.rt_std.to(device)
        # else:
        #     y[...,0, 0:3] = (x[..., 0, 0:3] - self.rtvel_mean.to(device)) / self.rtvel_std.to(device)
        return x

    def backward(self, x, global_rt=False):
        device = x.device
        x = x.clone()
        if not global_rt:
            x = x * self.local_motion_std.to(device) + self.local_motion_mean.to(device)
        else:
            x[...,3:] = x[...,3:]* self.local_motion_std.to(device)[3:] + self.local_motion_mean.to(device)[3:]
            x[..., 0, 0:3] = x[..., 0, 0:3] * self.inter_std1.to(device) + self.inter_mean1.to(device)
            x[..., 1, 0:3] = x[..., 1, 0:3] * self.inter_std2.to(device) + self.inter_mean2.to(device)
        # y[...,1:, 0:3] = x[...,1:, 0:3] * self.rtvel_std.to(device) + self.rtvel_mean.to(device)
        # if global_rt:
        #     y[...,0, 0:3] = x[...,0, 0:3]* self.rt_std.to(device) + self.rt_mean.to(device)
        # else:
        #     y[...,0, 0:3] = x[..., 0, 0:3]* self.rtvel_std.to(device) + self.rtvel_mean.to(device)
        return x

from utils.paramUtil import *
from utils.quaternion import *

example_data = np.load('../data/HumanML3D/joints/000021.npy')
example_data = example_data.reshape(len(example_data), -1, 3)
example_data = torch.from_numpy(example_data)
tgt_skel = Skeleton(n_raw_offsets, kinematic_chain, 'cpu')
# (joints_num, 3)
tgt_offsets = tgt_skel.get_offsets_joints(example_data[0])


def uniform_skeleton(positions, target_offset):
    l_idx1, l_idx2 = 17, 18
    src_skel = Skeleton(n_raw_offsets, kinematic_chain, 'cpu')
    src_offset = src_skel.get_offsets_joints(torch.from_numpy(positions[0]))
    src_offset = src_offset.numpy()
    tgt_offset = target_offset.numpy()
    # print(src_offset)
    # print(tgt_offset)
    '''Calculate Scale Ratio as the ratio of legs'''
    src_leg_len = np.abs(src_offset[l_idx1]).max() + np.abs(src_offset[l_idx2]).max()
    tgt_leg_len = np.abs(tgt_offset[l_idx1]).max() + np.abs(tgt_offset[l_idx2]).max()

    scale_rt = tgt_leg_len / src_leg_len
    # print(scale_rt)
    src_root_pos = positions[:, 0]
    tgt_root_pos = src_root_pos * scale_rt

    '''Inverse Kinematics'''
    quat_params = src_skel.inverse_kinematics_np(positions, face_joint_indx)
    # print(quat_params.shape)

    '''Forward Kinematics'''
    src_skel.set_offset(target_offset)
    new_joints = src_skel.forward_kinematics_np(quat_params, tgt_root_pos)
    return new_joints

def process_motion_np(positions, feet_thre, global_rt=False):
    # (seq_len, joints_num, 3)
    #     '''Down Sample'''
    #     positions = positions[::ds_num]

    '''Uniform Skeleton'''
    positions = uniform_skeleton(positions, tgt_offsets)

    '''Put on Floor'''
    floor_height = positions.min(axis=0).min(axis=0)[1]
    positions[:, :, 1] -= floor_height
    #     print(floor_height)

    #     plot_3d_motion("./positions_1.mp4", kinematic_chain, positions, 'title', fps=20)

    '''XZ at origin'''
    root_pos_init = positions[0]
    root_pose_init_xz = root_pos_init[0] * np.array([1, 0, 1])
    positions = positions - root_pose_init_xz

    # '''Move the first pose to origin '''
    # root_pos_init = positions[0]
    # positions = positions - root_pos_init[0]

    '''All initially face Z+'''
    r_hip, l_hip, sdr_r, sdr_l = face_joint_indx
    across1 = root_pos_init[r_hip] - root_pos_init[l_hip]
    across2 = root_pos_init[sdr_r] - root_pos_init[sdr_l]
    across = across1 + across2
    across = across / np.sqrt((across ** 2).sum(axis=-1))[..., np.newaxis]

    # forward (3,), rotate around y-axis
    forward_init = np.cross(np.array([[0, 1, 0]]), across, axis=-1)
    # forward (3,)
    forward_init = forward_init / np.sqrt((forward_init ** 2).sum(axis=-1))[..., np.newaxis]

    #     print(forward_init)

    target = np.array([[0, 0, 1]])
    root_quat_init = qbetween_np(forward_init, target)
    root_quat_init_for_all = np.ones(positions.shape[:-1] + (4,)) * root_quat_init

    positions_b = positions.copy()

    positions = qrot_np(root_quat_init_for_all, positions)

    #     plot_3d_motion("./positions_2.mp4", kinematic_chain, positions, 'title', fps=20)

    '''New ground truth positions'''
    global_positions = positions.copy()

    # plt.plot(positions_b[:, 0, 0], positions_b[:, 0, 2], marker='*')
    # plt.plot(positions[:, 0, 0], positions[:, 0, 2], marker='o', color='r')
    # plt.xlabel('x')
    # plt.ylabel('z')
    # plt.axis('equal')
    # plt.show()

    """ Get Foot Contacts """

    def foot_detect(positions, thres):
        velfactor, heightfactor = np.array([thres, thres]), np.array([0.15, 0.1])

        feet_l_x = (positions[1:, fid_l, 0] - positions[:-1, fid_l, 0]) ** 2
        feet_l_y = (positions[1:, fid_l, 1] - positions[:-1, fid_l, 1]) ** 2
        feet_l_z = (positions[1:, fid_l, 2] - positions[:-1, fid_l, 2]) ** 2
        feet_l_h = positions[:-1,fid_l,1]
        feet_l = (((feet_l_x + feet_l_y + feet_l_z) < velfactor) & (feet_l_h < heightfactor)).astype(np.float)
        # feet_l = ((feet_l_x + feet_l_y + feet_l_z) < velfactor).astype(np.float32)

        feet_r_x = (positions[1:, fid_r, 0] - positions[:-1, fid_r, 0]) ** 2
        feet_r_y = (positions[1:, fid_r, 1] - positions[:-1, fid_r, 1]) ** 2
        feet_r_z = (positions[1:, fid_r, 2] - positions[:-1, fid_r, 2]) ** 2
        feet_r_h = positions[:-1,fid_r,1]
        feet_r = (((feet_r_x + feet_r_y + feet_r_z) < velfactor) & (feet_r_h < heightfactor)).astype(np.float)
        # feet_r = (((feet_r_x + feet_r_y + feet_r_z) < velfactor)).astype(np.float32)
        return feet_l, feet_r
    #
    feet_l, feet_r = foot_detect(positions, feet_thre)
    # feet_l, feet_r = foot_detect(positions, 0.002)

    '''Quaternion and Cartesian representation'''
    r_rot = None

    def get_rifke(positions):
        '''Local pose'''
        positions[..., 0] -= positions[:, 0:1, 0]
        positions[..., 2] -= positions[:, 0:1, 2]
        '''All pose face Z+'''
        positions = qrot_np(np.repeat(r_rot[:, None], positions.shape[1], axis=1), positions)
        return positions


    def get_cont6d_params(positions):
        T,N,D = positions.shape
        skel = Skeleton(n_raw_offsets, kinematic_chain, "cpu")
        # (seq_len, joints_num, 4)
        quat_params = skel.inverse_kinematics_np(positions, face_joint_indx, smooth_forward=True)

        '''Quaternion to continuous 6D'''
        cont_6d_params = quaternion_to_cont6d_np(quat_params)
        # (seq_len, 4)
        r_rot = quat_params[:, 0].copy()
        #     print(r_rot[0])
        '''Root Linear Velocity'''
        # (seq_len - 1, 3)
        velocity = (positions[1:, 0] - positions[:-1, 0]).copy()
        velocity = qrot_np(r_rot[1:], velocity)

        #     print(r_rot.shape, velocity.shape)
        '''Root Angular Velocity'''
        # (seq_len - 1, 4)
        r_velocity = qmul_np(r_rot[1:], qinv_np(r_rot[:-1]))
        # r_velocity = qmul_np(r_rot[:], qinv_np(np.ones([T,4])*r_rot[0:1]))

        # (seq_len, joints_num, 4)
        return cont_6d_params, r_velocity, velocity, r_rot

    cont_6d_params, r_velocity, velocity, r_rot = get_cont6d_params(positions)
    positions = get_rifke(positions)


    '''Root height'''
    root_y = positions[:, 0, 1:2]

    '''Root rotation and linear velocity'''
    # (seq_len-1, 1) rotation velocity along y-axis
    # (seq_len-1, 2) linear velovity on xz plane
    # r_velocity1 = np.arcsin(r_velocity[:,2:3])

    r_rot_y = np.arctan2(r_rot[:-1,2:3],r_rot[:-1,0:1])
    r_trans_xz = global_positions[:-1, 0, [0, 2]]
    # print("r_trans_xz",r_trans_xz.shape)
    r_velocity = np.arctan2(r_velocity[:,2:3],r_velocity[:,0:1])
    # print("r_velocity ", np.sum(r_velocity1-r_velocity))
    l_velocity = velocity[:, [0, 2]]
    # print("l_velocity",l_velocity.shape)
    #     print(r_velocity.shape, l_velocity.shape, root_y.shape)
    root_data = np.concatenate([r_velocity, l_velocity, root_y[:-1]], axis=-1)
    if global_rt:
        root_data = np.concatenate([r_rot_y, r_trans_xz, root_y[:-1]], axis=-1)

    '''Get Joint Rotation Representation'''
    # (seq_len, (joints_num-1) *6) quaternion for skeleton joints
    rot_data = cont_6d_params[:, 1:].reshape(len(cont_6d_params), -1)

    '''Get Joint Rotation Invariant Position Represention'''
    # (seq_len, (joints_num-1)*3) local joint position
    ric_data = positions[:, 1:].reshape(len(positions), -1)

    '''Get Joint Velocity Representation'''
    # (seq_len-1, joints_num*3)
    local_vel = qrot_np(np.repeat(r_rot[:-1, None], global_positions.shape[1], axis=1),
                        global_positions[1:] - global_positions[:-1])
    local_vel = local_vel.reshape(len(local_vel), -1)
    # local_vel = np.concatenate([local_vel[0:1], local_vel], axis=0)
    # feet_l = np.concatenate([feet_l[0:1], feet_l], axis=0)
    # feet_r = np.concatenate([feet_r[0:1], feet_r], axis=0)


    data = root_data
    data = np.concatenate([data, ric_data[:-1]], axis=-1)

    data = np.concatenate([data, rot_data[:-1]], axis=-1)

    #     print(data.shape, local_vel.shape)
    data = np.concatenate([data, local_vel], axis=-1)

    data = np.concatenate([data, feet_l, feet_r], axis=-1)


    return data, root_quat_init, root_pose_init_xz[None], global_positions





def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

COLORS = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
          [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
          [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

MISSING_VALUE = -1

def save_image(image_numpy, image_path):
    img_pil = Image.fromarray(image_numpy)
    img_pil.save(image_path)


def save_logfile(log_loss, save_path):
    with open(save_path, 'wt') as f:
        for k, v in log_loss.items():
            w_line = k
            for digit in v:
                w_line += ' %.3f' % digit
            f.write(w_line + '\n')


def print_current_loss(start_time, niter_state, losses, epoch=None, inner_iter=None, lr=None):

    def as_minutes(s):
        m = math.floor(s / 60)
        s -= m * 60
        return '%dm %ds' % (m, s)

    def time_since(since, percent):
        now = time.time()
        s = now - since
        es = s / percent
        rs = es - s
        return '%s (- %s)' % (as_minutes(s), as_minutes(rs))

    if epoch is not None and lr is not None :
        print('epoch: %3d niter:%6d inner_iter:%4d lr:%5f' % (epoch, niter_state, inner_iter, lr), end=" ")
    elif epoch is not None:
        print('epoch: %3d niter:%6d inner_iter:%4d' % (epoch, niter_state, inner_iter), end=" ")

    now = time.time()
    message = '%s'%(as_minutes(now - start_time))

    for k, v in losses.items():
        message += ' %s: %.4f ' % (k, v)
    print(message)


def compose_gif_img_list(img_list, fp_out, duration):
    img, *imgs = [Image.fromarray(np.array(image)) for image in img_list]
    img.save(fp=fp_out, format='GIF', append_images=imgs, optimize=False,
             save_all=True, loop=0, duration=duration)


def save_images(visuals, image_path):
    if not os.path.exists(image_path):
        os.makedirs(image_path)

    for i, (label, img_numpy) in enumerate(visuals.items()):
        img_name = '%d_%s.jpg' % (i, label)
        save_path = os.path.join(image_path, img_name)
        save_image(img_numpy, save_path)


def save_images_test(visuals, image_path, from_name, to_name):
    if not os.path.exists(image_path):
        os.makedirs(image_path)

    for i, (label, img_numpy) in enumerate(visuals.items()):
        img_name = "%s_%s_%s" % (from_name, to_name, label)
        save_path = os.path.join(image_path, img_name)
        save_image(img_numpy, save_path)


def compose_and_save_img(img_list, save_dir, img_name, col=4, row=1, img_size=(256, 200)):
    # print(col, row)
    compose_img = compose_image(img_list, col, row, img_size)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    img_path = os.path.join(save_dir, img_name)
    # print(img_path)
    compose_img.save(img_path)


def compose_image(img_list, col, row, img_size):
    to_image = Image.new('RGB', (col * img_size[0], row * img_size[1]))
    for y in range(0, row):
        for x in range(0, col):
            from_img = Image.fromarray(img_list[y * col + x])
            # print((x * img_size[0], y*img_size[1],
            #                           (x + 1) * img_size[0], (y + 1) * img_size[1]))
            paste_area = (x * img_size[0], y*img_size[1],
                                      (x + 1) * img_size[0], (y + 1) * img_size[1])
            to_image.paste(from_img, paste_area)
            # to_image[y*img_size[1]:(y + 1) * img_size[1], x * img_size[0] :(x + 1) * img_size[0]] = from_img
    return to_image


def list_cut_average(ll, intervals):
    if intervals == 1:
        return ll

    bins = math.ceil(len(ll) * 1.0 / intervals)
    ll_new = []
    for i in range(bins):
        l_low = intervals * i
        l_high = l_low + intervals
        l_high = l_high if l_high < len(ll) else len(ll)
        ll_new.append(np.mean(ll[l_low:l_high]))
    return ll_new


def motion_temporal_filter(motion, sigma=1):
    motion = motion.reshape(motion.shape[0], -1)
    # print(motion.shape)
    for i in range(motion.shape[1]):
        motion[:, i] = gaussian_filter(motion[:, i], sigma=sigma, mode="nearest")
    return motion.reshape(motion.shape[0], -1, 3)

