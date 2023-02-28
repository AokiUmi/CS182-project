import torch
from torch.utils import data
import numpy as np
import os
from os.path import join as pjoin
import random
import codecs as cs
import pickle as pkl

from tqdm import tqdm
from operator import itemgetter

from common.quaternion import *
from common.skeleton import Skeleton
from human_body_prior.body_model.body_model import BodyModel
import json
import glob
from utils.utils import *
from utils.plot_script import *
from utils.motion_process import *

trans_matrix = torch.Tensor([[1.0, 0.0, 0.0],
                         [0.0, 0.0, -1.0],
                         [0.0, 1.0, 0.0]])

ex_fps = 20

comp_device = torch.device("cpu")
# comp_device = torch.device("cuda:0")
neutral_bm_path = './body_models/smplh/neutral/model.npz'
bm = BodyModel(neutral_bm_path, num_betas=10).to(comp_device)

def plot_t2m(mp_joint, result_path, caption, relative=None):
    # joint = motion_temporal_filter(joint, sigma=1)
    plot_3d_motion(result_path, paramUtil.t2m_kinematic_chain, mp_joint, title=caption, fps=20)

def amass_to_pose(bdata):
    fps = 59.94
    frame_number = bdata['trans'].shape[0]

    pose_seq = []

    down_sample = int(fps / ex_fps)
    #     print(frame_number)
    #     print(fps)
    down_sample_frames = [int(i*59.94/20) for i in range(0,100000) if int(i*59.94/20)<frame_number]

    with torch.no_grad():
        for fId in down_sample_frames:
            root_orient = torch.Tensor(bdata['poses'][fId:fId + 1, :3]).to(
                comp_device)  # controls the global root orientation
            pose_body = torch.Tensor(bdata['poses'][fId:fId + 1, 3:66]).to(comp_device)  # controls the body
            betas = torch.Tensor(bdata['betas'][:10][np.newaxis]).to(comp_device)  # controls the body shape
            trans = torch.Tensor(bdata['trans'][fId:fId + 1]).to(comp_device)

            body = bm(pose_body=pose_body, betas=betas, root_orient=root_orient)
            joint_loc = body.Jtr[0][:22] + trans
            pose_seq.append(joint_loc.unsqueeze(0))
    down_sample_frame_num = len(pose_seq)
    pose_seq = torch.cat(pose_seq, dim=0).reshape(-1,3)

    positions = torch.mm(trans_matrix.to(comp_device).T, pose_seq.permute([1,0])).permute([1,0]).cpu().numpy().reshape(down_sample_frame_num, -1, 3)
    print(positions.shape)

    bdata['positions'] = positions
    return positions

import hashlib
def gen_md5(item):
    md5 = hashlib.md5()
    md5.update(item.encode("utf-8"))
    return md5.hexdigest()




class InterHumanDataset(data.Dataset):
    """Dataset for Text2Interaction generation task.

    """
    def __init__(self, opt, motion_type="rot"):
        self.opt = opt
        self.max_length = 200
        self.min_length = 40
        self.motion_type = motion_type

        self.normalizer = MotionNormalizer()

        joints_num = opt.joints_num

        data_list = []
        data_dict = {}

        motion_dict = {}

        for root, dirs, files in os.walk(opt.data_root):
            for file in files:
                if "pkl" in file:# and "16" in root:
                    params_file = pjoin(root, "smpl_params.pkl")
                    caption_file = pjoin(root, "annots.json")
                    params = pkl.load(open(params_file, "rb"))
                    caption = json.load(open(caption_file, "r", encoding="utf-8"))

                    if os.path.exists(pjoin(root, "joints.npz")):
                    # if 0:
                        print("existing joints.npz")
                        joints = np.load(pjoin(root, "joints.npz"))
                        if joints["person1"].shape[0]<self.min_length:
                            continue
                        params["person1"]["positions"] = joints["person1"]
                        params["person2"]["positions"] = joints["person2"]
                    else:
                        amass_to_pose(params["person1"])
                        amass_to_pose(params["person2"])

                        joints_file = pjoin(root, "joints.npz")

                        np.savez(joints_file, person1=params["person1"]["positions"],person2=params["person2"]["positions"])
                    data_dict[root] = params
                    invert_params = {}
                    invert_params["person1"] = params["person2"]
                    invert_params["person2"] = params["person1"]
                    data_dict[root + "_invert"] = invert_params
                    # plot_t2m([params["person1"]["positions"],params["person2"]["positions"]],
                    #          "results/" + caption_file.replace("/", "_").replace("\\", "_") + ".mp4", caption["summary"][0])
                    print(caption_file)
                    if opt.dataset_name == "single":
                        # --------------------Single person-----------------------
                        if "details" in caption.keys():
                            for item in caption["details"][0]["Person_1"]:
                                cur_caption = item["description"]
                                if item["end"]-item["start"]<self.min_length/20:
                                    continue
                                data_list.append({"root":root, "text":cur_caption, "interval":(item["start"], item["end"])})
                                try: motion_dict[gen_md5(cur_caption)] = np.load("../data/interhuman/motions/" + str(gen_md5(cur_caption)) + ".npz")
                                except: pass
                            if "Person_2" in caption["details"][0]:
                                for item in caption["details"][0]["Person_2"]:
                                    cur_caption = item["description"]
                                    if item["end"] - item["start"] < self.min_length/20:
                                        continue
                                    data_list.append({"root":root+"_invert", "text":cur_caption, "interval":(item["start"], item["end"])})
                                    try: motion_dict[gen_md5(cur_caption)] = np.load("../data/interhuman/motions/" + str(gen_md5(cur_caption)) + ".npz")
                                    except: pass

                    else:
                        # --------------------Interaction-----------------------
                        for item in caption["summary"]:
                            data_list.append({"root":root, "text":item, "interval":None})
                            try: motion_dict[gen_md5(cur_caption)] = np.load("../data/interhuman/motions/" + str(gen_md5(cur_caption)) + ".npz")
                            except: pass

                        if "details" in caption.keys():
                            for item in caption["details"][0]["Interaction"]:
                                cur_caption = item["description"]

                                if item["end"]-item["start"]<2:
                                    continue
                                data_list.append({"root":root, "text":cur_caption, "interval":(item["start"], item["end"])})
                                try: motion_dict[gen_md5(cur_caption)] = np.load("../data/interhuman/motions/" + str(gen_md5(cur_caption)) + ".npz")
                                except: pass

                        # # --------------------SWAP person1 person2-----------------------
                        # for item in caption["summary"]:
                        #     cur_caption = item.replace("1","3").replace("2","1").replace("3","2")
                        #     data_list.append({"root":root+"_invert", "text":cur_caption, "interval":None})
                        #     try:motion_dict[gen_md5(cur_caption)] = np.load("../data/interhuman/motions/" + str(gen_md5(cur_caption)) + ".npz")
                        #     except:pass
                        #
                        # if "details" in caption.keys():
                        #     for item in caption["details"][0]["Interaction"]:
                        #         cur_caption = item["description"].replace("1","3").replace("2","1").replace("3","2")
                        #         if item["end"]-item["start"]<2:
                        #             continue
                        #         data_list.append({"root":root+"_invert", "text":cur_caption, "interval":(item["start"], item["end"])})
                        #         try: motion_dict[gen_md5(cur_caption)] = np.load("../data/interhuman/motions/" + str(gen_md5(cur_caption)) + ".npz")
                        #         except: pass
        if self.opt.dataset_name == "single":
            # data_list = []
            joints_file_list = glob.glob("../data/HumanML3D/joints/*.npy")
            for joints_file in tqdm(joints_file_list):
                name = os.path.basename(joints_file).split(".")[0]
                try: joints = np.load(joints_file)[:,:22]
                except: continue
                # joints = None
                if len(joints.shape) != 3 or joints.shape[0]<40:
                    continue
                params = {"person1":{"positions":joints},"person2":{"positions":None}}
                data_dict[name] = params


                with cs.open(pjoin("../data/HumanML3D/texts", name + '.txt'), encoding="utf-8") as f:
                    for line in f.readlines():
                        line_split = line.strip().split('#')
                        caption = line_split[0]
                        # plot_t2m([params["person1"]["positions"]],
                        #          "results/" + caption[:20] + ".mp4", caption[:20])
                        data_list.append({"root": name, "text": caption, "interval": None})
                        try:
                            motion_dict[gen_md5(caption)] = np.load("../data/interhuman/motions/" + str(gen_md5(caption)) + ".npz")
                        except:
                            continue


        self.data_dict = data_dict
        self.motion_dict = motion_dict

        # random.seed(666)
        #
        # random.shuffle(self.data_list)
        # print(self.data_list[0]["text"])
        text_file = pjoin(self.opt.data_root, "texts.txt")
        root_file = pjoin(self.opt.data_root, "files.txt")
        with open(root_file, "w") as f:
            for i, data in enumerate(data_list):
                params = self.data_dict[data["root"]]
                caption = data['text'].replace("\n", "").replace("\n\r", "").replace("\r", "").strip().replace("person1", "person 1").replace("person2", "person 2")
                interval = data["interval"]
                joints_file = pjoin(self.opt.data_root, "joints",f"{i:05d}")
                # f.write(caption.lower()+"\n")
                f.write(data['root']+"\n")

                # if interval is not None:
                #     start = int(interval[0] * 20)
                #     end = int(interval[1] * 20) + 1
                #     position1 = params["person1"]["positions"]
                #     position2 = params["person2"]["positions"]
                #     position1 = position1[start:end]
                #     position2 = position2[start:end]
                # else:
                #     position1 = params["person1"]["positions"]
                #     position2 = params["person2"]["positions"]
                # np.savez(joints_file, person1=position1, person2=position2)

        self.data_list = []
        with open(pjoin(self.opt.data_root, "texts.txt")) as f:
            texts = f.readlines()
            print(len(texts))
        with open(pjoin(self.opt.data_root, "files.txt")) as f:
            roots = f.readlines()
        for root, dirs, files in os.walk(pjoin(opt.data_root, "joints")):
            for i, file in enumerate(files):
                joints = np.load(pjoin(root, file))
                # print(joints["person1"].shape)
                # print(texts[i])
                self.data_list.append({"root":str(i+1), "caption":texts[i], "positions1":joints["person1"], "positions2":joints["person2"]})

        # all_index = [str(i)+"\n" for i in range(len(self.data_list))]
        # random.shuffle(all_index)
        # train_set = all_index[len(self.data_list)//10:]
        # test_set = all_index[:len(self.data_list)//10]
        # with open(pjoin(self.opt.data_root, "train.txt"),"w") as f:
        #     f.writelines(train_set)
        # with open(pjoin(self.opt.data_root, "test.txt"),"w") as f:
        #     f.writelines(test_set)

        with open(pjoin(self.opt.data_root, "train.txt")) as f:
            train_index = [int(s.replace("\n", "")) for s in f.readlines()]
        with open(pjoin(self.opt.data_root, "test.txt")) as f:
            test_index = [int(s.replace("\n", "")) for s in f.readlines()]

        if self.opt.mode == "train":
            self.data_list = itemgetter(*train_index)(self.data_list)
        if self.opt.mode == "test":
            self.data_list = itemgetter(*test_index)(self.data_list)


        print("total dataset: ", len(self.data_list))
        count = 0
        for k, v in self.data_dict.items():
            count += v["person1"]["positions"].shape[0]
        print(count)
        self.global_rt = True if self.opt.dataset_name == "interaction" else False
        # self.calc_mean_std()

    def calc_mean_std(self):
        motions = []
        for data in tqdm(self.data_list):
            position1 = data["positions1"]
            motion1, root_quat_init1, root_pos_init1, global_positions1 = process_motion_np(position1, 0.002, self.global_rt)
            if np.isnan(motion1).sum() > 0.5:
                print("has nan")
                continue
            motions.append(motion1)


            position2 = data["positions2"]
            if position2 is None:
                continue
            motion2, root_quat_init2, root_pos_init2, global_positions2 = process_motion_np(position2, 0.002, self.global_rt)
            r_relative = qmul_np(root_quat_init2, qinv_np(root_quat_init1))
            angle = np.arctan2(r_relative[:, 2:3], r_relative[:, 0:1])
            xz = qrot_np(root_quat_init1, root_pos_init2 - root_pos_init1)[:, [0, 2]]
            relative = np.concatenate([angle, xz], axis=-1)[0]

            if self.global_rt:
                relative_rot = relative[0]
                relative_t = relative[1:3]

                relative_r_rot_quat = np.zeros(motion2.shape[:-1] + (4,))
                relative_r_rot_quat[..., 0] = np.cos(relative_rot)
                relative_r_rot_quat[..., 2] = np.sin(relative_rot)

                global_r_rot_quat = np.zeros(motion2.shape[:-1] + (4,))
                global_r_rot_quat[..., 0] = np.cos(motion2[..., 0])
                global_r_rot_quat[..., 2] = np.sin(motion2[..., 0])

                global_r_rot_quat = qmul_np(global_r_rot_quat, relative_r_rot_quat)
                motion2[..., 0:1] = np.arctan2(global_r_rot_quat[:, 2:3], global_r_rot_quat[:, 0:1])

                r_postions = np.concatenate([motion2[...,1:2], np.zeros_like(motion2[...,1:2]), motion2[...,2:3]], axis=-1)
                # print(r_postions.shape)

                r_postions = qrot_np(qinv_np(relative_r_rot_quat), r_postions)
                r_postions[..., [0, 2]] += relative_t
                motion2[..., 1:3] = r_postions[..., [0, 2]]
            if np.isnan(motion2).sum() > 0.5:
                print("has nan")
                continue
            motions.append(motion2)

        for data in tqdm(self.data_list):
            position1 = data["positions1"]
            position1 = swap_left_right(position1.copy())
            motion1, root_quat_init1, root_pos_init1, global_positions1 = process_motion_np(position1, 0.002, self.global_rt)
            if np.isnan(motion1).sum() > 0.5:
                print("has nan")
                continue
            motions.append(motion1)


            position2 = data["positions2"]
            position2 = swap_left_right(position2.copy())
            if position2 is None:
                continue
            motion2, root_quat_init2, root_pos_init2, global_positions2 = process_motion_np(position2, 0.002, self.global_rt)
            r_relative = qmul_np(root_quat_init2, qinv_np(root_quat_init1))
            angle = np.arctan2(r_relative[:, 2:3], r_relative[:, 0:1])
            xz = qrot_np(root_quat_init1, root_pos_init2 - root_pos_init1)[:, [0, 2]]
            relative = np.concatenate([angle, xz], axis=-1)[0]

            if self.global_rt:
                relative_rot = relative[0]
                relative_t = relative[1:3]

                relative_r_rot_quat = np.zeros(motion2.shape[:-1] + (4,))
                relative_r_rot_quat[..., 0] = np.cos(relative_rot)
                relative_r_rot_quat[..., 2] = np.sin(relative_rot)

                global_r_rot_quat = np.zeros(motion2.shape[:-1] + (4,))
                global_r_rot_quat[..., 0] = np.cos(motion2[..., 0])
                global_r_rot_quat[..., 2] = np.sin(motion2[..., 0])

                global_r_rot_quat = qmul_np(global_r_rot_quat, relative_r_rot_quat)
                motion2[..., 0:1] = np.arctan2(global_r_rot_quat[:, 2:3], global_r_rot_quat[:, 0:1])

                r_postions = np.concatenate([motion2[...,1:2], np.zeros_like(motion2[...,1:2]), motion2[...,2:3]], axis=-1)
                # print(r_postions.shape)

                r_postions = qrot_np(qinv_np(relative_r_rot_quat), r_postions)
                r_postions[..., [0, 2]] += relative_t
                motion2[..., 1:3] = r_postions[..., [0, 2]]
            if np.isnan(motion2).sum() > 0.5:
                print("has nan")
                continue
            motions.append(motion2)

        for data in tqdm(self.data_list):
            position1 = data["positions2"]
            motion1, root_quat_init1, root_pos_init1, global_positions1 = process_motion_np(position1, 0.002, self.global_rt)
            if np.isnan(motion1).sum() > 0.5:
                print("has nan")
                continue
            motions.append(motion1)


            position2 = data["positions1"]
            if position2 is None:
                continue
            motion2, root_quat_init2, root_pos_init2, global_positions2 = process_motion_np(position2, 0.002, self.global_rt)
            r_relative = qmul_np(root_quat_init2, qinv_np(root_quat_init1))
            angle = np.arctan2(r_relative[:, 2:3], r_relative[:, 0:1])
            xz = qrot_np(root_quat_init1, root_pos_init2 - root_pos_init1)[:, [0, 2]]
            relative = np.concatenate([angle, xz], axis=-1)[0]

            if self.global_rt:
                relative_rot = relative[0]
                relative_t = relative[1:3]

                relative_r_rot_quat = np.zeros(motion2.shape[:-1] + (4,))
                relative_r_rot_quat[..., 0] = np.cos(relative_rot)
                relative_r_rot_quat[..., 2] = np.sin(relative_rot)

                global_r_rot_quat = np.zeros(motion2.shape[:-1] + (4,))
                global_r_rot_quat[..., 0] = np.cos(motion2[..., 0])
                global_r_rot_quat[..., 2] = np.sin(motion2[..., 0])

                global_r_rot_quat = qmul_np(global_r_rot_quat, relative_r_rot_quat)
                motion2[..., 0:1] = np.arctan2(global_r_rot_quat[:, 2:3], global_r_rot_quat[:, 0:1])

                r_postions = np.concatenate([motion2[...,1:2], np.zeros_like(motion2[...,1:2]), motion2[...,2:3]], axis=-1)
                # print(r_postions.shape)

                r_postions = qrot_np(qinv_np(relative_r_rot_quat), r_postions)
                r_postions[..., [0, 2]] += relative_t
                motion2[..., 1:3] = r_postions[..., [0, 2]]
            if np.isnan(motion2).sum() > 0.5:
                print("has nan")
                continue
            motions.append(motion2)

        for data in tqdm(self.data_list):
            position1 = data["positions2"]
            position1 = swap_left_right(position1.copy())
            motion1, root_quat_init1, root_pos_init1, global_positions1 = process_motion_np(position1, 0.002, self.global_rt)
            if np.isnan(motion1).sum() > 0.5:
                print("has nan")
                continue
            motions.append(motion1)


            position2 = data["positions1"]
            position2 = swap_left_right(position2.copy())
            if position2 is None:
                continue
            motion2, root_quat_init2, root_pos_init2, global_positions2 = process_motion_np(position2, 0.002, self.global_rt)
            r_relative = qmul_np(root_quat_init2, qinv_np(root_quat_init1))
            angle = np.arctan2(r_relative[:, 2:3], r_relative[:, 0:1])
            xz = qrot_np(root_quat_init1, root_pos_init2 - root_pos_init1)[:, [0, 2]]
            relative = np.concatenate([angle, xz], axis=-1)[0]

            if self.global_rt:
                relative_rot = relative[0]
                relative_t = relative[1:3]

                relative_r_rot_quat = np.zeros(motion2.shape[:-1] + (4,))
                relative_r_rot_quat[..., 0] = np.cos(relative_rot)
                relative_r_rot_quat[..., 2] = np.sin(relative_rot)

                global_r_rot_quat = np.zeros(motion2.shape[:-1] + (4,))
                global_r_rot_quat[..., 0] = np.cos(motion2[..., 0])
                global_r_rot_quat[..., 2] = np.sin(motion2[..., 0])

                global_r_rot_quat = qmul_np(global_r_rot_quat, relative_r_rot_quat)
                motion2[..., 0:1] = np.arctan2(global_r_rot_quat[:, 2:3], global_r_rot_quat[:, 0:1])

                r_postions = np.concatenate([motion2[...,1:2], np.zeros_like(motion2[...,1:2]), motion2[...,2:3]], axis=-1)
                # print(r_postions.shape)

                r_postions = qrot_np(qinv_np(relative_r_rot_quat), r_postions)
                r_postions[..., [0, 2]] += relative_t
                motion2[..., 1:3] = r_postions[..., [0, 2]]
            if np.isnan(motion2).sum() > 0.5:
                print("has nan")
                continue
            motions.append(motion2)


        motions = torch.from_numpy(np.concatenate(motions, axis=0)).to(comp_device)


        inter_mean = torch.mean(motions,dim=0)
        inter_std = torch.std(motions,dim=0)

        np.save(pjoin(self.opt.data_root, 'inter_mean.npy'), inter_mean)
        np.save(pjoin(self.opt.data_root, 'inter_std.npy'), inter_std)

    def inv_transform(self, data):
        return data * self.std + self.mean

    def real_len(self):
        return len(self.data_list)


    def __len__(self):
        return self.real_len()*4


    def __getitem__(self, item):

        idx = item % self.real_len()
        data = self.data_list[idx]

        caption = data["caption"]
        positions1 = data["positions1"]
        positions2 = data["positions2"]

        if np.random.rand() > 0.5:
            positions1 = data["positions2"]
            positions2 = data["positions1"]
            caption = caption.replace("1","3").replace("2","1").replace("3","2")

        caption = caption.strip().replace("\n","").replace("practicing","performing")
        if np.random.rand() > 0.5:
            positions1 = swap_left_right(positions1.copy())
            positions2 = swap_left_right(positions2.copy())
            caption = caption.lower().replace("left","tmp").replace("right","left").replace("tmp","right")
            # plot_t2m([position1, position2],
            #          "1.mp4", "")

        length = positions1.shape[0]
        if length >= self.max_length+1:
            idx = random.randint(0, length - self.max_length+1)
            # idx = 0
            positions1 = positions1[idx:self.max_length+1 + idx]
            if positions2 is not None:
                positions2 = positions2[idx:self.max_length+1 + idx]

        relative = None
        hash_name =str(gen_md5(caption))

        # print("gggggggggg")
        # print(position1.shape)

        motion1, root_quat_init1, root_pos_init1, global_positions1 = process_motion_np(positions1, 0.002, self.global_rt)
        if positions2 is not None:
            motion2, root_quat_init2, root_pos_init2, global_positions2 = process_motion_np(positions2, 0.002, self.global_rt)
            r_relative = qmul_np(root_quat_init2, qinv_np(root_quat_init1))
            # print("r_relative ", r_relative)
            # angle = np.arcsin(r_relative[:, 2:3])
            # print("angle1 ", angle)
            angle = np.arctan2(r_relative[:, 2:3], r_relative[:, 0:1])
            # print("angle2 ", angle)
            xz = qrot_np(root_quat_init1, root_pos_init2 - root_pos_init1)[:, [0, 2]]
            relative = np.concatenate([angle, xz], axis=-1)[0]

            if self.global_rt:
                relative_rot = relative[0]
                relative_t = relative[1:3]

                relative_r_rot_quat = np.zeros(motion2.shape[:-1] + (4,))
                relative_r_rot_quat[..., 0] = np.cos(relative_rot)
                relative_r_rot_quat[..., 2] = np.sin(relative_rot)

                global_r_rot_quat = np.zeros(motion2.shape[:-1] + (4,))
                global_r_rot_quat[..., 0] = np.cos(motion2[..., 0])
                global_r_rot_quat[..., 2] = np.sin(motion2[..., 0])

                global_r_rot_quat = qmul_np(global_r_rot_quat, relative_r_rot_quat)
                motion2[..., 0:1] = np.arctan2(global_r_rot_quat[:, 2:3], global_r_rot_quat[:, 0:1])

                r_postions = np.concatenate([motion2[...,1:2], np.zeros_like(motion2[...,1:2]), motion2[...,2:3]], axis=-1)
                # print(r_postions.shape)

                r_postions = qrot_np(qinv_np(relative_r_rot_quat), r_postions)
                r_postions[..., [0, 2]] += relative_t
                motion2[..., 1:3] = r_postions[..., [0, 2]]
        else:
            motion2, root_quat_init2, root_pos_init2 = None, None, None
        motion_file = "../data/interhuman/motions/" + str(gen_md5(caption)) + ".npz"
        # np.savez(motion_file, motion1=motion1, motion2=motion2)



        if self.opt.dataset_name=="single":
            caption = caption.lower().replace("person 1", "the person").replace("person1", "the person").replace("person 2", "the person").replace("person2", "the person")
        else:
            if np.random.rand()>0:
                caption = caption.lower().replace("person 1 and person 2", "two people").replace("person 2 and person 1", "two people").replace("person 1", "the first person").replace("person1", "the first person").replace("person 2", "the second person").replace("person2", "the second person")
            else:
                caption = caption.lower().replace("person 1", "Adam").replace("person1", "Adam").replace(
                "person 2", "Eve").replace("person2", "Eve")

        m_length = len(motion1)
        if m_length < self.max_length:
            padding_len = self.max_length - m_length
            D = motion1.shape[1]
            padding_zeros = np.zeros((padding_len, D))
            motion1 = np.concatenate((motion1, padding_zeros), axis=0)
            if positions2 is not None:
                motion2 = np.concatenate((motion2, padding_zeros), axis=0)


        assert len(motion1) == self.max_length
        "Z Normalization"
        # motion1 = (motion1 - self.mean1) / self.std1
        if positions2 is not None:
            # motion2 = (motion2 - self.mean2) / self.std2
            m_length2 = m_length
        else:
            motion2 = np.zeros_like(motion1)
            m_length2 = 0

        motion = np.concatenate((motion1[:, None], motion2[:, None]), axis=1)
        m_length = np.array([m_length, m_length2])



        if self.opt.dataset_name == "single":
            # return caption, motion[:, 0], m_length[0:1]
            return data["root"],caption, motion[:, 0], m_length[0:1], np.zeros(3).astype(np.float32)
        else:
            # return caption, motion, m_length
            if relative is not None:
                return data["root"],caption, motion, m_length, relative
            else:
                return data["root"], caption, motion, m_length, np.zeros(3).astype(np.float32)


class Text2MotionDataset(data.Dataset):
    """Dataset for Text2Motion generation task.

    """
    def __init__(self, opt, mean, std, split_file, times=1, w_vectorizer=None, eval_mode=False):
        self.opt = opt
        self.max_length = 600
        self.times = times
        self.w_vectorizer = w_vectorizer
        self.eval_mode = eval_mode
        min_motion_len = 41 if self.opt.dataset_name =='t2m' else 24
        # min_motion_len = 40 if self.opt.dataset_name =='t2m' else 24

        joints_num = opt.joints_num

        data_dict = {}
        id_list = []
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())

        new_name_list = []
        length_list = []
        for name in tqdm(id_list):
            try:
                motion = np.load(pjoin(opt.motion_dir, name + '.npy'))

                if (len(motion)) < min_motion_len or (len(motion) >= self.max_length):
                    continue
                text_data = []
                flag = False
                with cs.open(pjoin(opt.text_dir, name + '.txt')) as f:

                    for line in f.readlines():
                        text_dict = {}
                        line_split = line.strip().split('#')
                        caption = line_split[0]
                        tokens = line_split[1].split(' ')
                        f_tag = float(line_split[2])
                        to_tag = float(line_split[3])
                        f_tag = 0.0 if np.isnan(f_tag) else f_tag
                        to_tag = 0.0 if np.isnan(to_tag) else to_tag

                        text_dict['caption'] = caption
                        text_dict['tokens'] = tokens
                        if f_tag == 0.0 and to_tag == 0.0:
                            flag = True
                            text_data.append(text_dict)
                        else:
                            n_motion = motion[int(f_tag*20) : int(to_tag*20)]
                            if (len(n_motion)) < min_motion_len or (len(n_motion) >= self.max_length):
                                continue
                            new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                            while new_name in data_dict:
                                new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                            data_dict[new_name] = {'motion': n_motion,
                                                    'length': len(n_motion),
                                                    'text':[text_dict]}
                            new_name_list.append(new_name)
                            length_list.append(len(n_motion))

                if flag:
                    data_dict[name] = {'motion': motion,
                                       'length': len(motion),
                                       'text':text_data}
                    new_name_list.append(name)
                    length_list.append(len(motion))
            except:
                # Some motion may not exist in KIT dataset
                pass


        name_list, length_list = zip(*sorted(zip(new_name_list, length_list), key=lambda x: x[1]))

        if opt.is_train:
            # root_rot_velocity (B, seq_len, 1)
            std[0:1] = std[0:1] / opt.feat_bias
            # root_linear_velocity (B, seq_len, 2)
            std[1:3] = std[1:3] / opt.feat_bias
            # root_y (B, seq_len, 1)
            std[3:4] = std[3:4] / opt.feat_bias
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
                                                              4 + (joints_num - 1) * 9 + joints_num * 3:] / opt.feat_bias

            assert 4 + (joints_num - 1) * 9 + joints_num * 3 + 4 == mean.shape[-1]
            np.save(pjoin(opt.meta_dir, 'mean.npy'), mean)
            np.save(pjoin(opt.meta_dir, 'std.npy'), std)

        self.mean = mean
        self.std = std
        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.name_list = name_list

    def inv_transform(self, data):
        return data * self.std + self.mean

    def real_len(self):
        return len(self.data_dict)

    def __len__(self):
        return self.real_len() * self.times

    def __getitem__(self, item):
        idx = item % self.real_len()
        data = self.data_dict[self.name_list[idx]]
        motion, m_length, text_list = data['motion'], data['length'], data['text']
        # Randomly select a caption
        text_data = random.choice(text_list)
        caption = text_data['caption']

        max_motion_length = self.opt.max_motion_length
        if m_length >= self.opt.max_motion_length:
            idx = random.randint(0, len(motion) - max_motion_length)
            motion = motion[idx: idx + max_motion_length]
        else:
            padding_len = max_motion_length - m_length
            D = motion.shape[1]
            padding_zeros = np.zeros((padding_len, D))
            motion = np.concatenate((motion, padding_zeros), axis=0)

        assert len(motion) == max_motion_length
        "Z Normalization"
        # motion = (motion - self.mean) / self.std


        if self.eval_mode:
            tokens = text_data['tokens']
            if len(tokens) < self.opt.max_text_len:
                # pad with "unk"
                tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
                sent_len = len(tokens)
                tokens = tokens + ['unk/OTHER'] * (self.opt.max_text_len + 2 - sent_len)
            else:
                # crop
                tokens = tokens[:self.opt.max_text_len]
                tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
                sent_len = len(tokens)
            pos_one_hots = []
            word_embeddings = []
            for token in tokens:
                word_emb, pos_oh = self.w_vectorizer[token]
                pos_one_hots.append(pos_oh[None, :])
                word_embeddings.append(word_emb[None, :])
            pos_one_hots = np.concatenate(pos_one_hots, axis=0)
            word_embeddings = np.concatenate(word_embeddings, axis=0)
            return word_embeddings, pos_one_hots, caption, sent_len, motion, m_length
        return "aaa", caption, motion, np.array(m_length)[None], ""
