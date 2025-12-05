import cv2 
import numpy as np 
import os 
import json
import time 
import pickle as pkl
import yaml
import copy
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from common_transforms import rotm_to_rot6d,cart2se3,vee_map,get_rel_command

class DataProcessorStateAction: 

    def __init__(self, cfg_path): 

        with open(cfg_path, 'r') as f:
            self.cfg = yaml.safe_load(f)
        self.src_dir = self.cfg["src_dir"]
        self.ep_paths = os.listdir(self.src_dir)
        self.data_paths = []
        for ep_path in self.ep_paths: 
            self.data_paths.append(os.path.join(self.src_dir, ep_path))

        self.cameras = self.cfg["cameras"]
        self.state_vars = self.cfg["state"]
    
    def proc_state(self, state): 
        
        trans = state[:3]
        ori = state[3:7]
        ori_rot6d = rotm_to_rot6d(R.from_quat(ori).as_matrix())
        proc_state = np.concatenate([trans, ori_rot6d])
        return proc_state 
    
    def proc_img(self, img, size=[256, 256], convert_color=True ): 
        
        img = cv2.resize(img, size)

        if convert_color:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        _, img_encoded = cv2.imencode('.jpg', img)
        return img_encoded
    
    def proc_action(self, action): 
        
        trans = action[:3]
        ori = action[3:6]
        gripper = action[-1]
        ori_rot6d = rotm_to_rot6d(R.from_rotvec(ori).as_matrix())
        proc_action = np.concatenate([trans, ori_rot6d, [gripper]])
        return proc_action 
    
    def proc_ep(self, ep): 

        ep_data = []
        ep_states = []
        ep_actions = []
        ep_len = len(ep["observations"])
        obs = ep["observations"]
        acs = ep["actions"]
        for t in range(ep_len): 
            obs_t = obs[t]
            state = []
            for var in self.state_vars:
                state.append(obs_t[var])
            state = np.concatenate(state)
            proc_state = self.proc_state(state)
            # print(f"Default state: {obs_t['robot0_eef_pos']} and processed_state: {proc_state}")
            proc_action = self.proc_action(acs[t])
            ep_states.append(proc_state)
            ep_actions.append(proc_action)
            ep_data.append([proc_state, proc_action, 0.0])
        return ep_states, ep_actions, ep_data
    
    def proc_dataset(self): 
        
        save_dir = self.cfg["save_dir"]
        os.makedirs(save_dir, exist_ok=True)

        norm_dict = {}
        dataset = []
        states = []
        actions = []
        start_time = time.time()

        for i in tqdm(range(len(self.data_paths))): 
            path = self.data_paths[i]
            # print(path)
            with open(path, "rb") as f: 
                ep = pkl.load(f)
                ep_states, ep_actions, ep_data = self.proc_ep(ep)
                dataset.append(ep_data)
                states.extend(ep_states)
                actions.extend(ep_actions)

        if self.cfg["normalize_state"]: 
            norm_dict["state_norm"] = self._max_min_norm(copy.deepcopy(states))
            state_norm = norm_dict["state_norm"]
            # print(f"Normalizing state: {state_norm}")
            for ep_idx, ep_data in enumerate(dataset): 
                for t in range(len(ep_data)): 
                    state, act, rew = ep_data[t]
                    # print(f'Current State: {obs["state"]}, mean: {np.array(norm_dict["state_norm"]["loc"])}')
                    state = (state - np.array(norm_dict["state_norm"]["loc"])) / np.array(norm_dict["state_norm"]["scale"])
                    
                    dataset[ep_idx][t] = (state, act, rew) 
        else: 
            norm_dict["state_norm"] = {"loc": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], "scale": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,1.0,1.0]}
        if self.cfg["normalize_action"]:
            norm_dict["action_norm"] = self._max_min_norm(actions)
        else:
            norm_dict["state_norm"] = {"loc": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], "scale": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,1.0,1.0, 1.0]}
        # norm_dict["action_norm"] = {"loc": 1.0, "scale": 1.0}
        with open(os.path.join(save_dir, "ac_norm.json"), "w") as f:
            json.dump(norm_dict, f)
        with open(os.path.join(save_dir, 'buf.pkl'), "wb") as f: 
            pkl.dump(dataset, f)
        print(f"Processing time {time.time() - start_time}") 
        print(f"Saved dataset at path: {save_dir}")

    def _gaussian_norm_state(self, states):
        all_acs_arr = np.array(states)
        # print(f"State shape:{all_acs_arr.shape}")
        mean = np.mean(all_acs_arr, axis=0)
        std =  np.std(all_acs_arr, axis=0)
        if not std.all(): # handle situation w/ all 0 actions
            std[std == 0] = 1e-17
        return dict(loc=mean.tolist(), scale=std.tolist())
    
    def _max_min_norm_state(self, all_acs):
        # print('Using max min norm')
        all_acs_arr = np.array(all_acs)
        max_ac = np.max(all_acs_arr, axis=0)
        min_ac = np.min(all_acs_arr, axis=0)

        mid = (max_ac + min_ac) / 2
        delta = (max_ac - min_ac) / 2

        for a in all_acs:
            a -= mid
            a /= delta
        return dict(loc=mid.tolist(), scale=delta.tolist())
    
    def _gaussian_norm(self, all_acs):
        all_acs_arr = np.array(all_acs)
        mean = np.mean(all_acs_arr, axis=0)
        std =  np.std(all_acs_arr, axis=0)


        for a in all_acs:
            a -= mean
            a /= std

        return dict(loc=mean.tolist(), scale=std.tolist())
    
    def _max_min_norm(self, all_acs):
        print('Using max min norm')
        all_acs_arr = np.array(all_acs)
        max_ac = np.max(all_acs_arr, axis=0)
        min_ac = np.min(all_acs_arr, axis=0)

        mid = (max_ac + min_ac) / 2
        delta = (max_ac - min_ac) / 2

        for a in all_acs:
            a -= mid
            a /= delta
        return dict(loc=mid.tolist(), scale=delta.tolist())

if __name__ == "__main__": 
    cfg_path = "/home/blank/dl_proj/EquiContact-Simulation/data_collection/datasets/stack/DiTEquiContact.yaml"
    dp = DataProcessorStateAction(cfg_path)
    dp.proc_dataset()