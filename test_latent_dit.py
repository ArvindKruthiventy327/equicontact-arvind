import argparse
from pathlib import Path
import time
import cv2
import hydra
import numpy as np
import torch
import torchvision
import yaml
import os
import sys
import json
import random
import pickle as pkl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from robobuf import ReplayBuffer as RB
from data4robotics.transforms import get_transform_by_name
import robosuite

import robosuite.macros as macros
# macros.IMAGE_CONVENTION = "opencv"
from robosuite.controllers import load_composite_controller_config

from scripted_policy.policy_player_stack import PolicyPlayerStack
from scripted_policy.policy_player_pih import PolicyPlayerPIH
from collections import deque
from scipy.spatial.transform import Rotation as R
from common_transforms import rotm_to_rot6d,rot6d_to_rotm, cart2se3,vee_map,get_rel_command

sys.path.append("/home/blank/cs282_project/latent_actions")
from mlp_ae import MLPVAE, MLP_VQVAE


class BaselineLatentPolicy: 

    def __init__(self, agent_path, model_name, autoencoder_path): 

        with open(Path(agent_path, "agent_config.yaml"), "r") as f:
            config_yaml = f.read()
            agent_config = yaml.safe_load(config_yaml)
        with open(Path(agent_path, "exp_config.yaml"), "r") as f:
            config_yaml = f.read()
            exp_config = yaml.safe_load(config_yaml)
            self.cam_idx = exp_config['params']['task']['train_buffer']['cam_indexes']
        with open(Path(agent_path, "ac_norm.json"), "r") as f: 
            self.norm_config = json.load(f)
        with open(autoencoder_path, 'r') as f:
            self.ae_config = yaml.safe_load(f)
        with open(self.ae_config["autoencoder"]["norm_path"], "r") as f: 
            self.ae_norm = json.load(f)
        agent = hydra.utils.instantiate(agent_config)
        agent = hydra.utils.instantiate(agent_config)
        save_dict = torch.load(Path(agent_path, model_name), map_location="cpu")
        agent.load_state_dict(save_dict['model'])
        self.agent = agent.eval().to("cuda:0")
        self.device = "cuda:0"
        self.transform = get_transform_by_name('preproc')
        self.buffer = deque([])
        self.last_ac = None
        self.load_autoencoder()
        self.ac_chunk = self.ae_config["autoencoder"]["ac_dim"][0] 
        self.model_params = self.ae_config["autoencoder"]

    def load_autoencoder(self): 
        type = self.ae_config["autoencoder"]["type"]
        model_params = self.ae_config["autoencoder"]
        ckpt_path = model_params["ckpt_path"]
        if type == "MLP_VAE": 
            obs_dim = model_params["obs_dim"]
            ac_dim = model_params["ac_dim"]
            latent_dim = model_params["latent_dim"]
            hidden = model_params["hidden"]
            self.model = MLPVAE(obs_dim, ac_dim, latent_dim, hidden)
        elif type == "MLP_VQVAE": 
            obs_dim = model_params["obs_dim"]
            ac_dim = model_params["ac_dim"]
            latent_dim = model_params["latent_dim"]
            hidden = model_params["hidden"]
            self.model = MLP_VQVAE(obs_dim, ac_dim, latent_dim, hidden)
        # elif type == "CNN_VAE":
        #     obs_dim = model_params["obs_dim"]
        #     ac_dim = model_params["ac_dim"]
        #     latent_dim = model_params["latent_dim"]
        #     channels = model_params["channels"]
        #     dec_in_padding = model_params["dec_in_padding"]
        #     dec_out_padding =  model_params["dec_out_padding"]
        #     self.model = CNN_VAE(ac_dim[0], obs_dim[0], ac_dim[-1], channels,dec_in_padding, dec_out_padding, latent_dim)
        state_dict = torch.load(ckpt_path)
        self.model.load_state_dict(state_dict["model_state_dict"])
        self.model.eval().cuda()
        self.enc = self.model.enc
        self.dec = self.model.dec
        if type == "CNN_VQVAE" or type == "MLP_VQVAE": 
            self.quantizer = self.model.quantizer()
        with open(model_params["norm_path"], "r") as f: 
            self.norm = json.load(f)
        print(self.norm)
    
    def _proc_image(self, rgb_img, size=(256, 256)):
        cam_idx = 0 
        imgs = {}
        for img in rgb_img:
            # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            rgb_tensor = torch.from_numpy(img).to(self.device)
            rgb_tensor = rgb_tensor.float().permute((2, 0, 1)) / 255
            rgb_tensor = torchvision.transforms.Resize(size, antialias=True)(rgb_tensor)
            rgb_tensor = rgb_tensor.unsqueeze(0)
            imgs[f"cam{cam_idx}"] = rgb_tensor.unsqueeze(0)
            cam_idx+=1
        return imgs
    
    def __proc_state(self, state): 
        # state = state.cpu().numpy().T.reshape((state.shape[1],))
        xyz_state = state[:3] # Extract the XYZ position
        quat = state[3:7] 
        # print(state)
        # quat = np.array([quat[1], quat[2], quat[3], quat[0]]).reshape((4,))
        # Extract the roll, pitch, yaw angles
        gripper_state = state[7:]
        rotm = R.from_quat(quat).as_matrix()
        rot6d_state = rotm_to_rot6d(rotm)
        processed_state = np.concatenate((xyz_state, rot6d_state), axis=0)
        processed_state = torch.from_numpy(processed_state).to(self.device)
        mean = torch.from_numpy(np.array(self.norm_config["state_norm"]["loc"])).float().to(self.device)
        std = torch.from_numpy(np.array(self.norm_config["state_norm"]["scale"])).float().to(self.device)
        # print(mean, std)
        state_norm = (processed_state-mean)/std
        # print(f"Normalized state: {state_norm}")
        return processed_state, state_norm
    
    def denorm_action(self, raw_action): 
        mean = torch.from_numpy(np.array(self.norm_config["action_norm"]["loc"])).float().to(self.device)
        std = torch.from_numpy(np.array(self.norm_config["action_norm"]["scale"])).float().to(self.device)
        action = (raw_action * std) + mean
        return action 
    
    def __proc_action(self, raw_action): 
        xyz = raw_action[:3].reshape((-1,))
        rot6d = raw_action[3:9]
        rotm = rot6d_to_rotm(rot6d)
        rotvec = R.from_matrix(rotm).as_rotvec().reshape((-1,))
        gripper =  raw_action[-1]
        proc_action = np.concatenate((xyz, rotvec, np.array([gripper])))
        return proc_action
    
    def forward_latent(self, img, obs):
        if len(self.buffer) == 0:
            img_proc = self._proc_image(img)
            raw_state, state = self.__proc_state(obs)
            raw_state = raw_state
            state = state.float().float().to(self.device)
            with torch.no_grad(): 
                ac = self.agent.get_actions(img_proc, state.unsqueeze(0))
            ac = ac.squeeze(0)
            print(ac.shape)
            state_mean = torch.from_numpy(np.array(self.ae_norm["state_norm"]["loc"])).to(self.device)
            state_std = torch.from_numpy(np.array(self.ae_norm["state_norm"]["scale"])).to(self.device)
            state_norm = (raw_state - state_mean )/state_std
            state_norm = state_norm.reshape((1, -1)).float()
            z_sample, mu, sigma = self.model.reparametrize(ac)
            x_hat = self.model.dec(state_norm, mu)
            x_hat = x_hat.detach().cpu().squeeze(0)
            obs_dim = self.model_params["obs_dim"]
            ac_dim = self.model_params["ac_dim"]
            obs_dim_flat = obs_dim[0] * obs_dim[1]
            ac_dim_flat = ac_dim[0] * ac_dim[1]
            ac_mean = np.array(self.ae_norm["action_norm"]["loc"])
            ac_std = np.array(self.ae_norm["action_norm"]["scale"])
            pred_ac_chunk = x_hat[obs_dim_flat:].reshape(ac_dim)
            pred_ac_chunk = pred_ac_chunk * ac_std + ac_mean
            actions = []
            for i in range(20):
                ac = self.__proc_action(pred_ac_chunk[i])
                actions.append(ac)
            print(pred_ac_chunk)
            self.buffer.extend(actions)
            return self.buffer.popleft()
        else: 
            return self.buffer.popleft()
    
    def forward_ensemble_latent(self, img, obs): 
        img_proc = self._proc_image(img)
        raw_state, state = self.__proc_state(obs)
        raw_state = raw_state
        state = state.float().float().to(self.device)
        with torch.no_grad(): 
            ac = self.agent.get_actions(img_proc, state.unsqueeze(0))
        ac = ac.squeeze(0)
        print(ac.shape)
        state_mean = torch.from_numpy(np.array(self.ae_norm["state_norm"]["loc"])).to(self.device)
        state_std = torch.from_numpy(np.array(self.ae_norm["state_norm"]["scale"])).to(self.device)
        state_norm = (raw_state - state_mean )/state_std
        state_norm = state_norm.reshape((1, -1)).float()
        z_sample, mu, sigma = self.model.reparametrize(ac)
        x_hat = self.model.dec(state_norm, mu)
        x_hat = x_hat.detach().cpu().squeeze(0)
        obs_dim = self.model_params["obs_dim"]
        ac_dim = self.model_params["ac_dim"]
        obs_dim_flat = obs_dim[0] * obs_dim[1]
        ac_dim_flat = ac_dim[0] * ac_dim[1]
        ac_mean = np.array(self.ae_norm["action_norm"]["loc"])
        ac_std = np.array(self.ae_norm["action_norm"]["scale"])
        pred_ac_chunk = x_hat[obs_dim_flat:].reshape(ac_dim)
        pred_ac_chunk = pred_ac_chunk * ac_std + ac_mean
        actions = []
        for i in range(pred_ac_chunk.shape[0]):
            ac = self.__proc_action(pred_ac_chunk[i])
            actions.append(ac)
        print(pred_ac_chunk)
        self.buffer.append(actions)
        num_actions = len(self.buffer)
        if num_actions >30: 
            self.buffer.popleft()
            num_actions = 30
            # num_actions=30

        print("Num actions:", num_actions)
        curr_act_preds = np.stack(
            [
                pred_actions[i]
                for (i, pred_actions) in zip(
                    range(num_actions - 1, -1, -1), self.buffer
                )
            ]
        )

        # more recent predictions get exponentially *less* weight than older predictions
        weights = np.exp(-0.05* np.arange(num_actions))
        weights = weights / weights.sum()
        # weights = weights[::-1]
        agg_ac = np.sum(weights[:, None] * curr_act_preds, axis=0)
        proc_action = self.__proc_action(agg_ac)
        return proc_action

    def forward(self, img, obs):
        if len(self.buffer) == 0:
            img_proc = self._proc_image(img)
            state = self.__proc_state(obs).float().to(self.device)
            # state = torch.from_numpy(state)[None].float().cuda()
            
            with torch.no_grad(): 
                ac = self.agent.get_actions(img_proc, state.unsqueeze(0))
            ac = self.denorm_action(ac)
            ac = ac.squeeze(0).cpu().numpy().astype(np.float32)
            proc_ac = []
            for i in range(ac.shape[0]): 
                proc_action = self.__proc_action(ac[i])
                proc_ac.append(proc_action)
            print(proc_ac[-1])
            self.buffer.extend(proc_ac[:18])
            if self.last_ac is None:
                proc_ac = self.buffer.popleft()
            else: 
                proc_ac = self.last_ac * 0.0 + 1.0 * self.buffer.popleft()
            self.last_ac = proc_ac
        else: 
            if self.last_ac is None:
                proc_ac = self.buffer.popleft()
            else: 
                proc_ac = self.last_ac * 0.0 + 1.0 * self.buffer.popleft()
            self.last_ac = proc_ac
        return proc_ac
    
    def forward_ensemble(self, img, obs): 
        img_proc = self._proc_image(img)
        state = self.__proc_state(obs).float().to(self.device)
        # state = torch.from_numpy(state)[None].float().cuda()
        
        with torch.no_grad(): 
            ac = self.agent.get_actions(img_proc, state.unsqueeze(0))
        ac = self.denorm_action(ac)
        ac = ac.squeeze(0).cpu().numpy().astype(np.float32)
        proc_ac = []
        # for i in range(ac.shape[0]): 
        #     proc_action = self.__proc_action(ac[i])
        #     proc_ac.append(proc_action)
        self.buffer.append(ac)

        # potentially consider not ensembling every timestep.

        # handle temporal blending
        num_actions = len(self.buffer)
        if num_actions >30: 
            self.buffer.popleft()
            num_actions = 30
            # num_actions=30

        print("Num actions:", num_actions)
        curr_act_preds = np.stack(
            [
                pred_actions[i]
                for (i, pred_actions) in zip(
                    range(num_actions - 1, -1, -1), self.buffer
                )
            ]
        )

        # more recent predictions get exponentially *less* weight than older predictions
        weights = np.exp(-0.05* np.arange(num_actions))
        weights = weights / weights.sum()
        # weights = weights[::-1]
        agg_ac = np.sum(weights[:, None] * curr_act_preds, axis=0)
        proc_action = self.__proc_action(agg_ac)
        return proc_action

def create_base_env(cfg_path): 
    with open(cfg_path, 'r') as f:
        config = yaml.safe_load(f)
    env_config = config['env_parameters']
    task_config = config['task_parameters']

    # setup the environment
    controller_config = load_composite_controller_config(robot=env_config['robots'][0], controller=env_config['controller'])
    env = robosuite.make(
    env_config['env_name'],
    robots=env_config['robots'][0],
    controller_configs=controller_config,   # arms controlled via OSC, other parts via JOINT_POSITION/JOINT_VELOCITY
    has_renderer=True,                      # on-screen rendering
    render_camera=None,              # visualize the "frontview" camera
    has_offscreen_renderer=True,           # no off-screen rendering                       
    horizon=10000,                            # each episode terminates after 200 steps
    use_object_obs=False,                   # no observations needed
    use_camera_obs=True,
    camera_names=env_config['camera_names'],
    camera_heights=env_config['camera_heights'],
    camera_widths = env_config['camera_widths'],
    camera_depths = env_config['camera_depths'],
    control_freq=env_config['control_freq'],                       # 20 hz control for applied actions
    fix_initial_cube_pose = env_config['fix_initial_cube_pose'],
    training=True
    )
    return env

def extract_policy_obs(obs, img_keys= ["agentview_image", "robot0_eye_in_hand_image"], state_keys=["robot0_eef_pos",
                                  "robot0_eef_quat_site"]): 
    imgs = []
    for camera in img_keys: 
        img_mod = obs[camera]
        imgs.append(img_mod)
    state = []
    for state_var in state_keys: 
        state.append(obs[state_var])
    return np.array(imgs), np.concatenate(state)

def convert_pose(pose, env): 
    robot0_base_body_id = env.sim.model.body_name2id("robot0_base")
    robot0_base_pos = env.sim.data.body_xpos[robot0_base_body_id]
    robot0_base_ori_rotm = env.sim.data.body_xmat[robot0_base_body_id].reshape((3,3))
    robot0_pos_world = pose[:3]
    robot0_rotm_world = R.from_quat(pose[3:7]).as_matrix()
    robot0_pos = robot0_base_ori_rotm.T @ (robot0_pos_world - robot0_base_pos)
    robot0_rotm = robot0_base_ori_rotm.T @ robot0_rotm_world
    return np.concatenate([robot0_pos, R.from_matrix(robot0_rotm).as_quat()])

def plot_video_sequence(image_list, frame_interval_ms=50):
    """
    Plots a sequence of images using Matplotlib's FuncAnimation.

    Args:
        image_list (list): A list of NumPy arrays, where each array is an image.
        frame_interval_ms (int): Delay between frames in milliseconds.
    """
    if not image_list:
        print("Error: The image list is empty.")
        return

    # 1. Setup the plot
    fig, ax = plt.subplots()
    
    # Initialize the image artist with the first frame
    # Use 'imshow' to display the image data
    first_image = image_list[0]
    im = ax.imshow(first_image)
    
    ax.set_title("Matplotlib Video Sequence")
    ax.axis('off') # Hide axes for a cleaner image display

    # 2. Update function for the animation
    def update(frame_index):
        """Updates the image data for the current frame."""
        # Set the image data for the imshow object
        im.set_array(image_list[frame_index])
        # Note: Must return an iterable of artists that were modified
        return [im]

    # 3. Create the animation
    # 'frames' is the number of times 'update' will be called (0 to N-1)
    # 'interval' is the delay between frames in milliseconds
    ani = animation.FuncAnimation(
        fig, 
        update, 
        frames=len(image_list),
        interval=frame_interval_ms,
        blit=True, # Optimized drawing (draws only what changed)
        repeat=True # Loop the animation
    )

    print(f"Starting animation with {len(image_list)} frames...")
    # Display the animation window
    plt.show()

def write_memory_images_to_video(image_list, output_name, fps=30):
    """
    input: image_list - A list of numpy arrays (or a 4D numpy array)
        Each image should be (Height, Width, 3)
    """
    
    if len(image_list) == 0:
        print("Error: Image list is empty.")
        return

    # 1. Get dimensions from the first image
    # Shape is typically (Height, Width, Channels)
    first_image = image_list[0]
    height, width, layers = first_image.shape

    # 2. Initialize VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'MJPG') #Codec
    video = cv2.VideoWriter(output_name, fourcc, fps, (width, height))
    
    print(f"Processing {len(image_list)} frames...")

    for img in image_list:
        # Check if resizing is needed (OpenCV requires strict size consistency)
        if img.shape[:2] != (height, width):
            img = cv2.resize(img, (width, height))

        # OPTIONAL: Convert RGB to BGR
        # Enable this line if your colors look wrong (blue faces, etc.)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        # Write the frame
        # Ensure data type is uint8 (0-255)
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
            
        video.write(img)

    video.release()
    print(f"Saved video: {output_name}")
if __name__ == "__main__": 
    import imageio
    # video_dir = "/home/blank/latent-diffusion-policy/inference/sim_videos"
    # os.makedirs(video_dir, exist_ok=True)
    # video_path = f"video_{0}.mp4"
    # video_writer = imageio.get_writer(video_path, fps=20)
    env = create_base_env("/home/blank/dl_proj/EquiContact-Simulation/config/train/DiT_stack_01.yaml")
    # agent = BaselinePolicy("/home/horowitz3/dit-policy/bc_finetune/test/wandb_None_equicontact_stacking_resnet_gn_nopool_2025-11-30_14-00-28", 
    #                        "test.ckpt")
    agent = BaselineLatentPolicy("/extra_storage/wandb_None_equicontact_latent_stacking_resnet_gn_nopool_2025-12-04_03-32-49", 
                           "test.ckpt", "/home/blank/cs282_project/latent_actions/latent_action_base.yaml")
    # agent = BaselinePolicy("/home/horowitz3/dit-policy/bc_finetune/test/wandb_None_equicontact_stacking_resnet_gn_nopool_2025-11-26_11-28-35", 
    #                        "test.ckpt")
    obs = env.reset()
    images = []
    timesteps = 1500
    action_buffer = []
    chunk = 30
    for i in range(timesteps): 
        # print(f"Action shape: {init_action.shape}")
        imgs, state = extract_policy_obs(obs, )
        state = convert_pose(state, env)
        images.append(obs["agentview_image"])
        # print(f"Current state: {state} at step:{i}")
        # ac = agent.forward_ensemble_latent(imgs, state)
        ac = agent.forward_latent(imgs, state)
    
        # if len(action_buffer) == 0:
        #     acs = agent.forward(imgs, state)
        #     action_buffer.extend(acs[:20])
        #     ac = action_buffer.pop(0)
        # else: 
        #     ac = action_buffer.pop(0)
        
        # acs = agent.forward(imgs, state)
        # action_buffer.extend(acs)
        # ac = acs[0]
        # acs = agent.forward(imgs, state)
        # # action_buffer.extend(acs)
        # ac = acs[0]
        if ac[-1] >= -0.6: 
            ac[-1] = 1
        elif ac[-1] < -0.6: 
            ac[-1] = -1
        print(f"Action : {ac} at step: {i}")
        obs, reward, done, info = env.step(ac)
        if done: 
            break
        env.render()
    print(done)
    write_memory_images_to_video(images, "fixed_demo_dit.avi")
    # plot_video_sequence(images)
