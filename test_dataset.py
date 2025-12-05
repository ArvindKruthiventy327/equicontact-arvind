import argparse
from pathlib import Path
import time
import cv2
# import h5py
# import hydra
import numpy as np
# import torch
# import torchvision
import yaml
import os
import json
import random
# import torch
import pickle as pkl
import matplotlib.pyplot as plt
import matplotlib.animation as animation 
# from robobuf import ReplayBuffer as RB
# from data4robotics.transforms import get_transform_by_name
# from mimicgen.envs.robosuite import three_piece_assembly
# import mimicgen.utils.robomimic_utils as RoboMimicUtils
# import robomimic.utils.file_utils as FileUtils

from scipy.spatial.transform import Rotation as R
import robosuite

import robosuite.macros as macros
# macros.IMAGE_CONVENTION = "opencv"
from robosuite.controllers import load_composite_controller_config

from scripted_policy.policy_player_stack import PolicyPlayerStack
from common_transforms import rotm_to_rot6d,rot6d_to_rotvec, cart2se3,vee_map,get_rel_command

# from data_processing.common_transforms import rotm_to_rot6d,rot6d_to_rotvec, cart2se3,vee_map,get_rel_command
def plot_video_sequence(image_list, frame_interval_ms=2):
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
        repeat=False # Loop the animation
    )

    print(f"Starting animation with {len(image_list)} frames...")
    # Display the animation window
    plt.show()

def read_demo(dir, demo=1): 
    data_file = os.path.join(dir, "buf.pkl")
    stats_file = os.path.join(dir, "ac_norm.json")
    actions = []
    with open(stats_file, "r") as s: 
        stats = json.load(s)
        ac_mean = stats["action_norm"]["loc"]
        ac_std = stats["action_norm"]["scale"]
        state_mean = stats["state_norm"]["loc"]
        state_std = stats["state_norm"]["scale"]
    states = []
    images = []
    with open(data_file, "rb") as f: 
        dit_data = pkl.load(f)
        for t in range(len(dit_data[demo])): 
            state_t = (dit_data[demo][t][0]["state"])
            agentview_img = cv2.imdecode(dit_data[demo][t][0]["enc_cam_0"], cv2.IMREAD_COLOR)
            images.append(agentview_img)
            # print(f"Action at time {t}: {dit_data[demo][t][1]}")
            action_t = (dit_data[demo][t][1] * ac_std) + ac_mean 
            # action_t = dit_data[demo][t][1]
            # print(f"State at time {t} : {state_t.shape}")
            # print(f"Action at time {t} : {action_t.shape}")
            actions.append(action_t)
            states.append(state_t)
    return np.array(actions), np.array(states), images, [ac_mean, ac_std, state_mean, state_std]

def initialize_env(cfg_path): 
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
    horizon=env_config['max_iter'],                            # each episode terminates after 200 steps
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

def proc_state(state, state_mean, state_std): 
        # state = state.cpu().numpy().T.reshape((state.shape[1],))
        state = np.concatenate(state)
        xyz_state = state[:3] # Extract the XYZ position
        quat = state[3:7] 
        # quat = np.array([quat[1], quat[2], quat[3], quat[0]]).reshape((4,))
        # Extract the roll, pitch, yaw angles
        gripper_state = state[7:]
        rotm = R.from_quat(quat).as_matrix()
        rot6d_state = rotm_to_rot6d(rotm)
        processed_state = np.concatenate((xyz_state, rot6d_state, gripper_state), axis=0)
        processed_state = processed_state
        mean = np.array(state_mean)
        std = np.array(state_std)
        # print(mean, std)
        # print(processed_state - mean, std)
        state_norm = (processed_state-mean)/std
        return state_norm, processed_state

def convert_pose(pose, env): 
    robot0_base_body_id = env.sim.model.body_name2id("robot0_base")
    robot0_base_pos = env.sim.data.body_xpos[robot0_base_body_id]
    robot0_base_ori_rotm = env.sim.data.body_xmat[robot0_base_body_id].reshape((3,3))
    robot0_pos_world = pose[:3]
    robot0_rotm_world = R.from_quat(pose[3:7]).as_matrix()
    robot0_pos = robot0_base_ori_rotm.T @ (robot0_pos_world - robot0_base_pos)
    robot0_rotm = robot0_base_ori_rotm.T @ robot0_rotm_world
    return [robot0_pos, R.from_matrix(robot0_rotm).as_quat()]

def proc_data_action(ac): 
    trans = ac[:3]
    ori_rot6d = ac[3:9]
    gripper = ac[-1]
    ori_rotvec = rot6d_to_rotvec(ori_rot6d)
    return np.concatenate([trans, ori_rotvec, [gripper]])
    # setup the scripted policy
    # if env_config['env_name'] == "Stack" or env_config['env_name'] == "StackCustom":
    #     player = PolicyPlayerStack(env, render = False, randomized = True, debug = False, save = False)

    # elif env_config['env_name'] == "PegInHole":
    #     player = PolicyPlayerPIH(env, render = False, randomized = False, debug = True, save = False)
    # else:
    #     raise NotImplementedError("Selected environment is not implemented for scripted policy")

if __name__ == "__main__":
    env = initialize_env("/home/blank/dl_proj/EquiContact-Simulation/config/train/DiT_stack_02.yaml")
    images = []
    for j in range(100):
        actions, states, images, norm_list = read_demo("/extra_storage/equicontact_stacking_small_random", demo = j)
        print(f"Episode: {j}")
        # plot_video_sequence(images)
        obs = env.reset()
        for i in range(actions.shape[0]):
            curr_state = []
            
            curr_state.append(obs["robot0_eef_pos"])
            curr_state.append(obs["robot0_eef_quat"])
            curr_state = convert_pose(np.concatenate(curr_state), env)
            norm_state, state = proc_state(curr_state, norm_list[2], norm_list[3])
            print(f"Dataset state: {(states[i])}")
            print(f"Denorm state diff: {(states[i]) -state }")
            print(f"Curr state: {state}")
            print(f"Normalized state: {norm_state}")
            print(f"Gripper command: {states[i, -1]}")
            command = actions[i]
            if actions[i, -1] >= -0.6:
                command[-1] =1 
            else: 
                command[-1] = -1
            ac_i = proc_data_action(command)
            obs,reward, done,info = env.step(ac_i)
            env.render()