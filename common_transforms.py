import numpy as np
import scipy
from scipy.spatial.transform import Rotation as R
import yaml
import os
from scipy.linalg import expm

def cart2se3(p): 
    #cart2se3 confirmed
    p = np.array(p)
    rot = R.from_euler('xyz', p[3:], degrees=True).as_matrix()
    homog_pose = np.eye(4)
    homog_pose[:3,:3] = rot
    homog_pose[:3,3] = p[:3]/1000
    return homog_pose

def get_rel_command(pose1, pose2):
    g_rel = np.linalg.inv(pose1) @ pose2
    command = np.zeros(6,)
    command[:3] = g_rel[:3,3] * 1000
    command[3:] = R.from_matrix(g_rel[:3,:3]).as_euler('xyz', degrees=True)

    return command # in [mm, degree]

def get_rel_twist(pose1, pose2): 
    g_rel = np.linalg.inv(pose1) @ pose2
    twist, _ = se3_logmap(g_rel, type="rotmat")
    return twist                 

def se3_logmap(pose, type="euler"): 
    """
    Convert a pose from the robot to a logmap representation
    """
    if type == "euler":
        pose = np.array(pose)
        p = pose[:3]
        rot = R.from_euler('xyz', pose[3:]).as_matrix()
    else: 
        rot = pose[:3,:3]
        p = pose[:3,3]
        
    w, theta = so3_logmap(rot)
    if theta == 0: 
        twist = np.concatenate((w*theta, p), axis = 0)
        return twist, np.eye(3)
    w_hat = hat_map(w)
    # V = np.eye(3) +((1 - np.cos(theta))/(theta**2)) * w_hat + ((theta - np.sin(theta))/(theta**3)) * (w_hat @ w_hat)
    V = np.eye(3) * (1/theta) - (0.5 * w_hat) + ((1/theta) - (0.5 * (1/np.tan(theta/2)))) * (w_hat @ w_hat)
    v = V @ p
    rot = R.from_matrix(rot)
    twist = np.concatenate((rot.as_rotvec(), v*theta), axis = 0)

    return twist, V        
        
def hat_map(w):
    if w.shape[0] == 3:
        # Perform SO(3) hat_map
        w_hat = np.array([[0, -w[2], w[1]],
                            [w[2], 0, -w[0]],
                            [-w[1], w[0], 0]])
    elif w.shape[0] == 6:
        t = w[3:]
        w_hat = np.array([[0, -w[2], w[1], t[0]],
                            [w[2], 0, -w[0],t[1]],
                            [-w[1], w[0], 0, t[2]], 
                            [0, 0, 0, 0]])
    return w_hat 

def vee_map(mat):
    if mat.shape == (3,3):
        # Perform SO(3) vee_map
        vec = np.array([mat[2, 1], mat[0, 2], mat[1, 0]])
    elif mat.shape == (4,4):
        # Perform SE(3) vee_map
        vec = np.array([mat[2, 1], mat[0, 2], mat[1, 0], mat[0, 3], mat[1, 3], mat[2, 3]])
    else:
        raise ValueError("Invalid shape for vee_map")
    return vec
        
def so3_logmap(rot): 
    """
    Convert a rotation from the robot to a logmap representation
            """
    if np.sum(np.eye(3) - rot) < 1e-8:
        return np.zeros(3), 0
    r = R.from_matrix(rot)
    test_theta = np.arccos(np.clip((np.trace(r.as_matrix()) - 1) / 2, -1, 1))
    test_omega  = (1/(2*np.sin(test_theta))) * np.array([rot[2, 1] - rot[1, 2],rot[0, 2] - rot[2, 0], rot[1, 0] - rot[0, 1]])
    return test_omega, test_theta
            
def g2cart(g): 
    p = g[:3, 3] * 1000
    rot = R.from_matrix(g[:3, :3])
    rot = rot.as_euler('xyz', degrees=True)
    return np.concatenate((p, rot), axis = 0)

def se3_expmap(twist): 
    twist_hat = hat_map(twist, type="twist")

    g = scipy.linalg.expm(twist_hat)
    return g

def command_to_pose_data(command):
    """
    convert a command format to a pose_data format
    """
    output = np.zeros((6,))
    output[:3] = command[:3] / 1000 # mm to m
    output[3:] = R.from_euler('xyz', command[3:], degrees=True).as_rotvec()
    
    return output

def command_to_pose_data_rot6d(command):
    """
    convert a command format to a pose_data format
    """
    output = np.zeros((9,))
    output[:3] = command[:3] / 1000 # mm to m
    rotvec = R.from_euler('xyz', command[3:], degrees=True).as_rotvec()
    output[3:] = rotvec_to_rot6d(rotvec)
    
    return output

def command_to_hom_matrix(command):
    """
    convert a command format to a homogenous matrix
    """
    output = np.eye(4)
    output[:3, 3] = command[:3] / 1000 # mm to m
    output[:3, :3] = R.from_euler('xyz', command[3:], degrees=True).as_matrix()
    
    return output

def hom_matrix_to_pose_data(matrix, type="rotvec"):
    """
    convert a homogenous matrix to a pose_data format
    """
    if type == "rotvec":
        output = np.zeros((6,))
        output[:3] = matrix[:3, 3] # in meter
        
        output[3:] = R.from_matrix(matrix[:3, :3]).as_rotvec()
    elif type == "rot6d":
        output = np.zeros((9,))
        output[:3] = matrix[:3, 3] # in meter
        output[3:] = rotm_to_rot6d(matrix[:3, :3])
    
    return output

def quat_to_rot6d(quat):
    """Convert quaternion to 6D rotation representation.
    Args:
        quat (np.array): quaternion in wxyz format
    Returns:
        np.array: 6D rotation representation
    """
    r = R.from_quat(quat).as_matrix()

    return r[:3, :2].T.flatten()

def rotm_to_rot6d(rotm):
    return rotm[:3, :2].T.flatten()

def rotvec_to_rot6d(rotvec):
    r = R.from_rotvec(rotvec).as_matrix()

    return r[:3, :2].T.flatten()

def rot6d_to_quat(rot6d):
    """Convert 6D rotation representation to quaternion.
    Args:
        rot6d (np.array): 6D rotation representation
    """
    x_raw = rot6d[:3]
    y_raw = rot6d[3:]
    x = x_raw / np.linalg.norm(x_raw)
    z = np.cross(x, y_raw)
    z = z / np.linalg.norm(z)
    y = np.cross(z, x)
    print(f"x: {x}, y: {y}, z: {z}")
    quat = R.from_matrix(np.column_stack((x, y, z))).as_quat()
    
    return quat

def rot6d_to_rotm(rot6d):
    """Convert 6D rotation representation to rotation matrix.
    Args:
        rot6d (np.array): 6D rotation representation
    """
    x_raw = rot6d[:3]
    y_raw = rot6d[3:]
    x = x_raw / np.linalg.norm(x_raw)
    z = np.cross(x, y_raw)
    z = z / np.linalg.norm(z)
    y = np.cross(z, x)

    return np.column_stack((x, y, z))

def rot6d_to_rotvec(rot6d):
    """Convert 6D rotation representation to rotation vector.
    Args:
        rot6d (np.array): 6D rotation representation
    """
    rotm = rot6d_to_rotm(rot6d)
    return R.from_matrix(rotm).as_rotvec()

def SE3_log_map(g):
    p, R = g[:3,3], g[:3,:3]
    r = R.as_rotvec()


def hat_map(w):
    return np.array([[0, -w[2], w[1]],
                        [w[2], 0, -w[0]],
                        [-w[1], w[0], 0]])

def vee_map(mat):
    return np.array([mat[2, 1], mat[0, 2], mat[1, 0]])

def SO3_log_map(R):
    theta = np.arccos((np.trace(R) - 1) / 2)
    if theta == 0:
        xi = np.zeros(3,)
    else:
        xi = theta / (2 * np.sin(theta)) * vee_map(R - R.T)

    return xi

def SE3_log_map(g):
    p, R = g[:3,3], g[:3,:3]
    
    psi = SO3_log_map(R)
    psi_norm = np.linalg.norm(psi)
    psi_hat = hat_map(psi)

    if np.isclose(psi_norm, 0):
        A_inv = np.eye(3) - 0.5 * psi_hat + 1 / 12.0 * psi_hat @ psi_hat

    else:
        cot = 1 / np.tan(psi_norm / 2)
        alpha = (psi_norm /2) * cot

        A_inv = np.eye(3) - 0.5 * psi_hat + (1 - alpha)/(psi_norm**2) * psi_hat @ psi_hat

    v = A_inv @ p
    xi = np.zeros(6,)
    xi[:3] = v
    xi[3:] = psi

    return xi

def SE3_exp_map(xi):
    v, omega = xi[:3], xi[3:]

    omega_hat = hat_map(omega)

    xi_hat = np.zeros((4, 4))
    xi_hat[:3, :3] = omega_hat
    xi_hat[:3, 3] = v

    g = expm(xi_hat)

    return g