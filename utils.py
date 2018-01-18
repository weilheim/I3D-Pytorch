import cv2
from cv2 import DualTVL1OpticalFlow_create as DualTVL1

import os
import numpy as np

def compute_TVL1(image_dir):
    """Compute TV-L1 optical flow."""
    image_list = sorted(os.listdir(image_dir))
    prev_image = cv2.imread(os.path.join(image_dir, image_list[0]))
    prev_image = cv2.cvtColor(prev_image, cv2.COLOR_RGB2GRAY)
    TVL1 = DualTVL1()
    flow = []
    for i in range(1, len(image_list)):
        cur_image = cv2.imread(os.path.join(image_dir, image_list[i]))
        cur_image = cv2.cvtColor(cur_image, cv2.COLOR_RGB2GRAY)
        cur_flow = TVL1.calc(prev_image, cur_image, None)
        prev_image = cur_image

        max_val = lambda x: max(max(x.flatten()), abs(min(x.flatten())))
        # cur_flow = cur_flow / max_val(cur_flow)
        flow.append(cur_flow)
    flow = np.array(flow)
    return flow

def compute_TVL1_ndarray(image_array):
    prev_image = image_array[0, 0, :]
    prev_image = np.squeeze(np.mean(prev_image, axis=-1))
    TVL1 = DualTVL1()
    flow = []
    for i in range(1, image_array.shape[1]):
        cur_image = image_array[0, i, :]
        cur_image = np.squeeze(np.mean(cur_image, axis=-1))
        cur_flow = TVL1.calc(prev_image, cur_image, None)
        prev_image = cur_image

        flow.append(cur_flow)
    flow = np.array(flow)
    return flow

# flow = np.load("/home/liusheng/Code/I3D/data/v_CricketShot_g04_c01_flow.npy")
# print flow.shape
# print np.max(flow)
# print np.min(flow)

image_dir = "/home/liusheng/UCF101/Orig_images/CricketShot/v_CricketShot_g04_c01/"
flow = compute_TVL1(image_dir)

# image_array_dir = "/home/liusheng/Code/I3D/data/v_CricketShot_g04_c01_rgb.npy"
# image_array = np.load(image_array_dir)
# print image_array.shape
# flow = compute_TVL1_ndarray(image_array)

flow[flow > 1.0] = 1.0
flow[flow < -1.0] = -1.0
gt_flow = np.load("/home/liusheng/Code/I3D/data/v_CricketShot_g04_c01_flow.npy")
gt_flow = np.squeeze(gt_flow)[:, :]
print gt_flow.shape
print gt_flow - flow
