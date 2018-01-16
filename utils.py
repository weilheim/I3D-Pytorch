import cv2
from cv2 import DualTVL1OpticalFlow_create as DualTVL1

import os
import numpy as np

def compute_TVL1(image_dir):
    """Compute TV-L1 optical flow."""
    image_list = sorted(os.listdir(image_dir))
    pre_image = cv2.imread()
    TVL1 = DualTVL1()
    flow = []
    for i in range(1, len(image_list)):
        cur_image = cv2.imread()
        cur_flow = TVL1.calc(pre_image, cur_image, None)
        pre_image = cur_image

        max_val = lambda x: max(max(x.flatten()), abs(min(x.flatten())))
        cur_flow = cur_flow / max_val(cur_flow)
        flow.append(cur_flow)
    flow = np.array(flow)
    return flow
