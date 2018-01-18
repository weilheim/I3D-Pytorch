import os
import torchvision
import numpy as np
from PIL import Image
from feature_extraction import DataLoader


gt_rgb = np.load("/home/liusheng/Code/I3D/data/v_CricketShot_g04_c01_rgb.npy")
print gt_rgb.shape
print gt_rgb.max()
print gt_rgb.min()

trans = torchvision.transforms.Compose([
    torchvision.transforms.Resize(256, interpolation=Image.BILINEAR),
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])
])
image_dir = "/home/liusheng/UCF101/Orig_images/CricketShot/v_CricketShot_g04_c01/"
image_list = sorted(os.listdir(image_dir))
# image = Image.open(os.path.join(image_dir, image_list[0]))
# image = trans(image)
# np_image = image.numpy()
# print np_image.shape
# print np_image.max()
# print np_image.min()
#
# pil_image = torchvision.transforms.ToPILImage()(image)
# pil_image.show()
#
# gt_image = np.transpose(gt_rgb[0, 0, :, :, :], axes=(2, 0, 1))
# print gt_image
# diff = np.sum(np.abs((gt_image + 1.0) / 2.0 - np_image), axis=(0, 1, 2))
# print diff
# total = np.sum(np.abs(gt_image))
# print total


images = []
for i in image_list:
    img = Image.open(os.path.join(image_dir, i))
    img = trans(img).permute(1, 2, 0)
    images.append(img.numpy())
images = np.array(images)[np.newaxis, :][:, 0:-1, :, :, :]
np.save('data/my_v_CricketShot_g04_c01_rgb.npy', images)
print images.shape
diff = np.sum(np.abs(images - gt_rgb))
print diff
total = np.sum(np.abs(gt_rgb))
print total
print diff / float(total) * 100, "%"

