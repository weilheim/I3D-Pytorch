from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random
import numpy as np
import time
import torch
import torchvision
# import torchvision.transforms as transforms
import sonnet as snt
import tensorflow as tf
from PIL import Image

import i3d
import transforms

# os.environ['CUDA_VISIBLE_DEVICES'] = '3'


_RESCALE_SIZE = 256
_CROP_SIZE = 224
_NUM_CLASSES = 400

_CHECKPOINT_PATHS = {
    'rgb': 'data/checkpoints/rgb_scratch/model.ckpt',
    'flow': 'data/checkpoints/flow_scratch/model.ckpt',
    'rgb_imagenet': 'data/checkpoints/rgb_imagenet/model.ckpt',
    'flow_imagenet': 'data/checkpoints/flow_imagenet/model.ckpt',
}

_LABEL_MAP_PATH = 'data/label_map.txt'

feature_type = 'rgb'  # FLAGS.eval_type
imagenet_pretrained = True

# root = '/home/liusheng/UCF101/Orig_images/'
root = '/home/liusheng/MSVD/Orig_images/'
feature_dir = '/home/liusheng/MSVD/I3D_features/'


class DataLoader(object):
    def __init__(self, root,
                 hierachy=1,
                 rescale_size=256,
                 crop_size=224,
                 seed=1):
        if not os.path.exists(root):
            raise ValueError('Root doest not exist')
        self.root = root
        if not hierachy in [1, 2]:
            raise ValueError('Bad `hierachy`, must be one of 1, 2.')
        self.hierachy = hierachy
        if self.hierachy == 1:
            self.videos = [os.path.join(self.root, v) for v in os.listdir(self.root)]
        else:
            cls = os.listdir(self.root)
            self.videos = [os.path.join(self.root, c, v) for c in cls for v in os.listdir(os.path.join(self.root, c))]
        self.videos = sorted(self.videos, key=str.lower)
        self.rescale_size = rescale_size
        self.crop_size = crop_size
        self.seed = seed
        random.seed(self.seed)
        np.random.seed(self.seed)
        self.offset = 0

        # self.trans = torchvision.transforms.Compose([
        #     torchvision.transforms.Resize(self.rescale_size, interpolation=Image.BILINEAR),
        #     torchvision.transforms.CenterCrop(self.crop_size),
        #     torchvision.transforms.ToTensor(),
        #     torchvision.transforms.Normalize(
        #         mean=[.485, .456, .406],
        #         std=[.229, .224, .225])
        # ])
        self.trans = torchvision.transforms.Compose([
            transforms.GroupScale(self.rescale_size, interpolation=Image.BILINEAR),
            transforms.GroupCenterCrop(self.crop_size),
            transforms.Stack4d(roll=False),
            transforms.ToTorchFormatTensor4d(div=True),
            transforms.GroupNormalize(
                mean=[.485, .456, .406],
                std=[.229, .224, .225])
        ])

    def __len__(self):
        return len(self.videos)

    def next_batch(self, shuffle=False, save_npy=False):
        if not shuffle:
            if self.offset == len(self.videos):
                raise ValueError('Done')
            ix = self.offset
            self.offset += 1
        else:
            ix = random.randint(0, len(self.videos)-1)
        video = self.videos[ix]
        image_dirs = sorted(os.listdir(video))
        images = [Image.open(os.path.join(video, id), 'r') for id in image_dirs]
        if save_npy:
            if images[0].mode == 'L':
                array = np.concatenate([np.array(i)[:, :, np.newaxis, np.newaxis] for i in images], axis=3)
            elif images[0].mode == 'RGB':
                array = np.concatenate([np.array(i)[:, :, :, np.newaxis] for i in images], axis=3)
                array = np.transpose(array, axes=(3, 2, 0, 1))
        tensor = self.trans(images)
        return tensor.numpy()[np.newaxis, :], video


def main(hierachy, eval_type='rgb', endpoint='Logits'):
    kinetics_classes = [x.strip() for x in open(_LABEL_MAP_PATH)]

    if eval_type not in ['rgb', 'flow', 'joint']:
        raise ValueError('Bad `eval_type`, must be one of rgb, flow, joint')

    if eval_type in ['rgb', 'joint']:
        # RGB input has 3 channels.
        rgb_input = tf.placeholder(tf.float32, shape=(1, None, _CROP_SIZE, _CROP_SIZE, 3))
        with tf.variable_scope('RGB'):
            rgb_model = i3d.InceptionI3d(_NUM_CLASSES, spatial_squeeze=True, final_endpoint=endpoint)
            rgb_logits, rgb_previous = rgb_model(rgb_input, is_training=False, dropout_keep_prob=1.0)
            rgb_predictions = tf.nn.softmax(rgb_logits)
        rgb_variable_map = {}
        for variable in tf.global_variables():
            if variable.name.split('/')[0] == 'RGB':
                rgb_variable_map[variable.name.replace(':0', '')] = variable
        rgb_saver = tf.train.Saver(var_list=rgb_variable_map, reshape=True)

    if eval_type in ['flow', 'joint']:
        # Flow input has only 2 channels.
        flow_input = tf.placeholder(tf.float32, shape=(1, None, _CROP_SIZE, _CROP_SIZE, 2))
        with tf.variable_scope('Flow'):
            flow_model = i3d.InceptionI3d(_NUM_CLASSES, spatial_squeeze=True, final_endpoint=endpoint)
            flow_logits, flow_previous = flow_model(flow_input, is_training=False, dropout_keep_prob=1.0)
            flow_predictions = tf.nn.softmax(flow_logits)
        flow_variable_map = {}
        for variable in tf.global_variables():
            if variable.name.split('/')[0] == 'Flow':
                flow_variable_map[variable.name.replace(':0', '')] = variable
        flow_saver = tf.train.Saver(var_list=flow_variable_map, reshape=True)

    loader = DataLoader(root, hierachy=hierachy, rescale_size=_RESCALE_SIZE, crop_size=_CROP_SIZE)
    with tf.Session() as sess:
        feed_dict = {}
        if eval_type in ['rgb', 'joint']:
            rgb_saver.restore(sess, _CHECKPOINT_PATHS['rgb_imagenet'])
            tf.logging.info('RGB checkpoint restored')

            for i in range(len(loader)):
                start_time = time.time()
                sample, video_name = loader.next_batch(shuffle=False)
                tf.logging.info('RGB data loaded, shape=%s', str(sample.shape))
                feed_dict[rgb_input] = sample

                logits, predictions, previous = sess.run([rgb_logits, rgb_predictions, rgb_previous],
                                                 feed_dict=feed_dict)
                logits = logits[0]
                predictions = predictions[0]
                mixed_5c = np.squeeze(previous['Mixed_5c'])
                features = np.squeeze(previous['Features'])
                print('\n', video_name)
                print('sample:', sample.shape)
                print('mixed_5c:', mixed_5c.shape)
                print('features:', features.shape)
                print('seconds:', time.time() - start_time)
                # np.save(os.path.join(feature_dir, video_name), features)
                feature_name = video_name.split('/')[-1]
                np.save(os.path.join(feature_dir, feature_name), features)

        if eval_type in ['flow', 'joint']:
            flow_saver.restore(sess, _CHECKPOINT_PATHS['flow_imagenet'])
            tf.logging.info('Flow checkpoint restored')

            for i in range(len(loader)):
                start_time = time.time()
                sample, video_name = loader.next_batch(shuffle=False)
                tf.logging.info('Flow data loaded, shape=%s', str(sample.shape))
                feed_dict[flow_input] = sample

                logits, predictions, previous = sess.run([flow_logits, flow_predictions, flow_previous],
                                                         feed_dict=feed_dict)
                logits = logits[0]
                predictions = predictions[0]
                mixed_5c = np.squeeze(previous['Mixed_5c'])
                features = np.squeeze(previous['Features'])
                print('\n', video_name)
                print('sample:', sample.shape)
                print('mixed_5c:', mixed_5c.shape)
                print('features:', features.shape)
                print('seconds:', time.time() - start_time)
                # np.save(os.path.join(feature_dir, video_name), features)
                feature_name = 'flow_' + video_name.split('/')[-1]
                np.save(os.path.join(feature_dir, feature_name), features)

                # sorted_indices = np.argsort(predictions)[::-1]
                # print('\nTop classes and probabilities for', video_name)
                # for ix in sorted_indices[:20]:
                #     print(predictions[ix], logits[ix], kinetics_classes[ix])


if __name__ == "__main__":
    # loader = DataLoader(root, 2)
    # tensor, video = loader.next_batch()
    # print(video, tensor.size())
    hierachy = 1
    main(hierachy)