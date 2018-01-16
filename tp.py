from __future__ import absolute_import
from __future__ import division

import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import sonnet as snt
import tensorflow as tf

import i3d as tf3d
import i3d_pytorch as py3d
# import load_pytorch as lp

os.environ['CUDA_VISIBLE_DEVICES'] = '3'

# tf_unit = tf3d.Unit3D(output_channels=3,
#                       kernel_shape=(1, 1, 1),
#                       stride=(1, 1, 1),
#                       activation_fn=None,
#                       use_batch_norm=True,
#                       use_bias=False,
#                       name='unit_3d')
# tf_input = tf.placeholder(tf.float32, shape=(1, 3, 5, 5, 2))
# tf_output = tf_unit(tf_input, is_training=False)
#
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     print tf.global_variables()
#     weight = sess.graph.get_tensor_by_name('unit_3d/conv_3d/w:0').eval()
#     beta = sess.graph.get_tensor_by_name( 'unit_3d/batch_norm/beta:0').eval()
#     mv_var = sess.graph.get_tensor_by_name('unit_3d/batch_norm/moving_variance:0').eval()
#     mv_mean = sess.graph.get_tensor_by_name('unit_3d/batch_norm/moving_mean:0').eval()
#
#     tf_y = sess.run([tf_output], feed_dict={tf_input: x})
#     print tf_y
#
# py_unit = py3d.Unit3D(input_channels=2,
#                       output_channels=3,
#                       kernel_shape=(1, 1, 1),
#                       stride=(1, 1, 1),
#                       activation_fn=None,
#                       use_batch_norm=True,
#                       use_bias=False)
# py_unit.eval()
# py_unit.conv3d.weight.data = torch.from_numpy(weight).float().permute(4, 3, 0, 1, 2)
# py_input = Variable(torch.from_numpy(x).float().permute(0, 4, 1, 2, 3))
# py_unit.bn.weight.data = torch.ones(3)
# py_unit.bn.bias.data = torch.from_numpy(beta).squeeze()
# py_unit.bn.running_mean = torch.from_numpy(mv_mean).squeeze()
# py_unit.bn.running_var = torch.from_numpy(mv_var).squeeze()
#
# py_y = py_unit(py_input)
# # 1, 3, 5, 5, 2
# # 1, 2, 3, 5, 5
# py_y = py_y.permute(0, 2, 3, 4, 1)
# py_y = np.squeeze(py_y.data.numpy())
# tf_y = np.squeeze(tf_y)
# print 'diff:', py_y - tf_y


# x = np.random.rand(1, 8, 5, 5, 2)
# snt_conv3d = snt.Conv3D(output_channels=3,
#                         kernel_shape=(3, 3, 3),
#                         stride=(2, 2, 2),
#                         padding=snt.SAME,
#                         use_bias=False,
#                         # initializers={"w": tf.constant_initializer(weight)},
#                         name='snt_conv3d')
# snt_input = tf.placeholder(tf.float32, shape=(1, 8, 5, 5, 2))
# snt_output = snt_conv3d(snt_input)
#
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     weight = sess.graph.get_tensor_by_name('snt_conv3d/w:0').eval()
#     snt_y = sess.run([snt_output], feed_dict={snt_input: x})
#
# py_conv3d = nn.Conv3d(in_channels=2,
#                       out_channels=3,
#                       kernel_size=(3, 3, 3),
#                       stride=(2, 2, 2),
#                       padding=(0, 0, 0),
#                       bias=False)
# # print py_conv3d.weight.data
# py_conv3d.weight.data = torch.from_numpy(weight).float().permute(4, 3, 0, 1, 2)
# py_input = Variable(torch.from_numpy(x).float().permute(0, 4, 1, 2, 3))
# # py_input = Variable(torch.rand((4, 3, 2, 1)))
# py_input = F.pad(py_input, (1, 1, 1, 1, 0, 1), mode='constant', value=0)
# py_y = py_conv3d(py_input)
#
# # 1, 3, 5, 5, 2
# # 1, 2, 3, 5, 5
# py_y = py_y.permute(0, 2, 3, 4, 1)
# py_y = np.squeeze(py_y.data.numpy())
# snt_y = np.squeeze(snt_y)
# print py_y - snt_y


# # height: 6, width: 5
# x = np.random.rand(1, 6, 5, 2)
# # weight = np.load('weight.npy')
# snt_conv2d = snt.Conv2D(output_channels=3,
#                         kernel_shape=(3, 3),
#                         stride=(2, 2),
#                         padding=snt.SAME,
#                         use_bias=False,
#                         # initializers={"w": tf.constant_initializer(weight)},
#                         name='snt_conv2d')
# snt_input = tf.placeholder(tf.float32, shape=(1, 6, 5, 2))
# snt_output = snt_conv2d(snt_input)
#
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     weight = sess.graph.get_tensor_by_name('snt_conv2d/w:0').eval()
#     snt_y = sess.run(snt_output, feed_dict={snt_input: x})
#     print snt_y.shape
#
# py_padding = nn.ZeroPad2d((1, 1, 0, 1))
# py_conv3d = nn.Conv2d(in_channels=2,
#                       out_channels=3,
#                       kernel_size=(3, 3),
#                       stride=(2, 2),
#                       padding=(0, 0),
#                       bias=False)
# py_conv3d.weight.data = torch.from_numpy(weight).float().permute(3, 2, 0, 1)
# py_input = Variable(torch.from_numpy(x).float().permute(0, 3, 1, 2))
# py_input = py_padding(py_input)
# py_y = py_conv3d(py_input)
#
# # 1, 5, 5, 3
# # 1, 3, 5, 5
# py_y = py_y.permute(0, 2, 3, 1)
# py_y = np.squeeze(py_y.data.numpy())
# snt_y = np.squeeze(snt_y)
# print py_y - snt_y


# _CHECKPOINT_PATHS = {
#     'rgb': 'data/checkpoints/rgb_scratch/model.ckpt',
#     'flow': 'data/checkpoints/flow_scratch/model.ckpt',
#     'rgb_imagenet': 'data/checkpoints/rgb_imagenet/model.ckpt',
#     'flow_imagenet': 'data/checkpoints/flow_imagenet/model.ckpt',
# }
# _SAMPLE_PATHS = {
#     'rgb': 'data/v_CricketShot_g04_c01_rgb.npy',
#     'flow': 'data/v_CricketShot_g04_c01_flow.npy',
# }
# _LABEL_MAP_PATH = 'data/label_map.txt'
# kinetics_classes = [x.strip() for x in open(_LABEL_MAP_PATH)]
# rgb_sample = np.load(_SAMPLE_PATHS['rgb'])
#
# tf_unit = tf3d.Unit3D(output_channels=64,
#                       kernel_shape=(7, 7, 7),
#                       stride=(2, 2, 2),
#                       activation_fn=tf.nn.relu,
#                       use_batch_norm=True,
#                       use_bias=False,
#                       name='unit_3d')
# tf_input = tf.placeholder(tf.float32, shape=(1, 79, 224, 224, 3))
# tf_output, tf_orig_output = tf_unit(tf_input, is_training=False)
# variable_list = tf.global_variables()
# rgb_variable_map = {}
# rgb_variable_map['RGB/inception_i3d/Conv3d_1a_7x7/conv_3d/w'] = [v for v in variable_list if v.name == 'unit_3d/conv_3d/w:0'][0]
# rgb_variable_map['RGB/inception_i3d/Conv3d_1a_7x7/batch_norm/beta'] = [v for v in variable_list if v.name == 'unit_3d/batch_norm/beta:0'][0]
# rgb_variable_map['RGB/inception_i3d/Conv3d_1a_7x7/batch_norm/moving_mean'] = [v for v in variable_list if v.name == 'unit_3d/batch_norm/moving_mean:0'][0]
# rgb_variable_map['RGB/inception_i3d/Conv3d_1a_7x7/batch_norm/moving_variance'] = [v for v in variable_list if v.name == 'unit_3d/batch_norm/moving_variance:0'][0]
# rgb_saver = tf.train.Saver(var_list=rgb_variable_map, reshape=True)
# with tf.Session() as sess:
#     print tf.global_variables()
#     rgb_saver.restore(sess, _CHECKPOINT_PATHS['rgb_imagenet'])
#     weight = sess.graph.get_tensor_by_name('unit_3d/conv_3d/w:0').eval()
#     beta = sess.graph.get_tensor_by_name('unit_3d/batch_norm/beta:0').eval()
#     mv_var = sess.graph.get_tensor_by_name('unit_3d/batch_norm/moving_variance:0').eval()
#     mv_mean = sess.graph.get_tensor_by_name('unit_3d/batch_norm/moving_mean:0').eval()
#     np.save('weight.npy', weight)
#     np.save('beta.npy', beta)
#     np.save('mv_var.npy', mv_var)
#     np.save('mv_mean.npy', mv_mean)
#
#     tf_y, tf_orig_y = sess.run([tf_output, tf_orig_output], feed_dict={tf_input: rgb_sample})
#     # print 'tf_y\n', tf_y
#     # print 'tf_orig_y\n', tf_orig_y
#     np.save('Conv_1a_out.npy', tf_y)
#     np.save('Conv_1a_conv_out.npy', tf_orig_y)
#
#
# py_unit = py3d.Unit3D(input_channels=3,
#                       output_channels=64,
#                       kernel_shape=(7, 7, 7),
#                       stride=(2, 2, 2),
#                       padding=(0, 0, 0),
#                       activation_fn=F.relu,
#                       cal_padding=True,
#                       use_batch_norm=True,
#                       use_bias=False)
# py_unit.eval()
# py_unit.conv3d.weight.data = torch.from_numpy(weight).float().permute(4, 3, 0, 1, 2)
# py_input = Variable(torch.from_numpy(rgb_sample).float().permute(0, 4, 1, 2, 3))
# py_unit.bn.weight.data = torch.ones(64)
# py_unit.bn.bias.data = torch.from_numpy(beta).squeeze()
# py_unit.bn.running_mean = torch.from_numpy(mv_mean).squeeze()
# py_unit.bn.running_var = torch.from_numpy(mv_var).squeeze()
# py_unit.eval()
#
# # py_input = F.pad(py_input, (2, 3, 2, 3, 3, 3))
# py_y, py_orig_y = py_unit(py_input)
# # 1, C, 5, 5, T
# # 1, T, C, 5, 5
# py_y = py_y.permute(0, 2, 3, 4, 1)
# py_y = np.squeeze(py_y.data.numpy())
# py_orig_y = py_orig_y.permute(0, 2, 3, 4, 1)
# py_orig_y = np.squeeze(py_orig_y.data.numpy())
# tf_y = np.squeeze(tf_y)
# tf_orig_y = np.squeeze(tf_orig_y)
# print '\ny difference\n', np.sum(np.abs(py_y - tf_y))
# print '\norig y difference\n', np.sum(np.abs(py_orig_y - tf_orig_y))


print np.load('Logits.npy').shape
print np.load('Features.npy').shape

# unit_dict = {}
# lp.load_unit3d(unit_dict, 'RGB', 'Conv3d_1a_7x7', '', 'param/')
# py_unit.load_state_dict(unit_dict)
# py_input = Variable(torch.from_numpy(rgb_sample).float().permute(0, 4, 1, 2, 3))
# py_y, py_conv_y = py_unit(py_input)
# # 1, 3, 5, 5, 2
# # 1, 2, 3, 5, 5
# py_y = py_y.permute(0, 2, 3, 4, 1)
# py_y = np.squeeze(py_y.data.numpy())
# print py_y.shape
# py_conv_y  = py_conv_y.permute(0, 2, 3, 4, 1)
# py_conv_y = np.squeeze(py_conv_y.data.numpy())
# print py_conv_y.shape
#
#
#
# # tf_y = np.squeeze(tf_y)
# # print 'diff:', py_y - tf_y
