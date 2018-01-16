# import torch
# import torch.nn as nn
# from torch.autograd import Variable
#
# # x = Variable(torch.zeros(1, 3, 64))
# # conv = nn.Conv1d(3, 1, kernel_size=(6), stride=2, padding=(3))
# # y = conv(x)
# # print y.size()
#
# bn = nn.BatchNorm1d(2, affine=True)
# bn.weight.data.fill_(1)
# bn.bias.data = torch.FloatTensor([0.5, 0])
# # bn.running_mean.fill_(0)
# # bn.running_var.fill_(0)
# print 'Initial running_mean:', bn.running_mean
# print 'Initial running_var:', bn.running_var
#
# bn.eval()
# x = Variable(torch.FloatTensor([[1, 2], [0, 4]]))
# print x
# print bn(x)
# print 'Current running_mean:', bn.running_mean
# print 'Current running_var:', bn.running_var

import numpy as np
import sonnet as snt
import tensorflow as tf

bn = snt.BatchNorm(scale=False, decay_rate=0.9, eps=1e-5,
                   initializers={"beta": tf.constant_initializer([0.5, 0]),
                                 "moving_mean": tf.constant_initializer(0.0),
                                 "moving_variance": tf.constant_initializer(1.0)})
bn_input = tf.placeholder(tf.float32, shape=(2, 2))
bn_output = bn(bn_input, is_training=False, test_local_stats=False)
x = np.array([[1, 2], [0, 4]])
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    y = sess.run(bn_output, feed_dict={bn_input: x})
    print y

    var_list = tf.trainable_variables()
    beta = [v for v in tf.trainable_variables() if v.name == "batch_norm/beta:0"][0]
    moving_mean = [v for v in tf.global_variables() if v.name == "batch_norm/moving_mean:0"][0]
    moving_variance = [v for v in tf.global_variables() if v.name == "batch_norm/moving_variance:0"][0]

    print sess.run(beta)
    print sess.run(moving_mean)
    print sess.run(moving_variance)

# import os
# import numpy as np
# import tensorflow as tf
# from i3d import Unit3D, InceptionI3d

# _IMAGE_SIZE = 224
# _NUM_CLASSES = 400
#
# _SAMPLE_VIDEO_FRAMES = 79
# _SAMPLE_PATHS = {
#     'rgb': 'data/v_CricketShot_g04_c01_rgb.npy',
#     'flow': 'data/v_CricketShot_g04_c01_flow.npy',
# }
#
# _CHECKPOINT_PATHS = {
#     'rgb': 'data/checkpoints/rgb_scratch/model.ckpt',
#     'flow': 'data/checkpoints/flow_scratch/model.ckpt',
#     'rgb_imagenet': 'data/checkpoints/rgb_imagenet/model.ckpt',
#     'flow_imagenet': 'data/checkpoints/flow_imagenet/model.ckpt',
# }


# def tf_activation(eval_type, endpoint, imagenet_pretrained):
#     if eval_type not in ['rgb', 'flow', 'joint']:
#         raise ValueError('Bad `eval_type`, must be one of rgb, flow, joint')
#
#     rgb_saver = None
#     if eval_type in ['rgb', 'joint']:
#         rgb_input = tf.placeholder(tf.float32, shape=(1, _SAMPLE_VIDEO_FRAMES, _IMAGE_SIZE, _IMAGE_SIZE, 3))
#         with tf.variable_scope('RGB'):
#             rgb_model = InceptionI3d(_NUM_CLASSES, spatial_squeeze=True, final_endpoint='Logits')
#         rgb_logits, rgb_endpoints = rgb_model(rgb_input, is_training=False, dropout_keep_prob=1.0)
#         rgb_variable_map = {}
#         for variable in tf.global_variables():
#             if variable.name.split('/')[0] == 'RGB':
#                 rgb_variable_map[variable.name.replace(':0', '')] = variable
#             print(variable.name)
#             print(variable.value(), '\n')
#         rgb_saver = tf.train.Saver(var_list=rgb_variable_map, reshape=True)
#
#     with tf.Session() as sess:
#         feed_dict = {}
#         if eval_type in ['rgb', 'joint']:
#             if imagenet_pretrained:
#                 rgb_saver.restore(sess, _CHECKPOINT_PATHS['rgb_imagenet'])
#             else:
#                 rgb_saver.restore(sess, _CHECKPOINT_PATHS['rgb'])
#             tf.logging.info('RGB checkpoint restored')
#             rgb_sample = np.load(_SAMPLE_PATHS['rgb'])
#             tf.logging.info('RGB data loaded, shape=%s', str(rgb_sample.shape))
#             feed_dict[rgb_input] = rgb_sample
#
#             out_logits, out_predictions = sess.run([rgb_logits, rgb_endpoints], feed_dict=feed_dict)
#             out_logits = out_logits[0]
#             out_predictions = out_predictions[0]
#             print('Norm of logits: %f' % np.linalg.norm(out_logits))
