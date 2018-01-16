"""Load TensorFlow I3D model pretrained on Kinetics and dump all the weights."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import h5py
import math
import os
import tensorflow as tf

import i3d

_IMAGE_SIZE = 224
_NUM_CLASSES = 400

_SAMPLE_VIDEO_FRAMES = 79
_SAMPLE_PATHS = {
    'rgb': 'data/v_CricketShot_g04_c01_rgb.npy',
    'flow': 'data/v_CricketShot_g04_c01_flow.npy',
}

_CHECKPOINT_PATHS = {
    'rgb': 'data/checkpoints/rgb_scratch/model.ckpt',
    'flow': 'data/checkpoints/flow_scratch/model.ckpt',
    'rgb_imagenet': 'data/checkpoints/rgb_imagenet/model.ckpt',
    'flow_imagenet': 'data/checkpoints/flow_imagenet/model.ckpt',
}

_LABEL_MAP_PATH = 'data/label_map.txt'

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('eval_type', 'joint', 'rgb, flow, or joint')
tf.flags.DEFINE_boolean('imagenet_pretrained', True, '')


def load_tensorflow(h5_dir='param/'):
    eval_type = FLAGS.eval_type
    imagenet_pretrained = FLAGS.imagenet_pretrained

    if eval_type not in ['rgb', 'flow', 'joint']:
        raise ValueError('Bad `eval_type`, must be one of rgb, flow, joint')

    rgb_saver = None
    if eval_type in ['rgb', 'joint']:
        # RGB input has 3 channels.
        rgb_input = tf.placeholder(
            tf.float32,
            shape=(1, _SAMPLE_VIDEO_FRAMES, _IMAGE_SIZE, _IMAGE_SIZE, 3))
        with tf.variable_scope('RGB'):
            rgb_model = i3d.InceptionI3d(
                _NUM_CLASSES, spatial_squeeze=True, final_endpoint='Logits')
            rgb_logits, _ = rgb_model(
                rgb_input, is_training=False, dropout_keep_prob=1.0)
        rgb_variable_map = {}
        for variable in tf.global_variables():
            if variable.name.split('/')[0] == 'RGB':
                rgb_variable_map[variable.name.replace(':0', '')] = variable
            print(variable.name)
            print(variable.value(), '\n')
        rgb_saver = tf.train.Saver(var_list=rgb_variable_map, reshape=True)

    flow_saver = None
    if eval_type in ['flow', 'joint']:
        # Flow input has only 2 channels.
        flow_input = tf.placeholder(
            tf.float32,
            shape=(1, _SAMPLE_VIDEO_FRAMES, _IMAGE_SIZE, _IMAGE_SIZE, 2))
        with tf.variable_scope('Flow'):
            flow_model = i3d.InceptionI3d(
                _NUM_CLASSES, spatial_squeeze=True, final_endpoint='Logits')
            flow_logits, _ = flow_model(
                flow_input, is_training=False, dropout_keep_prob=1.0)
        flow_variable_map = {}
        for variable in tf.global_variables():
            if variable.name.split('/')[0] == 'Flow':
                flow_variable_map[variable.name.replace(':0', '')] = variable
        flow_saver = tf.train.Saver(var_list=flow_variable_map, reshape=True)

    with tf.Session() as sess:
        if eval_type in ['rgb', 'joint']:
            if imagenet_pretrained:
                rgb_saver.restore(sess, _CHECKPOINT_PATHS['rgb_imagenet'])
            else:
                rgb_saver.restore(sess, _CHECKPOINT_PATHS['rgb'])
            tf.logging.info('RGB checkpoint restored')
            dump_i3d(sess, 'RGB', h5_dir)
            tf.logging.info('RGB parameters dumped')

        if eval_type in ['flow', 'joint']:
            if imagenet_pretrained:
                flow_saver.restore(sess, _CHECKPOINT_PATHS['flow_imagenet'])
            else:
                flow_saver.restore(sess, _CHECKPOINT_PATHS['flow'])
            tf.logging.info('Flow checkpoint restored')
            dump_i3d(sess, 'Flow', h5_dir)
            tf.logging.info('Flow parameters dumped')


def make_padding(padding_name, kernel):
    """Return padding as a list.

    padding_name: string, one of VALID, SAME.
    kernel: int, kernel width.
    """
    padding_name = padding_name.decode("utf-8")

    if padding_name == "VALID":
        return 0
    elif padding_name == "SAME":
        return int(math.ceil((kernel - 1) / 2))
    else:
        raise ValueError('Bad padding_name, must be one of VALID, SAME')


def dump_unit3d(sess, eval_type, name, h5_dir, use_bn=True):
    weight = sess.graph.get_tensor_by_name(
        '{:s}/inception_i3d/{:s}/conv_3d/w:0'.format(eval_type, name)).eval()
    if use_bn:
        beta = sess.graph.get_tensor_by_name(
            '{:s}/inception_i3d/{:s}/batch_norm/beta:0'.format(eval_type, name)).eval()
        mv_var = sess.graph.get_tensor_by_name(
            '{:s}/inception_i3d/{:s}/batch_norm/moving_variance:0'.format(eval_type, name)).eval()
        mv_mean = sess.graph.get_tensor_by_name(
            '{:s}/inception_i3d/{:s}/batch_norm/moving_mean:0'.format(eval_type, name)).eval()

    h5f = h5py.File(os.path.join(h5_dir, '{:s}_{:s}.h5'.format(eval_type, name)), 'w')
    h5f.create_dataset("weight", data=weight)
    if use_bn:
        h5f.create_dataset("beta", data=beta)
        h5f.create_dataset("mv_var", data=mv_var)
        h5f.create_dataset("mv_mean", data=mv_mean)
    h5f.close()


def dump_logits(sess, eval_type, name, h5_dir):
    weight = sess.graph.get_tensor_by_name(
        '{:s}/inception_i3d/Logits/{:s}/conv_3d/w:0'.format(eval_type, name)).eval()
    bias = sess.graph.get_tensor_by_name(
        '{:s}/inception_i3d/Logits/{:s}/conv_3d/b:0'.format(eval_type, name)).eval()

    h5f = h5py.File(os.path.join(h5_dir, '{:s}_{:s}.h5'.format(eval_type, 'Logits')), 'w')
    h5f.create_dataset("weight", data=weight)
    h5f.create_dataset("bias", data=bias)
    h5f.close()


def dump_block(sess, eval_type, name, h5_dir):
    b0_weight = sess.graph.get_tensor_by_name(
        '{:s}/inception_i3d/{:s}/Branch_0/{:s}/conv_3d/w:0'.format(eval_type, name, 'Conv3d_0a_1x1')).eval()
    b0_beta = sess.graph.get_tensor_by_name(
        '{:s}/inception_i3d/{:s}/Branch_0/{:s}/batch_norm/beta:0'.format(eval_type, name, 'Conv3d_0a_1x1')).eval()
    b0_mv_var = sess.graph.get_tensor_by_name(
        '{:s}/inception_i3d/{:s}/Branch_0/{:s}/batch_norm/moving_variance:0'.format(eval_type, name, 'Conv3d_0a_1x1')).eval()
    b0_mv_mean = sess.graph.get_tensor_by_name(
        '{:s}/inception_i3d/{:s}/Branch_0/{:s}/batch_norm/moving_mean:0'.format(eval_type, name, 'Conv3d_0a_1x1')).eval()

    b1a_weight = sess.graph.get_tensor_by_name(
        '{:s}/inception_i3d/{:s}/Branch_1/{:s}/conv_3d/w:0'.format(eval_type, name, 'Conv3d_0a_1x1')).eval()
    b1a_beta = sess.graph.get_tensor_by_name(
        '{:s}/inception_i3d/{:s}/Branch_1/{:s}/batch_norm/beta:0'.format(eval_type, name, 'Conv3d_0a_1x1')).eval()
    b1a_mv_var = sess.graph.get_tensor_by_name(
        '{:s}/inception_i3d/{:s}/Branch_1/{:s}/batch_norm/moving_variance:0'.format(eval_type, name, 'Conv3d_0a_1x1')).eval()
    b1a_mv_mean = sess.graph.get_tensor_by_name(
        '{:s}/inception_i3d/{:s}/Branch_1/{:s}/batch_norm/moving_mean:0'.format(eval_type, name, 'Conv3d_0a_1x1')).eval()
    b1b_weight = sess.graph.get_tensor_by_name(
        '{:s}/inception_i3d/{:s}/Branch_1/{:s}/conv_3d/w:0'.format(eval_type, name, 'Conv3d_0b_3x3')).eval()
    b1b_beta = sess.graph.get_tensor_by_name(
        '{:s}/inception_i3d/{:s}/Branch_1/{:s}/batch_norm/beta:0'.format(eval_type, name, 'Conv3d_0b_3x3')).eval()
    b1b_mv_var = sess.graph.get_tensor_by_name(
        '{:s}/inception_i3d/{:s}/Branch_1/{:s}/batch_norm/moving_variance:0'.format(eval_type, name, 'Conv3d_0b_3x3')).eval()
    b1b_mv_mean = sess.graph.get_tensor_by_name(
        '{:s}/inception_i3d/{:s}/Branch_1/{:s}/batch_norm/moving_mean:0'.format(eval_type, name, 'Conv3d_0b_3x3')).eval()

    b2a_weight = sess.graph.get_tensor_by_name(
        '{:s}/inception_i3d/{:s}/Branch_2/{:s}/conv_3d/w:0'.format(eval_type, name, 'Conv3d_0a_1x1')).eval()
    b2a_beta = sess.graph.get_tensor_by_name(
        '{:s}/inception_i3d/{:s}/Branch_2/{:s}/batch_norm/beta:0'.format(eval_type, name, 'Conv3d_0a_1x1')).eval()
    b2a_mv_var = sess.graph.get_tensor_by_name(
        '{:s}/inception_i3d/{:s}/Branch_2/{:s}/batch_norm/moving_variance:0'.format(eval_type, name, 'Conv3d_0a_1x1')).eval()
    b2a_mv_mean = sess.graph.get_tensor_by_name(
        '{:s}/inception_i3d/{:s}/Branch_2/{:s}/batch_norm/moving_mean:0'.format(eval_type, name, 'Conv3d_0a_1x1')).eval()

    if name == 'Mixed_5b':
        # there is a bug, Conv3d_0a_3x3 should be Conv3d_0b_3x3, the author is careless
        b2b_weight = sess.graph.get_tensor_by_name(
            '{:s}/inception_i3d/{:s}/Branch_2/{:s}/conv_3d/w:0'.format(eval_type, name, 'Conv3d_0a_3x3')).eval()
        b2b_beta = sess.graph.get_tensor_by_name(
            '{:s}/inception_i3d/{:s}/Branch_2/{:s}/batch_norm/beta:0'.format(eval_type, name, 'Conv3d_0a_3x3')).eval()
        b2b_mv_var = sess.graph.get_tensor_by_name(
            '{:s}/inception_i3d/{:s}/Branch_2/{:s}/batch_norm/moving_variance:0'.format(eval_type, name, 'Conv3d_0a_3x3')).eval()
        b2b_mv_mean = sess.graph.get_tensor_by_name(
            '{:s}/inception_i3d/{:s}/Branch_2/{:s}/batch_norm/moving_mean:0'.format(eval_type, name, 'Conv3d_0a_3x3')).eval()
    else:
        b2b_weight = sess.graph.get_tensor_by_name(
            '{:s}/inception_i3d/{:s}/Branch_2/{:s}/conv_3d/w:0'.format(eval_type, name, 'Conv3d_0b_3x3')).eval()
        b2b_beta = sess.graph.get_tensor_by_name(
            '{:s}/inception_i3d/{:s}/Branch_2/{:s}/batch_norm/beta:0'.format(eval_type, name, 'Conv3d_0b_3x3')).eval()
        b2b_mv_var = sess.graph.get_tensor_by_name(
            '{:s}/inception_i3d/{:s}/Branch_2/{:s}/batch_norm/moving_variance:0'.format(eval_type, name, 'Conv3d_0b_3x3')).eval()
        b2b_mv_mean = sess.graph.get_tensor_by_name(
            '{:s}/inception_i3d/{:s}/Branch_2/{:s}/batch_norm/moving_mean:0'.format(eval_type, name, 'Conv3d_0b_3x3')).eval()

    b3_weight = sess.graph.get_tensor_by_name(
        '{:s}/inception_i3d/{:s}/Branch_3/{:s}/conv_3d/w:0'.format(eval_type, name, 'Conv3d_0b_1x1')).eval()
    b3_beta = sess.graph.get_tensor_by_name(
        '{:s}/inception_i3d/{:s}/Branch_3/{:s}/batch_norm/beta:0'.format(eval_type, name, 'Conv3d_0b_1x1')).eval()
    b3_mv_var = sess.graph.get_tensor_by_name(
        '{:s}/inception_i3d/{:s}/Branch_3/{:s}/batch_norm/moving_variance:0'.format(eval_type, name, 'Conv3d_0b_1x1')).eval()
    b3_mv_mean = sess.graph.get_tensor_by_name(
        '{:s}/inception_i3d/{:s}/Branch_3/{:s}/batch_norm/moving_mean:0'.format(eval_type, name, 'Conv3d_0b_1x1')).eval()


    h5f = h5py.File(os.path.join(h5_dir, '{:s}_{:s}.h5'.format(eval_type, name)), 'w')
    h5f.create_dataset("b0_weight", data=b0_weight)
    h5f.create_dataset("b0_beta", data=b0_beta)
    h5f.create_dataset("b0_mv_var", data=b0_mv_var)
    h5f.create_dataset("b0_mv_mean", data=b0_mv_mean)

    h5f.create_dataset("b1a_weight", data=b1a_weight)
    h5f.create_dataset("b1a_beta", data=b1a_beta)
    h5f.create_dataset("b1a_mv_var", data=b1a_mv_var)
    h5f.create_dataset("b1a_mv_mean", data=b1a_mv_mean)
    h5f.create_dataset("b1b_weight", data=b1b_weight)
    h5f.create_dataset("b1b_beta", data=b1b_beta)
    h5f.create_dataset("b1b_mv_var", data=b1b_mv_var)
    h5f.create_dataset("b1b_mv_mean", data=b1b_mv_mean)

    h5f.create_dataset("b2a_weight", data=b2a_weight)
    h5f.create_dataset("b2a_beta", data=b2a_beta)
    h5f.create_dataset("b2a_mv_var", data=b2a_mv_var)
    h5f.create_dataset("b2a_mv_mean", data=b2a_mv_mean)
    h5f.create_dataset("b2b_weight", data=b2b_weight)
    h5f.create_dataset("b2b_beta", data=b2b_beta)
    h5f.create_dataset("b2b_mv_var", data=b2b_mv_var)
    h5f.create_dataset("b2b_mv_mean", data=b2b_mv_mean)

    h5f.create_dataset("b3_weight", data=b3_weight)
    h5f.create_dataset("b3_beta", data=b3_beta)
    h5f.create_dataset("b3_mv_var", data=b3_mv_var)
    h5f.create_dataset("b3_mv_mean", data=b3_mv_mean)
    h5f.close()


def dump_i3d(sess, eval_type, h5_dir):
    # Conv3d 1
    dump_unit3d(sess, eval_type, 'Conv3d_1a_7x7', h5_dir)

    # Conv3d 2
    dump_unit3d(sess, eval_type, 'Conv3d_2b_1x1', h5_dir)
    dump_unit3d(sess, eval_type, 'Conv3d_2c_3x3', h5_dir)

    # Mixed 3
    dump_block(sess, eval_type, 'Mixed_3b', h5_dir)
    dump_block(sess, eval_type, 'Mixed_3c', h5_dir)

    # Mixed 4
    dump_block(sess, eval_type, 'Mixed_4b', h5_dir)
    dump_block(sess, eval_type, 'Mixed_4c', h5_dir)
    dump_block(sess, eval_type, 'Mixed_4d', h5_dir)
    dump_block(sess, eval_type, 'Mixed_4e', h5_dir)
    dump_block(sess, eval_type, 'Mixed_4f', h5_dir)

    # Mixed 5
    dump_block(sess, eval_type, 'Mixed_5b', h5_dir)
    dump_block(sess, eval_type, 'Mixed_5c', h5_dir)

    # Logits
    dump_logits(sess, eval_type, 'Conv3d_0c_1x1', h5_dir)


if __name__ == '__main__':
    load_tensorflow()
