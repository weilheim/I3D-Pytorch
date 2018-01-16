from __future__ import absolute_import
from __future__ import division

import h5py
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import i3d_pytorch


def load_unit3d(state_dict, eval_type, src_name, tgt_name, h5_dir,
                use_batch_norm=True, use_bias=False):
    """Load parameters of Unit3D component into state_dict.

    Pytorch conv3d weight: (C_out, C_in, D, H, W)
    Sonnet conv3d weight: (D, H, W, C_in, C_out)
    """
    h5f = h5py.File(os.path.join(h5_dir, '{:s}_{:s}.h5'.format(eval_type, src_name)), 'r')

    state_dict[tgt_name + '.conv3d.weight'] = torch.from_numpy(h5f['weight'][...]).permute(4, 3, 0, 1, 2)
    if use_bias:
        # print torch.from_numpy(h5f['bias'][...]).size()
        state_dict[tgt_name + '.conv3d.bias'] = torch.from_numpy(h5f['bias'][...])
        # print 'bias shape:', np.array(h5f['bias'][...]).shape
    if use_batch_norm:
        output_channels = state_dict[tgt_name + '.conv3d.weight'].size(0)
        # Sonnet model does not scale batch norm output, it only adds bias.
        state_dict[tgt_name + '.bn.weight'] = torch.ones(output_channels)
        state_dict[tgt_name + '.bn.bias'] = torch.from_numpy(h5f['beta'][...]).squeeze()
        state_dict[tgt_name + '.bn.running_mean'] = torch.from_numpy(h5f['mv_mean'][...]).squeeze()
        state_dict[tgt_name + '.bn.running_var'] = torch.from_numpy(h5f['mv_var'][...]).squeeze()
    h5f.close()


def load_block(state_dict, eval_type, src_name, tgt_name, h5_dir):
    """Load parameters of InceptionBlock component into state_dict."""
    h5f = h5py.File(os.path.join(h5_dir, '{:s}_{:s}.h5'.format(eval_type, src_name)), 'r')

    state_dict[tgt_name + '.branch_0.conv3d.weight'] = torch.from_numpy(h5f['b0_weight'][...]).permute(4, 3, 0, 1, 2)
    output_channels = state_dict[tgt_name + '.branch_0.conv3d.weight'].size(0)
    # Sonnet model does not scale batch norm output, it only adds bias.
    state_dict[tgt_name + '.branch_0.bn.weight'] = torch.ones(output_channels)
    state_dict[tgt_name + '.branch_0.bn.bias'] = torch.from_numpy(h5f['b0_beta'][...])
    state_dict[tgt_name + '.branch_0.bn.running_mean'] = torch.from_numpy(h5f['b0_mv_mean'][...])
    state_dict[tgt_name + '.branch_0.bn.running_var'] = torch.from_numpy(h5f['b0_mv_var'][...])

    state_dict[tgt_name + '.branch_1a.conv3d.weight'] = torch.from_numpy(h5f['b1a_weight'][...]).permute(4, 3, 0, 1, 2)
    output_channels = state_dict[tgt_name + '.branch_1a.conv3d.weight'].size(0)
    # Sonnet model does not scale batch norm output, it only adds bias.
    state_dict[tgt_name + '.branch_1a.bn.weight'] = torch.ones(output_channels)
    state_dict[tgt_name + '.branch_1a.bn.bias'] = torch.from_numpy(h5f['b1a_beta'][...])
    state_dict[tgt_name + '.branch_1a.bn.running_mean'] = torch.from_numpy(h5f['b1a_mv_mean'][...])
    state_dict[tgt_name + '.branch_1a.bn.running_var'] = torch.from_numpy(h5f['b1a_mv_var'][...])
    state_dict[tgt_name + '.branch_1b.conv3d.weight'] = torch.from_numpy(h5f['b1b_weight'][...]).permute(4, 3, 0, 1, 2)
    output_channels = state_dict[tgt_name + '.branch_1b.conv3d.weight'].size(0)
    # Sonnet model does not scale batch norm output, it only adds bias.
    state_dict[tgt_name + '.branch_1b.bn.weight'] = torch.ones(output_channels)
    state_dict[tgt_name + '.branch_1b.bn.bias'] = torch.from_numpy(h5f['b1b_beta'][...])
    state_dict[tgt_name + '.branch_1b.bn.running_mean'] = torch.from_numpy(h5f['b1b_mv_mean'][...])
    state_dict[tgt_name + '.branch_1b.bn.running_var'] = torch.from_numpy(h5f['b1b_mv_var'][...])

    state_dict[tgt_name + '.branch_2a.conv3d.weight'] = torch.from_numpy(h5f['b2a_weight'][...]).permute(4, 3, 0, 1, 2)
    output_channels = state_dict[tgt_name + '.branch_2a.conv3d.weight'].size(0)
    # Sonnet model does not scale batch norm output, it only adds bias.
    state_dict[tgt_name + '.branch_2a.bn.weight'] = torch.ones(output_channels)
    state_dict[tgt_name + '.branch_2a.bn.bias'] = torch.from_numpy(h5f['b2a_beta'][...])
    state_dict[tgt_name + '.branch_2a.bn.running_mean'] = torch.from_numpy(h5f['b2a_mv_mean'][...])
    state_dict[tgt_name + '.branch_2a.bn.running_var'] = torch.from_numpy(h5f['b2a_mv_var'][...])
    state_dict[tgt_name + '.branch_2b.conv3d.weight'] = torch.from_numpy(h5f['b2b_weight'][...]).permute(4, 3, 0, 1, 2)
    output_channels = state_dict[tgt_name + '.branch_2b.conv3d.weight'].size(0)
    # Sonnet model does not scale batch norm output, it only adds bias.
    state_dict[tgt_name + '.branch_2b.bn.weight'] = torch.ones(output_channels)
    state_dict[tgt_name + '.branch_2b.bn.bias'] = torch.from_numpy(h5f['b2b_beta'][...])
    state_dict[tgt_name + '.branch_2b.bn.running_mean'] = torch.from_numpy(h5f['b2b_mv_mean'][...])
    state_dict[tgt_name + '.branch_2b.bn.running_var'] = torch.from_numpy(h5f['b2b_mv_var'][...])

    state_dict[tgt_name + '.branch_3b.conv3d.weight'] = torch.from_numpy(h5f['b3_weight'][...]).permute(4, 3, 0, 1, 2)
    output_channels = state_dict[tgt_name + '.branch_3b.conv3d.weight'].size(0)
    # Sonnet model does not scale batch norm output, it only adds bias.
    state_dict[tgt_name + '.branch_3b.bn.weight'] = torch.ones(output_channels)
    state_dict[tgt_name + '.branch_3b.bn.bias'] = torch.from_numpy(h5f['b3_beta'][...])
    state_dict[tgt_name + '.branch_3b.bn.running_mean'] = torch.from_numpy(h5f['b3_mv_mean'][...])
    state_dict[tgt_name + '.branch_3b.bn.running_var'] = torch.from_numpy(h5f['b3_mv_var'][...])


def load_i3d(eval_type, h5_dir='param/'):
    """Load all the parameters of InceptionI3D model into state_dict."""
    state_dict = {}
    load_unit3d(state_dict, eval_type, 'Conv3d_1a_7x7', 'conv_1a', h5_dir)

    load_unit3d(state_dict, eval_type, 'Conv3d_2b_1x1', 'conv_2b', h5_dir)
    load_unit3d(state_dict, eval_type, 'Conv3d_2c_3x3', 'conv_2c', h5_dir)

    load_block(state_dict, eval_type, 'Mixed_3b', 'mixed_3b', h5_dir)
    load_block(state_dict, eval_type, 'Mixed_3c', 'mixed_3c', h5_dir)

    load_block(state_dict, eval_type, 'Mixed_4b', 'mixed_4b', h5_dir)
    load_block(state_dict, eval_type, 'Mixed_4c', 'mixed_4c', h5_dir)
    load_block(state_dict, eval_type, 'Mixed_4d', 'mixed_4d', h5_dir)
    load_block(state_dict, eval_type, 'Mixed_4e', 'mixed_4e', h5_dir)
    load_block(state_dict, eval_type, 'Mixed_4f', 'mixed_4f', h5_dir)

    load_block(state_dict, eval_type, 'Mixed_5b', 'mixed_5b', h5_dir)
    load_block(state_dict, eval_type, 'Mixed_5c', 'mixed_5c', h5_dir)

    load_unit3d(state_dict, eval_type, 'Logits', 'logits', h5_dir,
                use_batch_norm=False, use_bias=True)
    return state_dict


if __name__ == "__main__":
    _SAMPLE_PATHS = {
        'rgb': 'data/v_CricketShot_g04_c01_rgb.npy',
        'flow': 'data/v_CricketShot_g04_c01_flow.npy',
    }
    _LABEL_MAP_PATH = 'data/label_map.txt'
    kinetics_classes = [x.strip() for x in open(_LABEL_MAP_PATH)]
    rgb_sample = np.load(_SAMPLE_PATHS['rgb'])
    flow_sample = np.load(_SAMPLE_PATHS['flow'])
    print rgb_sample.shape   # (1, 79, 224, 224, 3)

    i3d_model = i3d_pytorch.InceptionI3D(input_channels=2,
                                         num_classes=400,
                                         dropout_prob=0.0,
                                         spatial_squeeze=True,
                                         final_endpoint='Logits')
    # state_dict = load_i3d('RGB')
    state_dict = load_i3d('Flow')
    # print '\n'.join(state_dict.keys())
    #
    # print '\n'
    # print '\n'.join(i3d_model.state_dict().keys())
    i3d_model.load_state_dict(state_dict)
    i3d_model.eval()


    # rgb_sample = Variable(torch.from_numpy(rgb_sample).permute(0, 4, 1, 2, 3))
    # out_logits, out_prevs = i3d_model(rgb_sample)
    flow_sample = Variable(torch.from_numpy(flow_sample).permute(0, 4, 1, 2, 3))
    out_logits, out_prevs = i3d_model(flow_sample)
    out_predictions = F.softmax(out_logits, dim=1)
    # k0, k1, k2, 64
    # 64, 3, k0, k1, k2
    logits = np.squeeze(out_prevs['Logits'].data.numpy())  # .permute(0, 2, 3, 4, 1).data.numpy()
    predictions = np.squeeze(out_predictions.data.numpy())
    features = out_prevs['Features'].permute(0, 2, 3, 4, 1).data.numpy()   # .permute(0, 2, 3, 4, 1).data.numpy()
    # tf_logits = np.squeeze(np.load('Logits.npy'))
    # tf_features = np.squeeze(np.load('Features.npy'))
    # print 'py logits:\n', logits
    # print 'tf logits:\n', tf_logits
    # print np.abs(logits - tf_logits)
    # print np.abs(features - tf_features)
    # print 'diff logits:\n', np.sum(np.abs(logits - tf_logits))
    # print 'diff avgpool:\n', np.sum(np.abs(np.squeeze(features) - np.squeeze(tf_features)))

    sorted_indices = np.argsort(predictions)[::-1]

    print('Norm of logits: %f' % np.linalg.norm(logits))
    print('\nTop classes and probabilities')
    for index in sorted_indices[:20]:
        print(predictions[index], logits[index], kinetics_classes[index])
