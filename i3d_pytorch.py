"""Pytorch implementation of Inception-v1 Inflated 3D ConvNet used for Kinetics CVPR paper.

The model is introduced in:

  Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset
  Joao Carreira, Andrew Zisserman
  https://arxiv.org/pdf/1705.07750v1.pdf.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F


def calculate_padding(input_shape, kernel_shape, stride):
    """Calculate padding for tensor of input_shape."""
    assert len(input_shape) == len(kernel_shape) + 2
    assert len(kernel_shape) == len(stride)
    # batch_size, input_channels
    padding_shape = ()
    for i in range(len(input_shape)-1, 1, -1):
        id = input_shape[i]
        kd = kernel_shape[i - 2]
        s = stride[i - 2]
        if id % s == 0:
            padding = max(kd - s, 0)
        else:
            padding = max(kd - (id % s), 0)
        pf = padding // 2
        pb = padding - pf
        padding_shape += (pf, pb)
    return padding_shape


class Unit3D(nn.Module):
    """Basic unit containing Conv3D + BatchNorm + non-linearity."""

    def __init__(self, input_channels,
                 output_channels,
                 kernel_shape=(1, 1, 1),
                 stride=(1, 1, 1),
                 padding=(0, 0, 0),
                 activation_fn=F.relu,
                 cal_padding=True,
                 use_batch_norm=True,
                 use_bias=False):
        """Initializes Unit3D module.

        kernel_shape: tuple, (D, H, W).
        stride: tuple, (D, H, W)
        padding: tuple, (D, H, W)
        """
        super(Unit3D, self).__init__()
        self.use_batch_norm = use_batch_norm
        self.use_bias = use_bias
        self.activation_fn = activation_fn

        if padding != (0, 0, 0) and cal_padding:
            raise ValueError('Bad padding option, padding and cal_padding cannot be used at the same time.')
        self.cal_padding = cal_padding
        self.kernel_shape = kernel_shape
        self.stride = stride

        self.conv3d = nn.Conv3d(in_channels=input_channels,
                                out_channels=output_channels,
                                kernel_size=kernel_shape,
                                stride=stride,
                                padding=padding,
                                bias=self.use_bias)
        if self.use_batch_norm:
            # eps and momentum are set according to sonnet implementation of batchnorm.
            self.bn = nn.BatchNorm3d(num_features=output_channels,
                                     eps=1e-3,
                                     momentum=0.01,
                                     affine=True)

    def forward(self, inputs):
        """Connects the module to inputs.

        inputs: 5D FloatTensor, (N, C, D, H, W), inputs to the Unit3D component.
        returns: 5D FloatTensor, (N, C', D, H, W)
        """
        if self.cal_padding:
            padding = calculate_padding(inputs.size(), self.kernel_shape, self.stride)
            inputs = F.pad(inputs, padding, 'constant', 0)
        x = self.conv3d(inputs)
        ox = x
        if self.use_batch_norm:
            x = self.bn(x)
        if self.activation_fn is not None:
            x = self.activation_fn(x)
        return x   # , ox


class InceptionBlock(nn.Module):
    """Basic unit of Inception v1.
    See Figure 3 of paper:
    Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset for details."""

    def __init__(self, input_channels,
               output_channels,
               activation_fn=F.relu,
               use_batch_norm=True,
               use_bias=False):
        """Initialize basic unit of Inception v1.

        output_channels: list, [b0_out, b1a_out, b1b_out, b2a_out, b2b_out, b3_out]
        """
        self.use_batch_norm = use_batch_norm
        self.use_bias = use_bias
        super(InceptionBlock, self).__init__()
        self.branch_0 = Unit3D(input_channels, output_channels[0],
                               kernel_shape=(1, 1, 1), stride=(1, 1, 1),
                               padding=0, cal_padding=False,
                               use_batch_norm = self.use_batch_norm,
                               use_bias = self.use_bias)

        self.branch_1a = Unit3D(input_channels, output_channels[1],
                                kernel_shape=(1, 1, 1), stride=(1, 1, 1),
                                padding=0, cal_padding=False,
                                use_batch_norm=self.use_batch_norm,
                                use_bias=self.use_bias)

        self.branch_1b = Unit3D(output_channels[1], output_channels[2],
                                kernel_shape=(3, 3, 3), stride=(1, 1, 1),
                                padding=(1, 1, 1), cal_padding=False,
                                use_batch_norm=self.use_batch_norm,
                                use_bias=self.use_bias)

        self.branch_2a = Unit3D(input_channels, output_channels[3],
                                kernel_shape=(1, 1, 1), stride=(1, 1, 1),
                                padding=0, cal_padding=False,
                                use_batch_norm=self.use_batch_norm,
                                use_bias=self.use_bias)
        self.branch_2b = Unit3D(output_channels[3], output_channels[4],
                                kernel_shape=(3, 3, 3), stride=(1, 1, 1),
                                padding=(1, 1, 1), cal_padding=False,
                                use_batch_norm=self.use_batch_norm,
                                use_bias=self.use_bias)

        self.branch_3a = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(1, 1, 1),
                                      padding=(1, 1, 1))
        self.branch_3b = Unit3D(input_channels, output_channels[5],
                                kernel_shape=(1, 1, 1), stride=(1, 1, 1),
                                padding=0, cal_padding=False,
                                use_batch_norm=self.use_batch_norm,
                                use_bias=self.use_bias)

    def forward(self, inputs):
        """Connects the module to inputs.

        inputs: 5D FloatTensor, (N, C, D, H, W), inputs to the InceptionBlock component.
        returns: 5D FloatTensor, (N, C', D, H, W)
        """
        x_0 = self.branch_0(inputs)
        x_1 = self.branch_1a(inputs)
        x_1 = self.branch_1b(x_1)
        x_2 = self.branch_2a(inputs)
        x_2 = self.branch_2b(x_2)
        x_3 = self.branch_3a(inputs)
        x_3 = self.branch_3b(x_3)
        x = torch.cat((x_0, x_1, x_2, x_3), dim=1)
        return x


class InceptionI3D(nn.Module):
    """Inception-v1 I3D architecture."""

    # Endpoints of the model in order. During construction, all the endpoints up
    # to a designated `final_endpoint` are returned in a dictionary as the
    # second return value.
    VALID_ENDPOINTS = (
        'Conv3d_1a_7x7',
        'MaxPool3d_2a_3x3',
        'Conv3d_2b_1x1',
        'Conv3d_2c_3x3',
        'MaxPool3d_3a_3x3',
        'Mixed_3b',
        'Mixed_3c',
        'MaxPool3d_4a_3x3',
        'Mixed_4b',
        'Mixed_4c',
        'Mixed_4d',
        'Mixed_4e',
        'Mixed_4f',
        'MaxPool3d_5a_2x2',
        'Mixed_5b',
        'Mixed_5c',
        'Logits',
        'Predictions',
    )

    def __init__(self, input_channels=3,
                 num_classes=400,
                 dropout_prob = 0.0,
                 spatial_squeeze=True,
                 final_endpoint='Logits'):
        """Initializes I3D model instance.

        Args:
            num_classes: The number of outputs in the logit layer (default 400, which
                matches the Kinetics dataset).
            dropout_prob: Probability of dropout (float in [0, 1)).
            spatial_squeeze: Whether to squeeze the spatial dimensions for the logits
                before returning (default True).
            final_endpoint: The model contains many possible endpoints.
                `final_endpoint` specifies the last endpoint for the model to be built
                up to. In addition to the output at `final_endpoint`, all the outputs
                at endpoints up to `final_endpoint` will also be returned, in a
                dictionary. `final_endpoint` must be one of
                InceptionI3d.VALID_ENDPOINTS (default 'Logits').

        Raises:
            ValueError: if `final_endpoint` is not recognized.
        """
        if final_endpoint not in self.VALID_ENDPOINTS:
            raise ValueError('Unknown final endpoint %s' % final_endpoint)

        super(InceptionI3D, self).__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob
        self.spatial_squeeze = spatial_squeeze
        self.final_endpoint = final_endpoint

        # # Original padding scheme, different from TensorFlow and Sonnet
        # self.conv_1a = Unit3D(self.input_channels, output_channels=64,
        #                       kernel_shape=(7, 7, 7), stride=(2, 2, 2),
        #                       padding=(3, 3, 3))
        # self.maxpool_2a = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2),
        #                                padding=(0, 1, 1))
        # self.conv_2b = Unit3D(64, output_channels=64,
        #                       kernel_shape=(1, 1, 1), stride=(1, 1, 1))
        # self.conv_2c = Unit3D(64, output_channels=192,
        #                       kernel_shape=(3, 3, 3), stride=(1, 1, 1),
        #                       padding=(1, 1, 1))
        # self.maxpool_3a = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2),
        #                                padding=(0, 1, 1))
        # self.mixed_3b = InceptionBlock(192, output_channels=[64, 96, 128, 16, 32, 32])
        # self.mixed_3c = InceptionBlock(256, output_channels=[128, 128, 192, 32, 96, 64])
        # self.maxpool_4a = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(2, 2, 2),
        #                                padding=(1, 1, 1))
        # self.mixed_4b = InceptionBlock(480, output_channels=[192, 96, 208, 16, 48, 64])
        # self.mixed_4c = InceptionBlock(512, output_channels=[160, 112, 224, 24, 64, 64])
        # self.mixed_4d = InceptionBlock(512, output_channels=[128, 128, 256, 24, 64, 64])
        # self.mixed_4e = InceptionBlock(512, output_channels=[112, 144, 288, 32, 64, 64])
        # self.mixed_4f = InceptionBlock(528, output_channels=[256, 160, 320, 32, 128, 128])
        # self.maxpool_5a = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        # self.mixed_5b = InceptionBlock(832, output_channels=[256, 160, 320, 32, 128, 128])
        # self.mixed_5c = InceptionBlock(832, output_channels=[384, 192, 384, 48, 128, 128])
        # self.avgpool = nn.AvgPool3d(kernel_size=(2, 7, 7), stride=(1, 1, 1))
        # self.logits =  Unit3D(1024, output_channels=self.num_classes,
        #                       kernel_shape=(1, 1, 1), stride=(1, 1, 1),
        #                       use_batch_norm=False, use_bias=True)

        # New padding scheme, same as TensorFlow and Sonnet
        self.conv_1a = Unit3D(self.input_channels, output_channels=64,
                              kernel_shape=(7, 7, 7), stride=(2, 2, 2))
        self.maxpool_2a = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=0)
        self.conv_2b = Unit3D(64, output_channels=64,
                              kernel_shape=(1, 1, 1), stride=(1, 1, 1))
        self.conv_2c = Unit3D(64, output_channels=192,
                              kernel_shape=(3, 3, 3), stride=(1, 1, 1))
        self.maxpool_3a = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=0)
        self.mixed_3b = InceptionBlock(192, output_channels=[64, 96, 128, 16, 32, 32])
        self.mixed_3c = InceptionBlock(256, output_channels=[128, 128, 192, 32, 96, 64])
        self.maxpool_4a = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=0)
        self.mixed_4b = InceptionBlock(480, output_channels=[192, 96, 208, 16, 48, 64])
        self.mixed_4c = InceptionBlock(512, output_channels=[160, 112, 224, 24, 64, 64])
        self.mixed_4d = InceptionBlock(512, output_channels=[128, 128, 256, 24, 64, 64])
        self.mixed_4e = InceptionBlock(512, output_channels=[112, 144, 288, 32, 64, 64])
        self.mixed_4f = InceptionBlock(528, output_channels=[256, 160, 320, 32, 128, 128])
        self.maxpool_5a = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=0)
        self.mixed_5b = InceptionBlock(832, output_channels=[256, 160, 320, 32, 128, 128])
        self.mixed_5c = InceptionBlock(832, output_channels=[384, 192, 384, 48, 128, 128])
        self.avgpool = nn.AvgPool3d(kernel_size=(2, 7, 7), stride=(1, 1, 1))
        self.logits = Unit3D(1024, output_channels=self.num_classes,
                             kernel_shape=(1, 1, 1), stride=(1, 1, 1),
                             padding=0, cal_padding=False,
                             activation_fn=None,
                             use_batch_norm=False, use_bias=True)

    def forward(self, inputs):
        """Connects the model to inputs.

        Args:
            inputs: Inputs to the model, which should have dimensions (N, C, D, H, W)
                `batch_size` x `num_channels` x `num_frames` x 224 x 224.
                For the Tensorflow implementation, inputs should have dimension (N, D, H, W, C)

        Returns:
            A tuple consisting of:
            1. Network output at location `self.final_endpoint`.
            2. Dictionary containing all endpoints up to `self.final_endpoint`,
               indexed by endpoint name.
        """
        end_points = {}
        end_point = 'Conv3d_1a_7x7'
        x = self.conv_1a(inputs)
        end_points[end_point] = x
        if self.final_endpoint == end_point: return x, end_points

        end_point = 'MaxPool3d_2a_3x3'
        padding = calculate_padding(x.size(), self.maxpool_2a.kernel_size, self.maxpool_2a.stride)
        x = F.pad(x, padding, "constant", 0)
        x = self.maxpool_2a(x)
        end_points[end_point] = x
        if self.final_endpoint == end_point: return x, end_points
        end_point = 'Conv3d_2b_1x1'
        x = self.conv_2b(x)
        end_points[end_point] = x
        if self.final_endpoint == end_point: return x, end_points
        end_point = 'Conv3d_2c_3x3'
        x = self.conv_2c(x)
        end_points[end_point] = x
        if self.final_endpoint == end_point: return x, end_points

        end_point = 'MaxPool3d_3a_3x3'
        padding = calculate_padding(x.size(), self.maxpool_3a.kernel_size, self.maxpool_3a.stride)
        x = F.pad(x, padding, "constant", 0)
        x = self.maxpool_3a(x)
        end_points[end_point] = x
        if self.final_endpoint == end_point: return x, end_points
        end_point = 'Mixed_3b'
        x = self.mixed_3b(x)
        end_points[end_point] = x
        if self.final_endpoint == end_point: return x, end_points
        end_point = 'Mixed_3c'
        x = self.mixed_3c(x)
        end_points[end_point] = x
        if self.final_endpoint == end_point: return x, end_points

        end_point = 'MaxPool3d_4a_3x3'
        padding = calculate_padding(x.size(), self.maxpool_4a.kernel_size, self.maxpool_4a.stride)
        x = F.pad(x, padding, "constant", 0)
        x = self.maxpool_4a(x)
        end_points[end_point] = x
        if self.final_endpoint == end_point: return x, end_points
        end_point = 'Mixed_4b'
        x = self.mixed_4b(x)
        end_points[end_point] = x
        if self.final_endpoint == end_point: return x, end_points
        end_point = 'Mixed_4c'
        x = self.mixed_4c(x)
        end_points[end_point] = x
        if self.final_endpoint == end_point: return x, end_points
        end_point = 'Mixed_4d'
        x = self.mixed_4d(x)
        end_points[end_point] = x
        if self.final_endpoint == end_point: return x, end_points
        end_point = 'Mixed_4e'
        x = self.mixed_4e(x)
        end_points[end_point] = x
        if self.final_endpoint == end_point: return x, end_points
        end_point = 'Mixed_4f'
        x = self.mixed_4f(x)
        end_points[end_point] = x
        if self.final_endpoint == end_point: return x, end_points

        end_point = 'MaxPool3d_5a_2x2'
        padding = calculate_padding(x.size(), self.maxpool_5a.kernel_size, self.maxpool_5a.stride)
        x = F.pad(x, padding, "constant", 0)
        x = self.maxpool_5a(x)
        end_points[end_point] = x
        if self.final_endpoint == end_point: return x, end_points
        end_point = 'Mixed_5b'
        x = self.mixed_5b(x)
        end_points[end_point] = x
        if self.final_endpoint == end_point: return x, end_points
        end_point = 'Mixed_5c'
        x = self.mixed_5c(x)
        end_points[end_point] = x
        if self.final_endpoint == end_point: return x, end_points

        end_point = 'Logits'
        x = self.avgpool(x)
        end_points['Features'] = x
        print('x shape after avgpool: {}'.format(x.size()))
        x = F.dropout(x, p=self.dropout_prob, training=self.training)
        logits = self.logits(x)
        if self.spatial_squeeze:
            logits = torch.squeeze(logits, dim=4)
            logits = torch.squeeze(logits, dim=3) # (N, C', D'), c' equals to num_classes

        averaged_logits = torch.mean(logits, dim=2)   # (N, C')
        end_points[end_point] = averaged_logits
        if self.final_endpoint == end_point: return averaged_logits, end_points

        end_point = 'Predictions'
        predictions = F.softmax(averaged_logits)  # softmax along 1st dimension
        end_points[end_point] = predictions
        return predictions, end_points


if __name__ == "__main__":
    padding_shape = calculate_padding((1, 3, 79, 224, 224), (7, 7, 7), (2, 2, 2))
    print(padding_shape)