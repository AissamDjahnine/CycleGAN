import torch
import torch.nn as nn
from torch.nn import init
import functools


def init_weights(net, init_gain=0.02):

    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            init.normal_(m.weight.data, 0.0, init_gain)

        if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)

        elif classname.find('BatchNorm3d') != -1:
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_gain=init_gain)

    return net


def define_G(input_nc, output_nc, ngf, norm=nn.BatchNorm3d, init_gain=0.02, gpu_ids=[]):
    """Create a generator

    Parameters:
        input_nc (int) -- the number of channels in input images
        output_nc (int) -- the number of channels in output images
        ngf (int) -- the number of filters in the last conv layer
        norm (str) -- the name of normalization layers used in the network: 3D batch
        init_gain (float)  -- scaling factor for normal
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a generator
    """
    norm_layer = norm

    net = Generator(input_nc, output_nc, ngf, norm_layer=norm_layer)

    return init_net(net, init_type, init_gain, gpu_ids)


class Generator(nn.Module):

    def __init__(self, input_nc=3, output_nc=3, ngf=32, norm_layer=nn.BatchNorm3d):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels_in in input images
            output_nc (int)     -- the number of channels_out in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
        """

        ## Encoder ( 3 3D-Resnet block )

        ## Decoder ( 3 3D-Resnet block
        #             Spatial Self-Attention
        #             3D-Resnet block
        #             3D Batch Norm + RelU
        #             3D Conv
        #          )

        super(Generator, self).__init__()

        model = []

        # Encoder :

        model += [ResnetBlock(input_nc, ngf, norm_layer=norm_layer, use_bias=use_bias)]
        model += [ResnetBlock(ngf, ngf * 16, norm_layer=norm_layer, use_bias=use_bias)]
        model += [ResnetBlock(ngf * 16, ngf * 16, norm_layer=norm_layer, use_bias=use_bias)]

        # Decoder :

        model += [ResnetBlock(ngf * 16, 8 * ngf, norm_layer=norm_layer, use_bias=use_bias)]
        model += [ResnetBlock(8 * ngf, 4 * ngf, norm_layer=norm_layer, use_bias=use_bias)]
        model += [ResnetBlock(4 * ngf, 2 * ngf, norm_layer=norm_layer, use_bias=use_bias)]

        # Spatial Self attention

        model += [SelfAttention(2 * ngf)]

        model += [ResnetBlock(2 * ngf, ngf, norm_layer=norm_layer, use_bias=use_bias)]
        model += [nn.BatchNorm3d(ngf),
                  nn.ReLU(True)
                  ]
        model += [nn.Conv3d(ngf, output_nc, kernel_size=3, stride=1, padding=1, bias=use_bias),
                  nn.Tanh()
                  ]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        """<forward>"""
        return self.model(input)


class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim_in, dim_out, norm_layer, use_bias=False):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections.

        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        """

        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim_in, dim_out, norm_layer, use_bias=False)

    def build_conv_block(self, dim_in, dim_out, norm_layer, use_bias=False):
        """Construct a convolutional block.

        Parameters:
            dim_in (int)        -- the number of channels_in in the conv layer.
            dim_out (int)       -- the number of channels_out in the conv layer.
            norm_layer          -- normalization layer
            use_bias (bool)     -- if the conv layer uses bias or not

        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))

        """
        conv_block = []

        padding = 1

        conv_block += [nn.Conv3d(dim_in, dim_out, kernel_size=3, stride=1, padding=padding, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        conv_block += [nn.Conv3d(dim_in, dim_out, kernel_size=3, stride=1, padding=padding, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out


class SelfAttention(nn.Module):
    """ Define a Self-attention Layer

    Inspired from SAGAN paper

    """
    def __init__(self, in_dim):
        super(Self_Attn, self).__init__()
        self.chanel_in = in_dim
        self.activation = activation

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)  # g(x)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)    # f(x)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)       # h(x)

        # Scale factor
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X T X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (with N = Width*Height)
        """
        batch_size, nc, t, width, height = x.size()
        # reshaping x to [ batch_size* , nc , width , height ]
        x = x.view(-1, nc, width, height)

        m_batch_size,nc, width, height = x.size()

        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X C X N
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)                       # B X C x N
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)                   # B X C X N

        energy = torch.bmm(proj_query, proj_key)

        attention = self.softmax(energy)  # B X N X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))

        out = out.view(batch_size, nc, t, width, height)

        out = self.gamma*out + x

        return out