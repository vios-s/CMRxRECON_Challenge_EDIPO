from typing import List, Tuple
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.append("..")
from utils.fft import *
from utils.math import *
from .datalayer import DCLayer

class CineNet_RNN_diff(nn.Module):
    """
    A hybrid model for Dynamic MRI Reconstruction, inspired by combining
    CineNet [1] and Recurrent Convolutional Neural Networks (RCNN) [2].
    
    Reference papers:
    [1] A. Kofler et al. `An end-to-end-trainable iterative network architecture 
        for accelerated radial multi-coil 2D cine MR image reconstruction.`
        In Medical Physics, 2021.
    [2] C. Qin et al. `Convolutional Recurrent Neural Networks for Dynamic MR
        Image Reconstruction`. In IEEE Transactions on Medical Imaging 38.1,
        pp. 280–290, 2019.
    """
    def __init__(
        self,
        num_cascades: int = 10,
        CG_iters: int = 4,
        chans: int = 64,
        datalayer= DCLayer,
    ):
        """
        Args:
            num_cascades: Number of alternations between CG and RCNN modules.
            CG_iters: Number of  CG iterations in the CG module.
            chans: Number of channels for convolutional layers of the RCNN.
        """
        super(CineNet_RNN_diff, self).__init__()
        
        self.num_cascades = num_cascades
        self.CG_iters = CG_iters
        self.chans = chans
        self.datalayer = datalayer
        
        self.bcrnn = BCRNNlayer(input_size=2, hidden_size=self.chans, kernel_size=3)
        self.bcrnn2 = BCRNNlayer(input_size=2, hidden_size=self.chans, kernel_size=3)
        self.conv1_x = nn.Conv2d(self.chans, self.chans, 3, padding = 3//2)
        self.conv1_h = nn.Conv2d(self.chans, self.chans, 3, padding = 3//2)
        self.conv2_x = nn.Conv2d(self.chans, self.chans, 3, padding = 3//2)
        self.conv2_h = nn.Conv2d(self.chans, self.chans, 3, padding = 3//2)
        self.conv3_x = nn.Conv2d(self.chans, self.chans, 3, padding = 3//2)
        self.conv3_h = nn.Conv2d(self.chans, self.chans, 3, padding = 3//2)
        self.conv4_x = nn.Conv2d(self.chans, 2, 3, padding = 3//2)
        self.relu = nn.ReLU(inplace=True)
        dcs = []
        # self.Softplus = nn.Softplus(1.) 
        lambda_init = np.log(np.exp(1)-1.)/1.
        #self.lambda_reg = nn.Parameter(lambda_init*torch.ones(1), requires_grad=True)
        for i in range(self.num_cascades):
            dcs.append(self.datalayer)
            #dcs.append(cl.DataConsistencyInKspace(norm='ortho'))
        self.dcs = dcs


    def forward(self, x, y, mask, acc):
        """
        Args:
            x: Input image, shape (b, w, h, t, ch)
            y: Input k-space, shape (b, w, h, t, ch)
            mask: Input mask, shape (b, w, h, ch)
            acc: Acceleration Rate
        Returns:
            x: Reconstructed image, shape (b, t, w, h, ch)
        """
        if acc==4:
            iter_num = self.num_cascades-3
        elif acc==8:
            iter_num = self.num_cascades-2
        else:
            iter_num = self.num_cascades

        # print(iter_num,acc)
        # x_ref = self.sens_reduce(ref_kspace, sens_maps)
        #
        # x = x_ref.clone().squeeze(2).permute(0, 4, 2, 3, 1)
        x = x.clone().permute(0, 4, 1, 2, 3)
        y = y.clone().permute(0, 4, 1, 2, 3)
        x = x.float()
        b, ch, w, h, t = x.size()

        mask = mask.unsqueeze(-1).float()
        # mask = mask.clone().permute(0, 4, 1, 2, 3)
        # b, ch, h, w, t = x.size()
        size_h = [t * b, self.chans, w, h]

        # Initialise parameters of rcnn layers at the first iteration to zero
        net = {}
        rcnn_layers = 6
        for j in range(rcnn_layers - 1):
            net['t0_x%d' % j] = torch.zeros(size_h).to(x.device)

        # Recurrence through iterations
        for i in range(1, iter_num + 1):

            x = x.permute(4,0,1,2,3) #[t b ch w h]
            x = x.contiguous()

            net['t%d_x0' % (i - 1)] = net['t%d_x0' % (i - 1)].view(t, b, self.chans, w, h)
            # print(net['t%d_x0' % (i - 1)].get_device(), x.get_device(), y.get_device(), mask.get_device())
            net['t%d_x0' % i] = self.bcrnn(x, net['t%d_x0' % (i - 1)])
            net['t%d_x1' % i] = self.bcrnn2(x, net['t%d_x0' % i])
            net['t%d_x1' % i] = net['t%d_x1' % i].view(-1, self.chans, w, h)

            net['t%d_x2' % i] = self.conv1_x(net['t%d_x1' % i])
            net['t%d_h2' % i] = self.conv1_h(net['t%d_x2' % (i - 1)])
            net['t%d_x2' % i] = self.relu(net['t%d_h2' % i] + net['t%d_x2' % i])

            net['t%d_x3' % i] = self.conv2_x(net['t%d_x2' % i])
            net['t%d_h3' % i] = self.conv2_h(net['t%d_x3' % (i - 1)])
            net['t%d_x3' % i] = self.relu(net['t%d_h3' % i] + net['t%d_x3' % i])

            net['t%d_x4' % i] = self.conv3_x(net['t%d_x3' % i])
            net['t%d_h4' % i] = self.conv3_h(net['t%d_x4' % (i - 1)])
            net['t%d_x4' % i] = self.relu(net['t%d_h4' % i] + net['t%d_x4' % i])
            net['t%d_x5' % i] = self.conv4_x(net['t%d_x4' % i])
            x = x.view(-1, ch, w, h)
            net['t%d_out' % i] = x + net['t%d_x5' % i]

            net['t%d_out' % i] = net['t%d_out' % i].view(-1, b, ch, w, h)

            net['t%d_out' % i] = net['t%d_out' % i].permute(1, 3, 4, 0, 2)
            net['t%d_out' % i].contiguous()
            #print(net['t%d_out' % i].shape)
            net['t%d_out' % i] = self.dcs[i - 1](net['t%d_out' % i], y.permute(0, 2, 3, 4, 1), mask).permute(0, 4, 1, 2, 3)
            net['t%d_out' % i] = net['t%d_out' % i]

            x = net['t%d_out' % i]

        out = net['t%d_out' % i]

        return out.permute(0,2,3,4,1)


class CRNNcell(nn.Module):
    """
    Convolutional RNN cell that evolves over both time and iterations.
    """
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        kernel_size: int,
    ):
        """
        Args:
            input_size: Number of input channels
            hidden_size: Number of RCNN hidden layers channels
            kernel_size: Size of convolutional kernel
        """
        super(CRNNcell, self).__init__()
        
        # Convolution for input
        self.i2h = nn.Conv2d(input_size, hidden_size, kernel_size, padding=kernel_size // 2)
        # Convolution for hidden states in temporal dimension
        self.h2h = nn.Conv2d(hidden_size, hidden_size, kernel_size, padding=kernel_size // 2)
        # Convolution for hidden states in iteration dimension
        self.ih2ih = nn.Conv2d(hidden_size, hidden_size, kernel_size, padding=kernel_size // 2)
        
        self.relu = nn.ReLU(inplace=True)

    def forward(
        self,
        input: torch.Tensor,
        hidden_iteration: torch.Tensor,
        hidden: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            input: Input 4D tensor of shape `(b, ch, h, w)`
            hidden_iteration: hidden states in iteration dimension, 4d tensor of shape (b, hidden_size, h, w)
            hidden: hidden states in temporal dimension, 4d tensor of shape (b, hidden_size, h, w)
        Returns:
            Output tensor of shape `(b, hidden_size, h, w)`.
        """
        in_to_hid = self.i2h(input)
        hid_to_hid = self.h2h(hidden)
        ih_to_ih = self.ih2ih(hidden_iteration)

        hidden = self.relu(in_to_hid + hid_to_hid + ih_to_ih)

        return hidden


class BCRNNlayer(nn.Module):
    """
    Bidirectional Convolutional RNN layer
    """
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        kernel_size: int,
    ):
        """
        Args:
            input_size: Number of input channels
            hidden_size: Number of RCNN hidden layers channels
            kernel_size: Size of convolutional kernel
        """
        super(BCRNNlayer, self).__init__()
        
        self.hidden_size = hidden_size
        self.CRNN_model = CRNNcell(input_size, self.hidden_size, kernel_size)

    def forward(self, input: torch.Tensor, hidden_iteration: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: Input 5D tensor of shape `(t, b, ch, h, w)`
            hidden_iteration: hidden states (output of BCRNNlayer) from previous
                    iteration, 5d tensor of shape (t, b, hidden_size, h, w)
        Returns:
            Output tensor of shape `(t, b, hidden_size, h, w)`.
        """
        t, b, ch, h, w = input.shape
        size_h = [b, self.hidden_size, h, w]
        
        hid_init = torch.zeros(size_h).to(input.device)
        output_f = []
        output_b = []
        
        # forward
        hidden = hid_init
        for i in range(t):
            hidden = self.CRNN_model(input[i], hidden_iteration[i], hidden)
            output_f.append(hidden)
        output_f = torch.cat(output_f)

        # backward
        hidden = hid_init
        for i in range(t):
            hidden = self.CRNN_model(input[t - i - 1], hidden_iteration[t - i -1], hidden)
            output_b.append(hidden)
        output_b = torch.cat(output_b[::-1])

        output = output_f + output_b

        if b == 1:
            output = output.view(t, 1, self.hidden_size, h, w)

        return output
