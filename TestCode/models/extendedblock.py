import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import Tensor

# --------------------------------------------
# im2patches
# --------------------------------------------
class Im2patches(nn.Module):
    def __init__(self, patch_size:int, stride=1, dilate=1):
        super(Im2patches, self).__init__()
        self.patch_size = patch_size
        self.stride     = stride
        self.dilate     = dilate
    
    def _im2patches_forward_(self, input:Tensor, patch_size:int, stride:int, dilate:int):
        batch_size, channels, in_height, in_width = input.size()
        unfold = torch.nn.Unfold(kernel_size=(patch_size, patch_size), dilation=dilate, padding=0, stride=stride)
        unfolded_input = unfold(input)
        output = unfolded_input.view(batch_size, channels*patch_size**2, 
                                 (in_height-(patch_size-1)*dilate-1)//stride + 1,
                                 (in_width -(patch_size-1)*dilate-1)//stride + 1)
        return output

    def forward(self, input):
        return self._im2patches_forward_(input, self.patch_size, self.stride, self.dilate)

    def extra_repr(self):
        return 'patch_size={}, stride={}, dilate={}'.format(self.patch_size, self.stride, self.dilate)

# --------------------------------------------
# Patches2im
# --------------------------------------------

class Patches2im(nn.Module):
    def __init__(self, patch_size:int, stride=1, dilate=1):
        super(Patches2im, self).__init__()
        self.patch_size = patch_size
        self.stride     = stride
        self.dilate     = dilate

    def _patches2im_forward_(self, input:Tensor, patch_size:int, stride:int, dilate:int):
        batch_size, channels, in_height, in_width = input.size()

        out_height  = (patch_size-1)*dilate + (in_height-1)*stride + 1
        out_widthut = (patch_size-1)*dilate + (in_width-1)*stride  + 1

        input  = input.view(batch_size, channels, in_height*in_width)
        snum   = torch.ones((batch_size, patch_size**2, in_height*in_width), dtype=input.dtype, device=input.device)
        #snum   = torch.ones_like(input)
        fold   = torch.nn.Fold(output_size=(out_height, out_widthut), kernel_size=(patch_size, patch_size), dilation=dilate, padding=0, stride=stride)
        output = fold(input)
        counts = fold(snum)
        output /= counts
        return output

    def forward(self, input):
        return self._patches2im_forward_(input, self.patch_size, self.stride, self.dilate)

    def extra_repr(self):
        return 'patch_size={}, stride={}, dilate={}'.format(self.patch_size, self.stride, self.dilate)

# --------------------------------------------
# GetPatchDCs
# --------------------------------------------
class GetPatchDCs(nn.Module):
    def __init__(self, patch_size:int):
        super(GetPatchDCs, self).__init__()
        self.patch_size = patch_size

    def _getpatchdcs_forward(self, input:Tensor, patch_size:int):
        batch_size, channels, in_height, in_width = input.size()

        input  = input.view(batch_size, channels//(patch_size**2), patch_size**2, in_height, in_width)
        output = torch.mean(input, dim=2, keepdim=True)
        output = output.view(batch_size, channels//(patch_size**2), in_height, in_width)
        return output

    def forward(self, input):
        return self._getpatchdcs_forward(input, self.patch_size)

    def extra_repr(self):
        return 'patch_size={}'.format(self.patch_size)

# --------------------------------------------
# MinusPatchDCs
# --------------------------------------------
class MinusPatchDCs(nn.Module):
    def __init__(self):
        super(MinusPatchDCs, self).__init__()

    def _minuspatchdcs_forward(self, patches:Tensor, patchDCs:Tensor):
        P_batch_size, P_channels, P_height, P_width     = patches.size()
        DC_batch_size, DC_channels, DC_height, DC_width = patchDCs.size()

        patches  = patches.view(P_batch_size, DC_channels, P_channels//DC_channels, P_height, P_width)
        patchDCs = patchDCs.view(DC_batch_size, DC_channels, 1, DC_height, DC_width)
        output   = patches - patchDCs
        output   = output.view(P_batch_size, P_channels, P_height, P_width)
        return output

    def forward(self, patches, patchDCs):
        return self._minuspatchdcs_forward(patches, patchDCs)
    
# --------------------------------------------
# PlusPatchDCs
# --------------------------------------------
class PlusPatchDCs(nn.Module):
    def __init__(self):
        super(PlusPatchDCs, self).__init__()

    def _pluspatchdcs_forward(self, patches:Tensor, patchDCs:Tensor):
        P_batch_size, P_channels, P_height, P_width     = patches.size()
        DC_batch_size, DC_channels, DC_height, DC_width = patchDCs.size()

        patches  = patches.view(P_batch_size, DC_channels, P_channels//DC_channels, P_height, P_width)
        patchDCs = patchDCs.view(DC_batch_size, DC_channels, 1, DC_height, DC_width)
        output   = patches + patchDCs
        output   = output.view(P_batch_size, P_channels, P_height, P_width)
        return output

    def forward(self, patches, patchDCs):
        return self._pluspatchdcs_forward(patches, patchDCs)
    
# --------------------------------------------
# Im2features
# --------------------------------------------
class Im2features(nn.Module):
    def __init__(self):
        super(Im2features, self).__init__()
    
    def _im2features_forward_(self, input:Tensor):
        patch_size = 2
        GKernel = torch.tensor([[0.25, 0.25, 0.25, 0.25], [-0.5, 0.5, -0.5, 0.5], [-0.5, -0.5, 0.5, 0.5], [-0.5, 0.5, 0.5, -0.5]]).to(input.device)

        batch_size, channels, in_height, in_width = input.size()
        unfold  = torch.nn.Unfold(kernel_size=(patch_size, patch_size), dilation=1, padding=0, stride=1)
        unfolded_input = unfold(input)
        Gwinds  = unfolded_input.view(batch_size, channels, patch_size**2, (in_height-1)*(in_width-1))
        feature = torch.matmul(GKernel, Gwinds)

        output1 = feature[:,:,0,:].view(batch_size, channels, in_height-1, in_width-1)
        output2 = feature[:,:,1,:].view(batch_size, channels, in_height-1, in_width-1)
        output3 = feature[:,:,2,:].view(batch_size, channels, in_height-1, in_width-1)
        output4 = feature[:,:,3,:].view(batch_size, channels, in_height-1, in_width-1)
        return output1, output2, output3, output4

    def forward(self, input):
        output1, output2, output3, output4 = self._im2features_forward_(input)
        return output1, output2, output3, output4
    
# --------------------------------------------
# Im2features
# --------------------------------------------
class Features2im(nn.Module):
    def __init__(self):
        super(Features2im, self).__init__()
    
    def _features2im_forward_(self, input1:Tensor, input2:Tensor, input3:Tensor, input4:Tensor):
        batch_size, channels, in_height, in_width = input1.size()
        input1 = input1.view(batch_size, channels, 1, in_height*in_width)
        input2 = input2.view(batch_size, channels, 1, in_height*in_width)
        input3 = input3.view(batch_size, channels, 1, in_height*in_width)
        input4 = input4.view(batch_size, channels, 1, in_height*in_width)

        feature = torch.cat((input1, input2, input3, input4), dim=2)

        patch_size = 2
        GKernel = torch.tensor([[0.25, 0.25, 0.25, 0.25], [-0.5, 0.5, -0.5, 0.5], [-0.5, -0.5, 0.5, 0.5], [-0.5, 0.5, 0.5, -0.5]]).to(feature.device)

        GK_inv = torch.inverse(GKernel)
        Gwinds = torch.matmul(GK_inv, feature)

        Gwinds = Gwinds.view(batch_size, channels*(patch_size**2), in_height*in_width)
        snum   = torch.ones((batch_size, patch_size**2, in_height*in_width), dtype=Gwinds.dtype, device=Gwinds.device)
        #snum   = torch.ones_like(Gwinds)
        fold   = torch.nn.Fold(output_size=(in_height+1, in_width+1), kernel_size=(patch_size, patch_size), dilation=1, padding=0, stride=1)
        output = fold(Gwinds)
        counts = fold(snum)
        output /= counts
        return output

    def forward(self, input1, input2, input3, input4):
        output = self._features2im_forward_(input1, input2, input3, input4)
        return output
