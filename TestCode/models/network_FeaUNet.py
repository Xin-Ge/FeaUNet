import torch
import torch.nn as nn
import models.basicblock as B
import models.extendedblock as E
import numpy as np
from torch import Tensor

    
class UNet(nn.Module):
    def __init__(self, nc=[64, 128, 256, 512], nb=4, act_mode='R', downsample_mode='strideconv', upsample_mode='convtranspose', bias=True):
        super(UNet, self).__init__()

        # downsample
        if downsample_mode == 'avgpool':
            downsample_block = B.downsample_avgpool
        elif downsample_mode == 'maxpool':
            downsample_block = B.downsample_maxpool
        elif downsample_mode == 'strideconv':
            downsample_block = B.downsample_strideconv
        else:
            raise NotImplementedError('downsample mode [{:s}] is not found'.format(downsample_mode))

        # upsample
        if upsample_mode == 'upconv':
            upsample_block = B.upsample_upconv
        elif upsample_mode == 'pixelshuffle':
            upsample_block = B.upsample_pixelshuffle
        elif upsample_mode == 'convtranspose':
            upsample_block = B.upsample_convtranspose
        else:
            raise NotImplementedError('upsample mode [{:s}] is not found'.format(upsample_mode))

        self.m_body1 = B.sequential(*[B.ResBlock(nc[0], nc[0], bias=bias, mode='C'+act_mode+'C') for _ in range(nb)])
        self.m_down1 = downsample_block(nc[0], nc[1], bias=False, mode='2')
        self.m_body2 = B.sequential(*[B.ResBlock(nc[1], nc[1], bias=bias, mode='C'+act_mode+'C') for _ in range(nb)])
        self.m_down2 = downsample_block(nc[1], nc[2], bias=False, mode='2')
        self.m_body3 = B.sequential(*[B.ResBlock(nc[2], nc[2], bias=bias, mode='C'+act_mode+'C') for _ in range(nb)])
        self.m_down3 = downsample_block(nc[2], nc[3], bias=False, mode='2')
        self.m_body4 = B.sequential(*[B.ResBlock(nc[3], nc[3], bias=bias, mode='C'+act_mode+'C') for _ in range(nb)])
        self.m_up3   = upsample_block(nc[3], nc[2], bias=False, mode='2')
        self.m_body5 = B.sequential(*[B.ResBlock(nc[2], nc[2], bias=bias, mode='C'+act_mode+'C') for _ in range(nb)])
        self.m_up2   = upsample_block(nc[2], nc[1], bias=False, mode='2')
        self.m_body6 = B.sequential(*[B.ResBlock(nc[1], nc[1], bias=bias, mode='C'+act_mode+'C') for _ in range(nb)])
        self.m_up1   = upsample_block(nc[1], nc[0], bias=False, mode='2')
        self.m_body7 = B.sequential(*[B.ResBlock(nc[0], nc[0], bias=bias, mode='C'+act_mode+'C') for _ in range(nb)])

    def forward(self, x0):
        x1 = self.m_body1(x0)

        x2 = self.m_down1(x1)
        x2 = self.m_body2(x2)

        x3 = self.m_down2(x2)
        x3 = self.m_body3(x3)

        x = self.m_down3(x3)
        x = self.m_body4(x)
        x = self.m_up3(x)

        x = self.m_body5(x+x3)
        x = self.m_up2(x)

        x = self.m_body6(x+x2)
        x = self.m_up1(x)

        x = self.m_body7(x+x1)
        return x

class FeaUNet(nn.Module):
    def __init__(self, niter=4, np=8, in_nc=64, out_nc=64, nc=[64, 128, 256, 512], nb=1, act_mode='R', downsample_mode='strideconv', upsample_mode='convtranspose', bias=True):
        super(FeaUNet, self).__init__()

        self.n_iter        = niter
        self.patchlen      = out_nc

        self.Im2features   = E.Im2features()

        self.im2patches    = E.Im2patches(patch_size=np, stride=1, dilate=1)

        self.m_heads: nn.ModuleList = nn.ModuleList()
        for _ in range(self.n_iter):
            self.m_heads       = self.m_heads.append(B.conv(4*in_nc+1, nc[0], kernel_size=1, padding=0, bias=False, mode='C'))
        self.UNet          = UNet(nc=nc, nb=nb, act_mode=act_mode, downsample_mode=downsample_mode, upsample_mode=upsample_mode, bias=bias)
        self.m_tails: nn.ModuleList = nn.ModuleList()
        for _ in range(self.n_iter):
            self.m_tails       = self.m_tails.append(B.conv(nc[0], 4*out_nc, kernel_size=1, padding=0, bias=False, mode='C'))

        self.patches2im    = E.Patches2im(patch_size=np, stride=1, dilate=1)

        self.Features2im   = E.Features2im()

    def forward(self, x0:Tensor, sigma:Tensor):
        fy1, fy2, fy3, fy4 = self.Im2features(x0)
        px1 = self.im2patches(fy1)
        px2 = self.im2patches(fy2)
        px3 = self.im2patches(fy3)
        px4 = self.im2patches(fy4)
        s   = sigma.repeat(1, 1, px1.size()[-2], px1.size()[-1])

        for i in range(self.n_iter):
            pm1 = px1 - torch.mean(px1, dim=1, keepdim=True) 
        
            pn  = torch.cat((pm1, px2, px3, px4, s), 1)

            pd  = self.m_heads[i](pn)
            pd  = self.UNet(pd)
            pd  = self.m_tails[i](pd)
            
            px1 = px1 + pd[:, 0*self.patchlen:1*self.patchlen, :, :]
            px2 = px2 + pd[:, 1*self.patchlen:2*self.patchlen, :, :]
            px3 = px3 + pd[:, 2*self.patchlen:3*self.patchlen, :, :]
            px4 = px4 + pd[:, 3*self.patchlen:4*self.patchlen, :, :]

        fx1 = self.patches2im(px1)
        fx2 = self.patches2im(px2)
        fx3 = self.patches2im(px3)
        fx4 = self.patches2im(px4)
                
        predx  = self.Features2im(fx1, fx2, fx3, fx4)
        return predx

class FeaUNet_woCompleteConv(nn.Module):
    def __init__(self, niter=4, np=8, in_nc=64, out_nc=64, nc=[64, 128, 256, 512], nb=1, act_mode='R', downsample_mode='strideconv', upsample_mode='convtranspose', bias=True):
        super(FeaUNet_woCompleteConv, self).__init__()

        self.n_iter        = niter

        self.im2patches    = E.Im2patches(patch_size=np, stride=1, dilate=1)
        self.GetPatchDCs   = E.GetPatchDCs(np)
        self.MinusPatchDCs = E.MinusPatchDCs()

        self.m_heads: nn.ModuleList = nn.ModuleList()
        for _ in range(self.n_iter):
            self.m_heads       = self.m_heads.append(B.conv(in_nc+1, nc[0], kernel_size=1, padding=0, bias=False, mode='C'))
        self.UNet          = UNet(nc=nc, nb=nb, act_mode=act_mode, downsample_mode=downsample_mode, upsample_mode=upsample_mode, bias=bias)
        self.m_tail        = B.conv(nc[0], out_nc, kernel_size=1, padding=0, bias=False, mode='C')

        self.patches2im    = E.Patches2im(patch_size=np, stride=1, dilate=1)

    def forward(self, y:Tensor, sigma:Tensor):
        px  = self.im2patches(y)
        s   = sigma.repeat(1, 1, px.size()[-2], px.size()[-1])

        for i in range(self.n_iter):
            pm  = px - torch.mean(px, dim=1, keepdim=True)      
        
            pn  = torch.cat((pm, s), 1)

            pd  = self.m_heads[i](pn)
            pd  = self.UNet(pd)
            pd  = self.m_tail(pd)
            
            px  = px + pd

        predx = self.patches2im(px)
                
        return predx
    
class FeaUNet_woPN(nn.Module):
    def __init__(self, niter=4, np=8, in_nc=64, out_nc=64, nc=[64, 128, 256, 512], nb=1, act_mode='R', downsample_mode='strideconv', upsample_mode='convtranspose', bias=True):
        super(FeaUNet_woPN, self).__init__()

        self.n_iter        = niter
        self.patchlen      = out_nc

        self.Im2features   = E.Im2features()

        self.im2patches    = E.Im2patches(patch_size=np, stride=1, dilate=1)

        self.m_heads: nn.ModuleList = nn.ModuleList()
        for _ in range(self.n_iter):
            self.m_heads       = self.m_heads.append(B.conv(4*in_nc+1, nc[0], kernel_size=1, padding=0, bias=False, mode='C'))
        self.UNet          = UNet(nc=nc, nb=nb, act_mode=act_mode, downsample_mode=downsample_mode, upsample_mode=upsample_mode, bias=bias)
        self.m_tails: nn.ModuleList = nn.ModuleList()
        for _ in range(self.n_iter):
            self.m_tails       = self.m_tails.append(B.conv(nc[0], 4*out_nc, kernel_size=1, padding=0, bias=False, mode='C'))

        self.patches2im    = E.Patches2im(patch_size=np, stride=1, dilate=1)

        self.Features2im   = E.Features2im()

    def forward(self, x0:Tensor, sigma:Tensor):
        fy1, fy2, fy3, fy4 = self.Im2features(x0)
        px1 = self.im2patches(fy1)
        px2 = self.im2patches(fy2)
        px3 = self.im2patches(fy3)
        px4 = self.im2patches(fy4)
        s   = sigma.repeat(1, 1, px1.size()[-2], px1.size()[-1])

        for i in range(self.n_iter):
            pn  = torch.cat((px1, px2, px3, px4, s), 1)

            pd  = self.m_heads[i](pn)
            pd  = self.UNet(pd)
            pd  = self.m_tails[i](pd)
            
            px1 = px1 + pd[:, 0*self.patchlen:1*self.patchlen, :, :]
            px2 = px2 + pd[:, 1*self.patchlen:2*self.patchlen, :, :]
            px3 = px3 + pd[:, 2*self.patchlen:3*self.patchlen, :, :]
            px4 = px4 + pd[:, 3*self.patchlen:4*self.patchlen, :, :]

        fx1 = self.patches2im(px1)
        fx2 = self.patches2im(px2)
        fx3 = self.patches2im(px3)
        fx4 = self.patches2im(px4)
                
        predx  = self.Features2im(fx1, fx2, fx3, fx4)
        return predx    

class FeaUNet_fullPN(nn.Module):
    def __init__(self, niter=4, np=8, in_nc=64, out_nc=64, nc=[64, 128, 256, 512], nb=1, act_mode='R', downsample_mode='strideconv', upsample_mode='convtranspose', bias=True):
        super(FeaUNet_fullPN, self).__init__()

        self.n_iter        = niter
        self.patchlen      = out_nc

        self.Im2features   = E.Im2features()

        self.im2patches    = E.Im2patches(patch_size=np, stride=1, dilate=1)

        self.m_heads: nn.ModuleList = nn.ModuleList()
        for _ in range(self.n_iter):
            self.m_heads       = self.m_heads.append(B.conv(4*in_nc+1, nc[0], kernel_size=1, padding=0, bias=False, mode='C'))
        self.UNet          = UNet(nc=nc, nb=nb, act_mode=act_mode, downsample_mode=downsample_mode, upsample_mode=upsample_mode, bias=bias)
        self.m_tails: nn.ModuleList = nn.ModuleList()
        for _ in range(self.n_iter):
            self.m_tails       = self.m_tails.append(B.conv(nc[0], 4*out_nc, kernel_size=1, padding=0, bias=False, mode='C'))

        self.patches2im    = E.Patches2im(patch_size=np, stride=1, dilate=1)

        self.Features2im   = E.Features2im()

    def forward(self, x0:Tensor, sigma:Tensor):
        fy1, fy2, fy3, fy4 = self.Im2features(x0)
        px1 = self.im2patches(fy1)
        px2 = self.im2patches(fy2)
        px3 = self.im2patches(fy3)
        px4 = self.im2patches(fy4)
        s   = sigma.repeat(1, 1, px1.size()[-2], px1.size()[-1])

        for i in range(self.n_iter):
            px  = torch.cat((px1, px2, px3, px4), 1)
            pm  = px - torch.mean(px, dim=1, keepdim=True) 
        
            pn  = torch.cat((pm, s), 1)

            pd  = self.m_heads[i](pn)
            pd  = self.UNet(pd)
            pd  = self.m_tails[i](pd)
            
            px1 = px1 + pd[:, 0*self.patchlen:1*self.patchlen, :, :]
            px2 = px2 + pd[:, 1*self.patchlen:2*self.patchlen, :, :]
            px3 = px3 + pd[:, 2*self.patchlen:3*self.patchlen, :, :]
            px4 = px4 + pd[:, 3*self.patchlen:4*self.patchlen, :, :]

        fx1 = self.patches2im(px1)
        fx2 = self.patches2im(px2)
        fx3 = self.patches2im(px3)
        fx4 = self.patches2im(px4)
                
        predx  = self.Features2im(fx1, fx2, fx3, fx4)
        return predx