import os.path
import logging
import time

import numpy as np
from collections import OrderedDict

import torch.nn as nn
import torch

from utils import utils_logger
from utils import utils_model
from utils import utils_image as util


"""
Spyder (Python 3.7)
PyTorch 1.6.0
Windows 10 or Linux
Kai Zhang (cskaizhang@gmail.com)
github: https://github.com/cszn/DPIR
        https://github.com/cszn/IRCNN
        https://github.com/cszn/KAIR
@article{zhang2020plug,
  title={Plug-and-Play Image Restoration with Deep Denoiser Prior},
  author={Zhang, Kai and Li, Yawei and Zuo, Wangmeng and Zhang, Lei and Van Gool, Luc and Timofte, Radu},
  journal={arXiv preprint},
  year={2020}
}
% If you have any question, please feel free to contact with me.
% Kai Zhang (e-mail: cskaizhang@gmail.com; homepage: https://cszn.github.io/)
by Kai Zhang (01/August/2020)

# --------------------------------------------
|--model_zoo               # model_zoo
   |--drunet_gray          # model_name, for color images
   |--drunet_color
|--testset                 # testsets
   |--set12                # testset_name
   |--bsd68
   |--cbsd68
|--results                 # results
   |--set12_dn_drunet_gray # result_name = testset_name + '_' + 'dn' + model_name
   |--set12_dn_drunet_color
# --------------------------------------------
"""


def main():

    # ----------------------------------------
    # Preparation
    # ----------------------------------------
    noise_level_img = 25                 # set AWGN noise level for noisy image
    noise_level_model = noise_level_img  # set noise level for model
    model_name = 'FeaUNet_3_4'        # set denoiser model, 'drunet_gray' | 'drunet_color'
    testset_name = 'Bsd68'            # set test set,  'Set12' | 'Bsd68' | 'Kodak24' | 'McMaster' | 'Urban100'
    show_img = False                     # default: False
    border = 0                           # shave boader to calculate PSNR and SSIM
    patch_size = 3
    n_iter = 4
    refield_extension = patch_size
    task_current = 'dn_color'             # 'dn_gray' for grayscale denoising, 'dn_color' for color denoising 

    model_pool = 'model_zoo'             # fixed
    testsets = 'testsets'                # fixed
    results  = 'results'                 # fixed
    if 'color' in task_current:
        n_channels = 3                   # 3 for color image
    else:
        n_channels = 1                   # 1 for grayscale image

    result_name  = testset_name + '_' + task_current + '_' + model_name + '_{}'.format(noise_level_img)

    if 'color' in task_current:
        model_path = os.path.join(model_pool, 'FeaUNet', 'FeaUNet_3_4_color.pth')
    else:
        model_path = os.path.join(model_pool, 'FeaUNet', 'FeaUNet_3_4_gray.pth')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.empty_cache()

    # ----------------------------------------
    # L_path, E_path, H_path
    # ----------------------------------------

    L_path = os.path.join(testsets, 'original', testset_name) # L_path, for Low-quality images
    E_path = os.path.join(results, task_current, model_name, 'sigma_{}'.format(noise_level_img), testset_name)   # E_path, for Estimated images
    util.mkdir(E_path)

    G_copy_path = os.path.join(testsets, 'gray_copy', testset_name) 
    util.mkdir(G_copy_path)
    if n_channels == 3:
        N_copy_path = os.path.join(testsets, 'noisy_copy', testset_name + '_color', 'sigma_{}'.format(noise_level_img)) 
    else:
        N_copy_path = os.path.join(testsets, 'noisy_copy', testset_name + '_gray', 'sigma_{}'.format(noise_level_img)) 
    util.mkdir(N_copy_path)

    logger_name = result_name
    utils_logger.logger_info(logger_name, log_path=os.path.join(E_path, logger_name+'.log'))
    logger = logging.getLogger(logger_name)

    # ----------------------------------------
    # load model
    # ----------------------------------------

    from models.network_FeaUNet import FeaUNet as net
    model = net(niter=n_iter, np=patch_size, in_nc=n_channels*(patch_size**2), out_nc=n_channels*(patch_size**2), nc=[64, 128, 256, 512], nb=4, act_mode='R', downsample_mode="strideconv", upsample_mode="convtranspose", bias=True)
    model.load_state_dict(torch.load(model_path), strict=True)
    model = model.to(device)
    model = nn.DataParallel(model)
    model.eval()
    for k, v in model.named_parameters():
        v.requires_grad = False

    logger.info('Model path: {:s}'.format(model_path))
    number_parameters = sum(map(lambda x: x.numel(), model.parameters()))
    logger.info('Params number: {}'.format(number_parameters))

    test_results = OrderedDict()
    test_results['psnr'] = []
    test_results['ssim'] = []
    test_results['retm'] = []

    logger.info('model_name:{}, model sigma:{}, image sigma:{}'.format(model_name, noise_level_img, noise_level_model))
    logger.info(L_path)
    L_paths = util.get_image_paths(L_path)

    with torch.no_grad():
        for idx, img in enumerate(L_paths):

            # ------------------------------------
            # (1) img_L
            # ------------------------------------

            img_name, ext = os.path.splitext(os.path.basename(img))
            # logger.info('{:->4d}--> {:>10s}'.format(idx+1, img_name+ext))
            img_H = util.imread_uint(img, n_channels=n_channels)
            if n_channels == 1 and not os.path.exists(os.path.join(G_copy_path, img_name+ext)):
                util.imsave(img_H, os.path.join(G_copy_path, img_name+ext))
            img_L = util.uint2single(img_H)
            print(img_L.shape, img_L.max())

            # Add noise without clipping
            np.random.seed(seed=0)  # for reproducibility
            img_L += np.random.normal(0, noise_level_img/255., img_L.shape)
            if not os.path.exists(os.path.join(N_copy_path, img_name+ext)):
                img_L_save = util.single2uint(img_L)
                util.imsave(img_L_save, os.path.join(N_copy_path, img_name+ext))

            util.imshow(util.single2uint(img_L), title='Noisy image with noise level {}'.format(noise_level_img)) if show_img else None

            img_L = util.single2tensor4(img_L)
            print(img_L.shape, img_L.max())
            
            #img_L = torch.cat((img_L, torch.FloatTensor([noise_level_model/255.]).repeat(1, 1, img_L.shape[2], img_L.shape[3])), dim=1)
            img_L = img_L.to(device)
            img_S = torch.FloatTensor([noise_level_model/255.]).to(device)

            # ------------------------------------
            # (2) img_E
            # ------------------------------------
            if idx == 0:
                img_E = utils_model.test_impading_LS(model, img_L, img_S, refield=8, refield_extension=refield_extension)
            # 预处理一次，以避免统计系统对显存分配的时间

            start_time = time.perf_counter()  # 记录开始时间
            img_E = utils_model.test_impading_LS(model, img_L, img_S, refield=8, refield_extension=refield_extension)
            end_time = time.perf_counter()    # 记录结束时间
            retm = end_time - start_time  # 计算运行时间

            img_E = util.tensor2uint(img_E)

            # --------------------------------
            # PSNR and SSIM
            # --------------------------------

            if n_channels == 1:
                img_H = img_H.squeeze() 
            psnr = util.calculate_psnr(img_E, img_H, border=border)
            ssim = util.calculate_ssim(img_E, img_H, border=border)
            test_results['psnr'].append(psnr)
            test_results['ssim'].append(ssim)
            test_results['retm'].append(retm)
            logger.info('{:s} - PSNR: {:.4f} dB; SSIM: {:.6f}; Time: {:.8f}.'.format(img_name+ext, psnr, ssim, retm))

            # ------------------------------------
            # save results
            # ------------------------------------

            util.imsave(img_E, os.path.join(E_path, img_name+ext))

    ave_psnr = sum(test_results['psnr']) / len(test_results['psnr'])
    ave_ssim = sum(test_results['ssim']) / len(test_results['ssim'])
    ave_retm = sum(test_results['retm']) / len(test_results['retm'])
    logger.info('Average PSNR/SSIM(RGB)/Time - {} - PSNR: {:.4f} dB; SSIM: {:.6f}; Time: {:.8f}'.format(result_name, ave_psnr, ave_ssim, ave_retm))

if __name__ == '__main__':

    main()
