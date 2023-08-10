import torch
from torchvision import transforms
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
from torchvision.utils import save_image
from torch.distributions.beta import Beta
from torch.distributions import Normal

def denorm(tensor, device, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    std = torch.Tensor(std).reshape(-1, 1, 1).to(device)
    mean = torch.Tensor(mean).reshape(-1, 1, 1).to(device)
    res = torch.clamp(tensor * std + mean, 0, 1)
    return res

def norm(input, device, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    input = input.permute(3,2,0,1) # bchw
    input = input.float()/255
    std = torch.Tensor(std).reshape(-1, 1, 1).to(device)
    mean = torch.Tensor(mean).reshape(-1, 1, 1).to(device)
    res = (input-mean)/ std
    return res

def save_image_from_tensor_batch(batch, column, path, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], device='cpu'):
    batch = denorm(batch, device, mean, std)
    save_image(batch, path, nrow=column)

def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

def factorization_loss(f_a, f_b):
    # empirical cross-correlation matrix
    f_a_norm = (f_a - f_a.mean(0)) / (f_a.std(0)+1e-6)
    f_b_norm = (f_b - f_b.mean(0)) / (f_b.std(0)+1e-6)
    c = torch.mm(f_a_norm.T, f_b_norm) / f_a_norm.size(0)

    on_diag = torch.diagonal(c).add_(-1).pow_(2).mean()
    off_diag = off_diagonal(c).pow_(2).mean()
    loss = on_diag + 0.005 * off_diag

    return loss

def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))


def linear_rampup(current, rampup_length):
    """Linear rampup"""
    assert current >= 0 and rampup_length >= 0
    if current >= rampup_length:
        return 1.0
    else:
        return current / rampup_length


def step_rampup(current, rampup_length):
    assert current >= 0 and rampup_length >= 0
    if current >= rampup_length:
        return 1.0
    else:
        return 0.0


def get_current_consistency_weight(epoch, weight, rampup_length, rampup_type='step'):
    if rampup_type == 'step':
        rampup_func = step_rampup
    elif rampup_type == 'linear':
        rampup_func = linear_rampup
    elif rampup_type == 'sigmoid':
        rampup_func = sigmoid_rampup
    else:
        raise ValueError("Rampup schedule not implemented")

    return weight * rampup_func(epoch, rampup_length)


def colorful_spectrum_mix_torch(img1, img2, alpha, ratio,device,lam_dis="uniform"):
    """Input image size: ndarray of [H, W, C]"""
       
    ratio = torch.Tensor([ratio]).to(device)
    # lam = np.random.uniform(0, alpha)
    if lam_dis == "uniform":
        lam = torch.Tensor(1).uniform_(0,alpha).to(device)
    elif lam_dis == "beta":# beta 分布
        lam = Beta(torch.tensor([0.5]), torch.tensor([0.5])).sample().to(device)
    else:# 正太分布
        lam = Normal(torch.Tensor([0.0]), torch.Tensor([ratio])).sample().to(device)
    if lam < 0.0:
        lam = torch.Tensor([0.0]).to(device)
    if lam > 1.0:
        lam = torch.Tensor([1.0]).to(device)
    assert img1.shape == img2.shape
    h, w, c,b = img1.shape
    h_crop = h * torch.sqrt(ratio)
    h_crop = h_crop.int().to(device)
    w_crop = w * torch.sqrt(ratio)
    w_crop = w_crop.int().to(device)
    h_start = h // 2 - h_crop // 2
    h_start = h_start.to(device)
    w_start = w // 2 - w_crop // 2
    w_start = w_start.to(device)
    
    img1_fft = torch.fft.fft2(img1, dim=(0, 1))
    img2_fft = torch.fft.fft2(img2, dim=(0, 1))
    img1_abs, img1_pha = torch.abs(img1_fft), torch.angle(img1_fft)
    img2_abs, img2_pha = torch.abs(img2_fft), torch.angle(img2_fft)

    img1_abs = torch.fft.fftshift(img1_abs, dim=(0, 1))
    img2_abs = torch.fft.fftshift(img2_abs, dim=(0, 1))

    img1_abs_ = torch.clone(img1_abs)
    img2_abs_ = torch.clone(img2_abs)
    img1_abs[h_start:h_start + h_crop, w_start:w_start + w_crop] = lam * img2_abs_[h_start:h_start + h_crop, w_start:w_start + w_crop] + (1 - lam) * img1_abs_[h_start:h_start + h_crop, w_start:w_start + w_crop]
    img2_abs[h_start:h_start + h_crop, w_start:w_start + w_crop] = lam * img1_abs_[h_start:h_start + h_crop, w_start:w_start + w_crop] + (1 - lam) * img2_abs_[h_start:h_start + h_crop,w_start:w_start + w_crop]

    img1_abs = torch.fft.ifftshift(img1_abs, dim=(0, 1))
    img2_abs = torch.fft.ifftshift(img2_abs, dim=(0, 1))

    e=torch.exp(torch.Tensor([1])).to(device)

    img21 = img1_abs * (e ** (1j * img1_pha))
    img12 = img2_abs * (e ** (1j * img2_pha))
    img21 = torch.real(torch.fft.ifft2(img21, dim=(0, 1)))
    img12 = torch.real(torch.fft.ifft2(img12, dim=(0, 1)))
    img21 = torch.clip(img21, 0, 255)
    img12 = torch.clip(img12, 0, 255)
    img21=torch.as_tensor(img21,dtype=torch.uint8)
    img12=torch.as_tensor(img12,dtype=torch.uint8)
    return img21, img12    ###torch.Size([128, 3, 224, 224])

def post_process(input,device):
    '''
    input hwcb--> bhwc

    '''
    
    input = input.permute(3,2,0,1) # bchw
    input = input.float()/255
    
    final_input = []
    for i in range(input.shape[0]):
        final_input.append(transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(input[i]))####normalize[C,H,W]
    res = torch.stack(final_input, dim=0).to(device)####b,c,h,w
    
    return res

def Fourier_Transform(batches,env0,env1,device,alpha=1.0,lam_dis = "uniform"):
    '''
    data are all get form dataloader,the data has not be process by to tensor() and norimalze
    '''
    data1 = batches.get("x")[env0].clone() # ensure the data not change
    data2 = batches.get("x")[env1].clone()
    

    data2to1,_ = colorful_spectrum_mix_torch(data1,data2,alpha,1.0,device,lam_dis)
    # data2to1,_ = colorful_spectrum_mix_phase(data1,data2,1.0,1.0,device)
    data2to1  = data2to1.to(device)
    # print("2-the size of data is: {}".format(data2to1.shape))

    batches["x"].append(data2to1)# env0
    batches["y"].append(batches.get("y")[env0])#label
    # print("len of batch x is: {}".format(len(batches["y"])))

def Batch_Fourier(batches,device,alpha=1.0,lam_dis="uniform",times=1):
    '''
    {"y":[tensor1,tensor2,tensor3],"x":[tensor1,tensor2,tensor3]}
    '''
    # transform data to B* H* W* C and in 0~255
    train_envs = len(batches["y"])
    for i in range(train_envs):
        data = batches["x"][i].permute(2,3,1,0) # hwcb
        data = torch.clip(data*255, 0, 255)
        data = torch.as_tensor(data,dtype=torch.uint8)
        batches["x"][i] = data
        
    for t in range(times):
        envs_id = [i for i in range(train_envs)]
        aug_env = np.random.choice(envs_id,1)[0]# random select one domain as target domain
        envs_id.remove(aug_env)
        aug_env_aug = np.random.choice(envs_id,1)[0]# 
        envs_id = [i for i in range(train_envs)]
        for env0 in envs_id:
            env1 = aug_env_aug if env0==aug_env else aug_env
            Fourier_Transform(batches,env0,env1,device,alpha,lam_dis)
            
    for key, tensorlist in batches.items():
        tensors = []
        if key=="x":
            for tensor in tensorlist:
                # tensors.append(post_process(tensor,device))
                tensors.append(norm(tensor,device))
    
    batches["x"] = tensors
    return batches
              
def colorful_spectrum_mix_phase(img1, img2, alpha, ratio,device):
    """Input image size: ndarray of [H, W, C]"""
    ratio = torch.Tensor([0.3]).to(device)
    lam = torch.Tensor(1).uniform_(0,alpha).to(device)
    # # beta 
    # lam = Beta(torch.tensor([0.5]), torch.tensor([0.5])).sample().to(device)
    # # gaussian
    # lam = Normal(torch.Tensor([0.0]), torch.Tensor([1.0])).sample().to(device)
    # if lam < 0.0:
    #     lam = torch.Tensor([0.0]).to(device)
    # if lam > 1.0:
    #     lam = torch.Tensor([1.0]).to(device)
    assert img1.shape == img2.shape
    h, w, c,b = img1.shape
    h_crop = (h * torch.sqrt(ratio)).int().to(device)
    w_crop = (w * torch.sqrt(ratio)).int().to(device)
    h_start = (h // 2 - h_crop // 2).to(device)
    w_start = (w // 2 - w_crop // 2).to(device)

    img1_fft = torch.fft.fft2(img1, dim=(0, 1))
    img2_fft = torch.fft.fft2(img2, dim=(0, 1))
    img1_abs, img1_pha = torch.abs(img1_fft), torch.angle(img1_fft)
    img2_abs, img2_pha = torch.abs(img2_fft), torch.angle(img2_fft)

    img1_abs = torch.fft.fftshift(img1_abs, dim=(0, 1))
    img2_abs = torch.fft.fftshift(img2_abs, dim=(0, 1))

    img1_abs[h_start:h_start + h_crop, w_start:w_start + w_crop] = 10000
    img2_abs[h_start:h_start + h_crop, w_start:w_start + w_crop] = 10000

    img1_abs = torch.fft.ifftshift(img1_abs, dim=(0, 1))
    img2_abs = torch.fft.ifftshift(img2_abs, dim=(0, 1))
    
    e = torch.exp(torch.Tensor([1])).to(device)
 
    
    img1c = img1_abs * (e ** (1j * img1_pha))
    img2c = img2_abs * (e ** (1j * img2_pha))

    img1c = torch.real(torch.fft.ifft2(img1c, dim=(0, 1)))
    img2c = torch.real(torch.fft.ifft2(img2c, dim=(0, 1)))
    img1c = torch.clip(img1c, 0, 255)
    img2c = torch.clip(img2c, 0, 255)
    img1c=torch.as_tensor(img1c,dtype=torch.uint8)
    img2c=torch.as_tensor(img2c,dtype=torch.uint8)

    b = torch.where(img1c>50)
    img1c[b] = 221
    b = torch.where(img1c<=50)
    img1c[b] = 255
    b = torch.where(img1c==221)
    img1c[b] = 0
    b = torch.where(img2c>50)
    img2c[b] = 255
    
    return img1c,img2c

