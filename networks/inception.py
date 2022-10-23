import torch
from torch import nn
from torch.nn import functional as F
from typing import Tuple

import sys

sys.path.append("/home/students/yiwei/yiwei_gitlab/EffFormer/networks/inception.py")
# # proj_dilation1 = nn.Conv2d(input_dim,in_dim[i_stage],kernel_size = patch_sizes1[i_stage], stride = strides[i_stage],padding=dil_padding_sizes1[i_stage], dilation=2)
# conv3_3_1 =  nn.Conv2d(input_dim, in_dim[i_stage], kernel_size=3, padding =1)
# conv3_3_2 =  nn.Conv2d(input_dim, in_dim[i_stage], kernel_size=3, padding =1)
#https://github.com/Cassieyy/MultiResUnet3D/blob/main/MultiResUnet3D.py

class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size=3, stride=1, padding=1, act='relu'):
        # print(ch_out)
        super(conv_block,self).__init__()
        if act == None:
            self.conv = nn.Sequential(
                nn.Conv2d(ch_in, ch_out, kernel_size=kernel_size,stride=stride,padding=padding),
                nn.BatchNorm2d(ch_out)
            )
        elif act == 'relu':
            self.conv = nn.Sequential(
                nn.Conv2d(ch_in, ch_out, kernel_size=kernel_size,stride=stride,padding=padding),
                nn.BatchNorm2d(ch_out),
                nn.ReLU(inplace=True)
            )
        elif act == 'sigmoid':
            self.conv = nn.Sequential(
                nn.Conv2d(ch_in, ch_out, kernel_size=kernel_size,stride=stride,padding=padding),
                nn.BatchNorm2d(ch_out),
                nn.Sigmoid()
            )

    def forward(self,x):
        x = self.conv(x)
        return x
    


class res_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(res_block,self).__init__()
        self.res = conv_block(ch_in,ch_out,1,1,0,None)
        self.main = conv_block(ch_in,ch_out)
        self.bn = nn.BatchNorm2d(ch_in)
    def forward(self,x):
        res_x = self.res(x)

        main_x = self.main(x)
        out = res_x.add(main_x)
        out = nn.ReLU(inplace=True)(out)
        # print(out.shape[1], type(out.shape[1]))
        # assert 1>3
        out = self.bn(out)
        return out
    


class MultiResBlock_1357(nn.Module):
    def __init__(self,in_ch,U,branch=1,downsample=2, alpha=1):
        super(MultiResBlock_1357,self).__init__()
        self.W = alpha * U
        self.one_ch = conv_block(in_ch, 1)
#         self.residual_layer = conv_block(1, self.W, 1, 1, 0, act=None)
        self.residual_layer = conv_block(1, self.W)
#         self.conv3x3 = conv_block(1, int(self.W*0.167))
#         self.conv5x5 = conv_block(int(self.W*0.167), int(self.W*0.333))
#         self.conv7x7 = conv_block(int(self.W*0.333), self.W-int(self.W*0.167)-int(self.W*0.333))
        self.conv3x3 = conv_block(1, int(self.W))
        self.conv5x5 = conv_block(int(self.W), int(self.W))
        self.conv7x7 = conv_block(int(self.W), self.W)
        self.maxpool = nn.MaxPool2d(downsample, stride=downsample)
        self.relu = nn.ReLU(inplace=True)
#         self.batchnorm_1 = nn.BatchNorm2d(int(self.W*0.167)+int(self.W*0.333)+int(self.W*0.5))
#         self.batchnorm_2 = nn.BatchNorm2d(int(self.W*0.167)+int(self.W*0.333)+int(self.W*0.5))
#         self.batchnorm_1 = nn.BatchNorm2d(self.W)
#         self.batchnorm_2 = nn.BatchNorm2d(self.W)
        self.norm = nn.LayerNorm(self.W)
        
    def forward(self, x):
        out = []
        print(x.shape) 
        print("\n W=alpha*U :{}\n".format(self.W))
        x = self.one_ch(x) 
        res = self.residual_layer(x)
        res_out = self.maxpool(res)
        res_out = (res_out.flatten(2)).transpose(1,2)
        out.append(res_out)
        print("\n res:{}\n".format(res_out.shape))
        
        sbs = self.conv3x3(x)
        sbs_out = self.maxpool(sbs)
        sbs_out = (sbs_out.flatten(2)).transpose(1,2)
        print("\n out_3*3:{}\n".format(sbs_out.shape))
        out.append(sbs_out)
        
        obo = self.conv5x5(sbs)
        obo_out = self.maxpool(obo)
        obo_out = (obo_out.flatten(2)).transpose(1,2)
        out.append(obo_out)
        print("\n out_5*5:{}\n".format(obo_out.shape))
        
        cbc = self.conv7x7(obo)
        cbc_out = self.maxpool(cbc)
        cbc_out = (cbc_out.flatten(2)).transpose(1,2)
        print("\n out_7*7:{}\n".format(cbc_out.shape))
        out.append(cbc_out)
        all_t = torch.cat((out[0], out[1], out[2],out[3]), 1)
        all_t = self.norm(all_t)
        print("\n cat_together:{}\n".format(all_t.shape))
#         all_t_b = self.batchnorm_1(all_t)
#         out = all_t_b.add(res)
#         out = self.relu(out)
#         out = self.batchnorm_2(out)      
        
        return all_t
    
    
class MultiResBlock_135(nn.Module):
    def __init__(self,in_ch,U,branch=1,downsample=2, alpha=1):
        super(MultiResBlock_135,self).__init__()
        self.W = alpha * U
        self.one_ch = conv_block(in_ch, 1)
#         self.residual_layer = conv_block(1, self.W, 1, 1, 0, act=None)
        self.residual_layer = conv_block(1, self.W)
#         self.conv3x3 = conv_block(1, int(self.W*0.167))
#         self.conv5x5 = conv_block(int(self.W*0.167), int(self.W*0.333))
#         self.conv7x7 = conv_block(int(self.W*0.333), self.W-int(self.W*0.167)-int(self.W*0.333))
        self.conv3x3 = conv_block(1, int(self.W))
        self.conv5x5 = conv_block(int(self.W), int(self.W))
        self.conv7x7 = conv_block(int(self.W), self.W)
        self.maxpool = nn.MaxPool2d(downsample, stride=downsample)
        self.relu = nn.ReLU(inplace=True)
#         self.batchnorm_1 = nn.BatchNorm2d(int(self.W*0.167)+int(self.W*0.333)+int(self.W*0.5))
#         self.batchnorm_2 = nn.BatchNorm2d(int(self.W*0.167)+int(self.W*0.333)+int(self.W*0.5))
#         self.batchnorm_1 = nn.BatchNorm2d(self.W)
#         self.batchnorm_2 = nn.BatchNorm2d(self.W)
        self.norm = nn.LayerNorm(self.W)
        
    def forward(self, x):
        out = []
        print(x.shape) 
        print("\n W=alpha*U :{}\n".format(self.W))
        x = self.one_ch(x) 
        res = self.residual_layer(x)
        res_out = self.maxpool(res)
        res_out = (res_out.flatten(2)).transpose(1,2)
        out.append(res_out)
        print("\n res:{}\n".format(res_out.shape))
        
        sbs = self.conv3x3(x)
        sbs_out = self.maxpool(sbs)
        sbs_out = (sbs_out.flatten(2)).transpose(1,2)
        print("\n out_3*3:{}\n".format(sbs_out.shape))
        out.append(sbs_out)
        
        obo = self.conv5x5(sbs)
        obo_out = self.maxpool(obo)
        obo_out = (obo_out.flatten(2)).transpose(1,2)
        out.append(obo_out)
        print("\n out_5*5:{}\n".format(obo_out.shape))
        
#         cbc = self.conv7x7(obo)
#         cbc_out = self.maxpool(cbc)
#         cbc_out = (cbc_out.flatten(2)).transpose(1,2)
#         out.append(cbc_out)
#         print("\n out_7*7:{}\n".format(cbc_out.shape))
        
        
        all_t = torch.cat(out, 1)
        all_t = self.norm(all_t)
        print("\n cat_together:{}\n".format(all_t.shape))
#         all_t_b = self.batchnorm_1(all_t)
#         out = all_t_b.add(res)
#         out = self.relu(out)
#         out = self.batchnorm_2(out)      
        
        return all_t

class MultiResBlock_157(nn.Module):
    def __init__(self,in_ch,U,branch=1,downsample=2, alpha=1):
        super(MultiResBlock_157,self).__init__()
        self.W = alpha * U
        self.one_ch = conv_block(in_ch, 1)
#         self.residual_layer = conv_block(1, self.W, 1, 1, 0, act=None)
        self.residual_layer = conv_block(1, self.W)
#         self.conv3x3 = conv_block(1, int(self.W*0.167))
#         self.conv5x5 = conv_block(int(self.W*0.167), int(self.W*0.333))
#         self.conv7x7 = conv_block(int(self.W*0.333), self.W-int(self.W*0.167)-int(self.W*0.333))
        self.conv3x3 = conv_block(1, int(self.W))
        self.conv5x5 = conv_block(int(self.W), int(self.W))
        self.conv7x7 = conv_block(int(self.W), self.W)
        self.maxpool = nn.MaxPool2d(downsample, stride=downsample)
        self.relu = nn.ReLU(inplace=True)
#         self.batchnorm_1 = nn.BatchNorm2d(int(self.W*0.167)+int(self.W*0.333)+int(self.W*0.5))
#         self.batchnorm_2 = nn.BatchNorm2d(int(self.W*0.167)+int(self.W*0.333)+int(self.W*0.5))
#         self.batchnorm_1 = nn.BatchNorm2d(self.W)
#         self.batchnorm_2 = nn.BatchNorm2d(self.W)
        self.norm = nn.LayerNorm(self.W)
        
    def forward(self, x):
        out = []
        print(x.shape) 
        print("\n W=alpha*U :{}\n".format(self.W))
        x = self.one_ch(x) 
        res = self.residual_layer(x)
        res_out = self.maxpool(res)
        res_out = (res_out.flatten(2)).transpose(1,2)
        out.append(res_out)
        print("\n res:{}\n".format(res_out.shape))
        
        sbs = self.conv3x3(x)
#         sbs_out = self.maxpool(sbs)
#         sbs_out = (sbs_out.flatten(2)).transpose(1,2)
#         print("\n out_3*3:{}\n".format(sbs_out.shape))
#         out.append(sbs_out)
        
        obo = self.conv5x5(sbs)
        obo_out = self.maxpool(obo)
        obo_out = (obo_out.flatten(2)).transpose(1,2)
        out.append(obo_out)
        print("\n out_5*5:{}\n".format(obo_out.shape))
        
        cbc = self.conv7x7(obo)
        cbc_out = self.maxpool(cbc)
        cbc_out = (cbc_out.flatten(2)).transpose(1,2)
        print("\n out_7*7:{}\n".format(cbc_out.shape))
        out.append(cbc_out)
        all_t = torch.cat(out, 1)
        all_t = self.norm(all_t)
        print("\n cat_together:{}\n".format(all_t.shape))
#         all_t_b = self.batchnorm_1(all_t)
#         out = all_t_b.add(res)
#         out = self.relu(out)
#         out = self.batchnorm_2(out)      
        
        return all_t
    

class MultiResBlock_15(nn.Module):
    def __init__(self,in_ch,U,branch=1,downsample=2, alpha=1):
        super(MultiResBlock_15,self).__init__()
        self.W = alpha * U
        self.one_ch = conv_block(in_ch, 1)
#         self.residual_layer = conv_block(1, self.W, 1, 1, 0, act=None)
        self.residual_layer = conv_block(1, self.W)
#         self.conv3x3 = conv_block(1, int(self.W*0.167))
#         self.conv5x5 = conv_block(int(self.W*0.167), int(self.W*0.333))
#         self.conv7x7 = conv_block(int(self.W*0.333), self.W-int(self.W*0.167)-int(self.W*0.333))
        self.conv3x3 = conv_block(1, int(self.W))
        self.conv5x5 = conv_block(int(self.W), int(self.W))
        self.conv7x7 = conv_block(int(self.W), self.W)
        self.maxpool = nn.MaxPool2d(downsample, stride=downsample)
        self.relu = nn.ReLU(inplace=True)
#         self.batchnorm_1 = nn.BatchNorm2d(int(self.W*0.167)+int(self.W*0.333)+int(self.W*0.5))
#         self.batchnorm_2 = nn.BatchNorm2d(int(self.W*0.167)+int(self.W*0.333)+int(self.W*0.5))
#         self.batchnorm_1 = nn.BatchNorm2d(self.W)
#         self.batchnorm_2 = nn.BatchNorm2d(self.W)
        self.norm = nn.LayerNorm(self.W)
        
    def forward(self, x):
        out = []
        print(x.shape) 
        print("\n W=alpha*U :{}\n".format(self.W))
        x = self.one_ch(x) 
        res = self.residual_layer(x)
        res_out = self.maxpool(res)
        res_out = (res_out.flatten(2)).transpose(1,2)
        out.append(res_out)
        print("\n res:{}\n".format(res_out.shape))
        
        sbs = self.conv3x3(x)
#         sbs_out = self.maxpool(sbs)
#         sbs_out = (sbs_out.flatten(2)).transpose(1,2)
#         print("\n out_3*3:{}\n".format(sbs_out.shape))
#         out.append(sbs_out)
        
        obo = self.conv5x5(sbs)
        obo_out = self.maxpool(obo)
        obo_out = (obo_out.flatten(2)).transpose(1,2)
        out.append(obo_out)
        print("\n out_5*5:{}\n".format(obo_out.shape))
        
#         cbc = self.conv7x7(obo)
#         cbc_out = self.maxpool(cbc)
#         cbc_out = (cbc_out.flatten(2)).transpose(1,2)
#         print("\n out_7*7:{}\n".format(cbc_out.shape))
#         out.append(cbc_out)
        all_t = torch.cat(out, 1)
        all_t = self.norm(all_t)
        print("\n cat_together:{}\n".format(all_t.shape))
#         all_t_b = self.batchnorm_1(all_t)
#         out = all_t_b.add(res)
#         out = self.relu(out)
#         out = self.batchnorm_2(out)      
        
        return all_t

    
class MultiResBlock_13(nn.Module):
    def __init__(self,in_ch,U,branch=1,downsample=2, alpha=1):
        super(MultiResBlock_13,self).__init__()
        self.W = alpha * U
        self.one_ch = conv_block(in_ch, 1)
#         self.residual_layer = conv_block(1, self.W, 1, 1, 0, act=None)
        self.residual_layer = conv_block(1, self.W)
#         self.conv3x3 = conv_block(1, int(self.W*0.167))
#         self.conv5x5 = conv_block(int(self.W*0.167), int(self.W*0.333))
#         self.conv7x7 = conv_block(int(self.W*0.333), self.W-int(self.W*0.167)-int(self.W*0.333))
        self.conv3x3 = conv_block(1, int(self.W))
        self.conv5x5 = conv_block(int(self.W), int(self.W))
        self.conv7x7 = conv_block(int(self.W), self.W)
        self.maxpool = nn.MaxPool2d(downsample, stride=downsample)
        self.relu = nn.ReLU(inplace=True)
#         self.batchnorm_1 = nn.BatchNorm2d(int(self.W*0.167)+int(self.W*0.333)+int(self.W*0.5))
#         self.batchnorm_2 = nn.BatchNorm2d(int(self.W*0.167)+int(self.W*0.333)+int(self.W*0.5))
#         self.batchnorm_1 = nn.BatchNorm2d(self.W)
#         self.batchnorm_2 = nn.BatchNorm2d(self.W)
        self.norm = nn.LayerNorm(self.W)
        
    def forward(self, x):
        out = []
        print(x.shape) 
        print("\n W=alpha*U :{}\n".format(self.W))
        x = self.one_ch(x) 
        res = self.residual_layer(x)
        res_out = self.maxpool(res)
        res_out = (res_out.flatten(2)).transpose(1,2)
        out.append(res_out)
        print("\n res:{}\n".format(res_out.shape))
        
        sbs = self.conv3x3(x)
        sbs_out = self.maxpool(sbs)
        sbs_out = (sbs_out.flatten(2)).transpose(1,2)
        print("\n out_3*3:{}\n".format(sbs_out.shape))
        out.append(sbs_out)
        
#         obo = self.conv5x5(sbs)
#         obo_out = self.maxpool(obo)
#         obo_out = (obo_out.flatten(2)).transpose(1,2)
#         out.append(obo_out)
#         print("\n out_5*5:{}\n".format(obo_out.shape))
        
#         cbc = self.conv7x7(obo)
#         cbc_out = self.maxpool(cbc)
#         cbc_out = (cbc_out.flatten(2)).transpose(1,2)
#         print("\n out_7*7:{}\n".format(cbc_out.shape))
#         out.append(cbc_out)
        all_t = torch.cat(out, 1)
        all_t = self.norm(all_t)
        print("\n cat_together:{}\n".format(all_t.shape))
#         all_t_b = self.batchnorm_1(all_t)
#         out = all_t_b.add(res)
#         out = self.relu(out)
#         out = self.batchnorm_2(out)      
        
        return all_t
    
    
    
class MultiResBlock_1(nn.Module):
    def __init__(self,in_ch,U,branch=1,downsample=2, alpha=1):
        super(MultiResBlock_1,self).__init__()
        self.W = alpha * U
        self.one_ch = conv_block(in_ch, 1)
#         self.residual_layer = conv_block(1, self.W, 1, 1, 0, act=None)
        self.residual_layer = conv_block(1, self.W)
#         self.conv3x3 = conv_block(1, int(self.W*0.167))
#         self.conv5x5 = conv_block(int(self.W*0.167), int(self.W*0.333))
#         self.conv7x7 = conv_block(int(self.W*0.333), self.W-int(self.W*0.167)-int(self.W*0.333))
        self.conv3x3 = conv_block(1, int(self.W))
        self.conv5x5 = conv_block(int(self.W), int(self.W))
        self.conv7x7 = conv_block(int(self.W), self.W)
        self.maxpool = nn.MaxPool2d(downsample, stride=downsample)
        self.relu = nn.ReLU(inplace=True)
#         self.batchnorm_1 = nn.BatchNorm2d(int(self.W*0.167)+int(self.W*0.333)+int(self.W*0.5))
#         self.batchnorm_2 = nn.BatchNorm2d(int(self.W*0.167)+int(self.W*0.333)+int(self.W*0.5))
#         self.batchnorm_1 = nn.BatchNorm2d(self.W)
#         self.batchnorm_2 = nn.BatchNorm2d(self.W)
        self.norm = nn.LayerNorm(self.W)
        
    def forward(self, x):
        out = []
        print(x.shape) 
        print("\n W=alpha*U :{}\n".format(self.W))
        x = self.one_ch(x) 
        res = self.residual_layer(x)
        res_out = self.maxpool(res)
        res_out = (res_out.flatten(2)).transpose(1,2)
        out.append(res_out)
        print("\n res:{}\n".format(res_out.shape))
        
#         sbs = self.conv3x3(x)
#         sbs_out = self.maxpool(sbs)
#         sbs_out = (sbs_out.flatten(2)).transpose(1,2)
#         print("\n out_3*3:{}\n".format(sbs_out.shape))
#         out.append(sbs_out)
        
#         obo = self.conv5x5(sbs)
#         obo_out = self.maxpool(obo)
#         obo_out = (obo_out.flatten(2)).transpose(1,2)
#         out.append(obo_out)
#         print("\n out_5*5:{}\n".format(obo_out.shape))
        
#         cbc = self.conv7x7(obo)
#         cbc_out = self.maxpool(cbc)
#         cbc_out = (cbc_out.flatten(2)).transpose(1,2)
#         print("\n out_7*7:{}\n".format(cbc_out.shape))
#         out.append(cbc_out)
        all_t = torch.cat(out, 1)
        all_t = self.norm(all_t)
        print("\n cat_together:{}\n".format(all_t.shape))
#         all_t_b = self.batchnorm_1(all_t)
#         out = all_t_b.add(res)
#         out = self.relu(out)
#         out = self.batchnorm_2(out)      
        
        return all_t
    
class MultiResBlock_3(nn.Module):
    def __init__(self,in_ch,U,branch=1,downsample=2, alpha=1):
        super(MultiResBlock_3,self).__init__()
        self.W = alpha * U
        self.one_ch = conv_block(in_ch, 1)
#         self.residual_layer = conv_block(1, self.W, 1, 1, 0, act=None)
        self.residual_layer = conv_block(1, self.W)
#         self.conv3x3 = conv_block(1, int(self.W*0.167))
#         self.conv5x5 = conv_block(int(self.W*0.167), int(self.W*0.333))
#         self.conv7x7 = conv_block(int(self.W*0.333), self.W-int(self.W*0.167)-int(self.W*0.333))
        self.conv3x3 = conv_block(1, int(self.W))
        self.conv5x5 = conv_block(int(self.W), int(self.W))
        self.conv7x7 = conv_block(int(self.W), self.W)
        self.maxpool = nn.MaxPool2d(downsample, stride=downsample)
        self.relu = nn.ReLU(inplace=True)
#         self.batchnorm_1 = nn.BatchNorm2d(int(self.W*0.167)+int(self.W*0.333)+int(self.W*0.5))
#         self.batchnorm_2 = nn.BatchNorm2d(int(self.W*0.167)+int(self.W*0.333)+int(self.W*0.5))
#         self.batchnorm_1 = nn.BatchNorm2d(self.W)
#         self.batchnorm_2 = nn.BatchNorm2d(self.W)
        self.norm = nn.LayerNorm(self.W)
        
    def forward(self, x):
        out = []
        print(x.shape) 
        print("\n W=alpha*U :{}\n".format(self.W))
        x = self.one_ch(x) 
#         res = self.residual_layer(x)
#         res_out = self.maxpool(res)
#         res_out = (res_out.flatten(2)).transpose(1,2)
#         out.append(res_out)
#         print("\n res:{}\n".format(res_out.shape))
        
        sbs = self.conv3x3(x)
        sbs_out = self.maxpool(sbs)
        sbs_out = (sbs_out.flatten(2)).transpose(1,2)
        print("\n out_3*3:{}\n".format(sbs_out.shape))
        out.append(sbs_out)
        
#         obo = self.conv5x5(sbs)
#         obo_out = self.maxpool(obo)
#         obo_out = (obo_out.flatten(2)).transpose(1,2)
#         out.append(obo_out)
#         print("\n out_5*5:{}\n".format(obo_out.shape))
        
#         cbc = self.conv7x7(obo)
#         cbc_out = self.maxpool(cbc)
#         cbc_out = (cbc_out.flatten(2)).transpose(1,2)
#         print("\n out_7*7:{}\n".format(cbc_out.shape))
#         out.append(cbc_out)
        all_t = torch.cat(out, 1)
        all_t = self.norm(all_t)
        print("\n cat_together:{}\n".format(all_t.shape))
#         all_t_b = self.batchnorm_1(all_t)
#         out = all_t_b.add(res)
#         out = self.relu(out)
#         out = self.batchnorm_2(out)      
        
        return all_t
    
class MultiResBlock_5(nn.Module):
    def __init__(self,in_ch,U,branch=1,downsample=2, alpha=1):
        super(MultiResBlock_5,self).__init__()
        self.W = alpha * U
        self.one_ch = conv_block(in_ch, 1)
#         self.residual_layer = conv_block(1, self.W, 1, 1, 0, act=None)
        self.residual_layer = conv_block(1, self.W)
#         self.conv3x3 = conv_block(1, int(self.W*0.167))
#         self.conv5x5 = conv_block(int(self.W*0.167), int(self.W*0.333))
#         self.conv7x7 = conv_block(int(self.W*0.333), self.W-int(self.W*0.167)-int(self.W*0.333))
        self.conv3x3 = conv_block(1, int(self.W))
        self.conv5x5 = conv_block(int(self.W), int(self.W))
        self.conv7x7 = conv_block(int(self.W), self.W)
        self.maxpool = nn.MaxPool2d(downsample, stride=downsample)
        self.relu = nn.ReLU(inplace=True)
#         self.batchnorm_1 = nn.BatchNorm2d(int(self.W*0.167)+int(self.W*0.333)+int(self.W*0.5))
#         self.batchnorm_2 = nn.BatchNorm2d(int(self.W*0.167)+int(self.W*0.333)+int(self.W*0.5))
#         self.batchnorm_1 = nn.BatchNorm2d(self.W)
#         self.batchnorm_2 = nn.BatchNorm2d(self.W)
        self.norm = nn.LayerNorm(self.W)
        
    def forward(self, x):
        out = []
        print(x.shape) 
        print("\n W=alpha*U :{}\n".format(self.W))
        x = self.one_ch(x) 
#         res = self.residual_layer(x)
#         res_out = self.maxpool(res)
#         res_out = (res_out.flatten(2)).transpose(1,2)
#         out.append(res_out)
#         print("\n res:{}\n".format(res_out.shape))
        
        sbs = self.conv3x3(x)
#         sbs_out = self.maxpool(sbs)
#         sbs_out = (sbs_out.flatten(2)).transpose(1,2)
#         print("\n out_3*3:{}\n".format(sbs_out.shape))
#         out.append(sbs_out)
        
        obo = self.conv5x5(sbs)
        obo_out = self.maxpool(obo)
        obo_out = (obo_out.flatten(2)).transpose(1,2)
        out.append(obo_out)
        print("\n out_5*5:{}\n".format(obo_out.shape))
        
#         cbc = self.conv7x7(obo)
#         cbc_out = self.maxpool(cbc)
#         cbc_out = (cbc_out.flatten(2)).transpose(1,2)
#         print("\n out_7*7:{}\n".format(cbc_out.shape))
#         out.append(cbc_out)
        
        all_t = torch.cat(out, 1)
        all_t = self.norm(all_t)
        print("\n cat_together:{}\n".format(all_t.shape))
#         all_t_b = self.batchnorm_1(all_t)
#         out = all_t_b.add(res)
#         out = self.relu(out)
#         out = self.batchnorm_2(out)      
        
        return all_t
    
    
    
class MultiResBlock_7(nn.Module):
    def __init__(self,in_ch,U,branch=1,downsample=2, alpha=1):
        super(MultiResBlock_7,self).__init__()
        self.W = alpha * U
        self.one_ch = conv_block(in_ch, 1)
#         self.residual_layer = conv_block(1, self.W, 1, 1, 0, act=None)
        self.residual_layer = conv_block(1, self.W)
#         self.conv3x3 = conv_block(1, int(self.W*0.167))
#         self.conv5x5 = conv_block(int(self.W*0.167), int(self.W*0.333))
#         self.conv7x7 = conv_block(int(self.W*0.333), self.W-int(self.W*0.167)-int(self.W*0.333))
        self.conv3x3 = conv_block(1, int(self.W))
        self.conv5x5 = conv_block(int(self.W), int(self.W))
        self.conv7x7 = conv_block(int(self.W), self.W)
        self.maxpool = nn.MaxPool2d(downsample, stride=downsample)
        self.relu = nn.ReLU(inplace=True)
#         self.batchnorm_1 = nn.BatchNorm2d(int(self.W*0.167)+int(self.W*0.333)+int(self.W*0.5))
#         self.batchnorm_2 = nn.BatchNorm2d(int(self.W*0.167)+int(self.W*0.333)+int(self.W*0.5))
#         self.batchnorm_1 = nn.BatchNorm2d(self.W)
#         self.batchnorm_2 = nn.BatchNorm2d(self.W)
        self.norm = nn.LayerNorm(self.W)
        
    def forward(self, x):
        out = []
        print(x.shape) 
        print("\n W=alpha*U :{}\n".format(self.W))
        x = self.one_ch(x) 
#         res = self.residual_layer(x)
#         res_out = self.maxpool(res)
#         res_out = (res_out.flatten(2)).transpose(1,2)
#         out.append(res_out)
#         print("\n res:{}\n".format(res_out.shape))
        
        sbs = self.conv3x3(x)
#         sbs_out = self.maxpool(sbs)
#         sbs_out = (sbs_out.flatten(2)).transpose(1,2)
#         print("\n out_3*3:{}\n".format(sbs_out.shape))
#         out.append(sbs_out)
        
        obo = self.conv5x5(sbs)
#         obo_out = self.maxpool(obo)
#         obo_out = (obo_out.flatten(2)).transpose(1,2)
#         out.append(obo_out)
#         print("\n out_5*5:{}\n".format(obo_out.shape))
        
        cbc = self.conv7x7(obo)
        cbc_out = self.maxpool(cbc)
        cbc_out = (cbc_out.flatten(2)).transpose(1,2)
        print("\n out_7*7:{}\n".format(cbc_out.shape))
        out.append(cbc_out)
        all_t = torch.cat(out, 1)
        all_t = self.norm(all_t)
        print("\n cat_together:{}\n".format(all_t.shape))
#         all_t_b = self.batchnorm_1(all_t)
#         out = all_t_b.add(res)
#         out = self.relu(out)
#         out = self.batchnorm_2(out)      
        
        return all_t





# class MiT_3inception_padding(nn.Module):
#     #really cannot be used...
#     def __init__(self, image_size, in_dim, key_dim, value_dim, layers, head_count=1, dil_conv=1, token_mlp='mix_skip'):
#         super().__init__()

#         self.Hs=[56, 28, 14, 7]
#         self.Ws=[56, 28, 14, 7]
#         patch_sizes = [7, 3, 3, 3]
#         strides = [4, 2, 2, 2]
#         padding_sizes = [3, 1, 1, 1]
#         if dil_conv:  
#             dilation = 2 
#             # # conv 5
#             # patch_sizes1 = [7, 3, 3, 3]
#             # patch_sizes2 = [5, 1, 1, 1]                
#             # dil_padding_sizes1 = [3, 0, 0, 0]
#             # dil_padding_sizes2 = [3, 0, 0, 0]
#             # # conv 3
#             # patch_sizes1 = [7, 3, 3, 3]
#             # dil_padding_sizes1 = [3, 0, 0, 0]    
#             # patch_sizes2 = [3, 1, 1, 1]
#             # dil_padding_sizes2 = [1, 0, 0, 0]
#             # conv 1
#             patch_sizes1 = [7, 3, 3, 3]
#             dil_padding_sizes1 = [3, 0, 0, 0]    
#             patch_sizes2 = [1, 1, 1, 1]
#             dil_padding_sizes2 = [0, 0, 0, 0]
#         else:
#             dilation = 1
#             patch_sizes1 = [7, 3, 3, 3]
#             patch_sizes2 = [5, 1, 1, 1]
#             dil_padding_sizes1 = [3, 1, 1, 1]
#             # dil_padding_sizes2 = [3, 0, 0, 0]
#             dil_padding_sizes2 = [1, 0, 0, 0]


#         # 1 by 1 convolution to alter the dimension
#         self.conv1_1_s1 = nn.Conv2d(2*in_dim[0], in_dim[0], 1)
#         self.conv1_1_s2 = nn.Conv2d(2*in_dim[1], in_dim[1], 1)
#         self.conv1_1_s3 = nn.Conv2d(2*in_dim[2], in_dim[2], 1)
#         self.conv1_1_s4 = nn.Conv2d(2*in_dim[3], in_dim[3], 1)

#         # patch_embed
#         # layers = [2, 2, 2, 2] dims = [64, 128, 320, 512]
#         self.patch_embed1 = OverlapPatchEmbeddings(image_size, patch_sizes[0], strides[0], padding_sizes[0], 3, in_dim[0])
        
#         self.patch_embed2_1 = OverlapPatchEmbeddings_fuse_padding(image_size//4, patch_sizes1[1], strides[1], dil_padding_sizes1[1],dilation, in_dim[0], in_dim[1], self.Hs[1])
#         self.patch_embed2_2 = OverlapPatchEmbeddings_fuse_padding(image_size//4, patch_sizes2[1], strides[1], dil_padding_sizes2[1],dilation, in_dim[0], in_dim[1], self.Hs[1])

#         self.patch_embed3_1 = OverlapPatchEmbeddings_fuse_padding(image_size//8, patch_sizes1[2], strides[2], dil_padding_sizes1[2],dilation, in_dim[1], in_dim[2], self.Hs[2])
#         self.patch_embed3_2 = OverlapPatchEmbeddings_fuse_padding(image_size//8, patch_sizes2[2], strides[2], dil_padding_sizes2[2],dilation, in_dim[1], in_dim[2], self.Hs[2])

#         self.patch_embed4_1 = OverlapPatchEmbeddings_fuse_padding(image_size//16, patch_sizes1[3], strides[3], dil_padding_sizes1[3],dilation, in_dim[2], in_dim[3], self.Hs[3])
#         self.patch_embed4_2 = OverlapPatchEmbeddings_fuse_padding(image_size//16, patch_sizes2[3], strides[3], dil_padding_sizes2[3],dilation, in_dim[2], in_dim[3], self.Hs[3])
        
#         # transformer encoder
#         self.block1 = nn.ModuleList([ 
#             EfficientTransformerBlock(in_dim[0], key_dim[0], value_dim[0], head_count, token_mlp)
#         for _ in range(layers[0])])
#         self.norm1 = nn.LayerNorm(in_dim[0])

#         self.block2 = nn.ModuleList([
#             EfficientTransformerBlockFuse(in_dim[1], key_dim[1], value_dim[1], head_count, token_mlp)
#         for _ in range(layers[1])])
#         self.norm2 = nn.LayerNorm(in_dim[1])

#         self.block3 = nn.ModuleList([
#             EfficientTransformerBlockFuse(in_dim[2], key_dim[2], value_dim[2], head_count, token_mlp)
#         for _ in range(layers[2])])
#         self.norm3 = nn.LayerNorm(in_dim[2])

#         self.block4 = nn.ModuleList([
#             EfficientTransformerBlockFuse(in_dim[3], key_dim[3], value_dim[3], head_count, token_mlp)
#         for _ in range(layers[3])])
#         self.norm4 = nn.LayerNorm(in_dim[3])
        

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         B = x.shape[0]
#         outs = []

#         # stage 1
#         x, H, W = self.patch_embed1(x)
#         for blk in self.block1:
#             x = blk(x, H, W)
#         x = self.norm1(x)
#         x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
#         outs.append(x)

      
#         # incept module1
        
#         x1, H1, W1 = self.patch_embed2_1(x)
#         _, nfx1_len, _ = x1.shape
#         x2, H2, W2 = self.patch_embed2_2(x)
#         _, nfx2_len, _ = x2.shape
#         nfx_cat = torch.cat((x1,x2),1)

#         # stage 2
#         # print("-------EN: Stage 2------\n\n")
#         for blk in self.block2:
#             nfx_cat = blk(nfx_cat, nfx1_len, nfx2_len, H1, W1, H2, W2)
#         nfx_cat = self.norm2(nfx_cat)
#         mx1 = nfx_cat[:, :nfx1_len, :]
#         mx2 = nfx_cat[:, nfx1_len:, :]
#         b, _, _ = mx1.shape
#         map_mx1 = mx1.reshape(b,H1,W1,-1)
#         map_mx2 = mx2.reshape(b,H2,W2,-1)
#         map_mx1 = map_mx1.permute(0,3,1,2)
#         map_mx2 = map_mx2.permute(0,3,1,2)
#         # map_mx1 = F.interpolate(map_mx1,[self.Hs[1], self.Ws[1]])
#         cat_maps = torch.cat((map_mx1, map_mx2),1)
#         x = self.conv1_1_s2(cat_maps)
#         outs.append(x)

#         # incept module 2
            
#         x1, H1, W1 = self.patch_embed3_1(x)
#         _, nfx1_len, _ = x1.shape
#         x2, H2, W2 = self.patch_embed3_2(x)
#         _, nfx2_len, _ = x2.shape
#         nfx_cat = torch.cat((x1,x2),1)

#         # stage 3  
#         for blk in self.block3:
#             nfx_cat = blk(nfx_cat, nfx1_len, nfx2_len, H1, W1, H2, W2)
#         nfx_cat = self.norm3(nfx_cat)
#         mx1 = nfx_cat[:, :nfx1_len, :]
#         mx2 = nfx_cat[:, nfx1_len: :]
#         b, _, _ = mx1.shape
#         map_mx1 = mx1.reshape(b,H1,W1,-1)
#         map_mx2 = mx2.reshape(b,H2,W2,-1)
#         map_mx1 = map_mx1.permute(0,3,1,2)
#         map_mx2 = map_mx2.permute(0,3,1,2)
#         # map_mx1 = F.interpolate(map_mx1,[self.Hs[2], self.Ws[2]])
#         cat_maps = torch.cat((map_mx1, map_mx2),1)
#         x = self.conv1_1_s3(cat_maps)
#         outs.append(x)

        
#         # incept module 3
#         x1, H1, W1 = self.patch_embed4_1(x)
#         _, nfx1_len, _ = x1.shape
#         x2, H2, W2 = self.patch_embed4_2(x)
#         _, nfx2_len, _ = x2.shape
#         nfx_cat = torch.cat((x1,x2),1)

#         # stage 4
#         for blk in self.block4:
#             nfx_cat = blk(nfx_cat, nfx1_len, nfx2_len, H1, W1, H2, W2)
#         nfx_cat = self.norm4(nfx_cat)
#         mx1 = nfx_cat[:, :nfx1_len, :]
#         mx2 = nfx_cat[:, nfx1_len: :]
#         b, _, _ = mx1.shape
#         map_mx1 = mx1.reshape(b,H1,W1,-1)
#         map_mx2 = mx2.reshape(b,H2,W2,-1)
#         map_mx1 = map_mx1.permute(0,3,1,2)
#         map_mx2 = map_mx2.permute(0,3,1,2)
#         # map_mx1 = F.interpolate(map_mx1,[self.Hs[3], self.Ws[3]])
#         cat_maps = torch.cat((map_mx1, map_mx2),1)
#         x = self.conv1_1_s4(cat_maps)
#         outs.append(x)

#         return outs