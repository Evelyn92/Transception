from cmath import sqrt
from re import X
import torch
import torch.nn as nn
from torch import einsum
from typing import Tuple
from einops import rearrange
from einops.layers.torch import Rearrange
from torch.nn import functional as F
from torch.nn import Module, Conv2d, Parameter, Softmax
from torchvision import models
from torchinfo import summary
# from torchstat import stat

from functools import partial
# from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
# from timm.models.layers import DropPath, trunc_normal_


class DWConv(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)

    def forward(self, x: torch.Tensor, H, W) -> torch.Tensor:
        B, N, C = x.shape
        # print('input in DWConv: {}'.format(x.shape))
        tx = x.transpose(1, 2).view(B, C, H, W)
        conv_x = self.dwconv(tx)
        return conv_x.flatten(2).transpose(1, 2)


 
class MixFFN(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.fc1 = nn.Linear(c1, c2)
        self.dwconv = DWConv(c2)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(c2, c1)
        
    def forward(self, x: torch.Tensor, H, W) -> torch.Tensor:
        ax = self.act(self.dwconv(self.fc1(x), H, W))
        out = self.fc2(ax)
        return out

class MixFFN_skip(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.fc1 = nn.Linear(c1, c2)
        self.dwconv = DWConv(c2)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(c2, c1)
        self.norm1 = nn.LayerNorm(c2)
        self.norm2 = nn.LayerNorm(c2)
        self.norm3 = nn.LayerNorm(c2)
    def forward(self, x: torch.Tensor, H, W) -> torch.Tensor:
        ax = self.act(self.norm1(self.dwconv(self.fc1(x), H, W)+self.fc1(x)))
        out = self.fc2(ax)
        return out

class MLP_FFN(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.fc1 = nn.Linear(c1, c2)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(c2, c1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class EfficientAttention(nn.Module):
    """
        input  -> x:[B, D, H, W]
        output ->   [B, D, H, W]
    
        in_channels:    int -> Embedding Dimension 
        key_channels:   int -> Key Embedding Dimension,   Best: (in_channels)
        value_channels: int -> Value Embedding Dimension, Best: (in_channels or in_channels//2) 
        head_count:     int -> It divides the embedding dimension by the head_count and process each part individually
        
        Conv2D # of Params:  ((k_h * k_w * C_in) + 1) * C_out)
    """
    
    def __init__(self, in_channels, key_channels, value_channels, head_count=1):
        super().__init__()
        self.in_channels = in_channels
        self.key_channels = key_channels
        self.head_count = head_count
        self.value_channels = value_channels

        self.keys = nn.Conv2d(in_channels, key_channels, 1) 
        self.queries = nn.Conv2d(in_channels, key_channels, 1)
        self.values = nn.Conv2d(in_channels, value_channels, 1)
        self.reprojection = nn.Conv2d(value_channels, in_channels, 1)

        
    def forward(self, input_):
        n, _, h, w = input_.size()
        # n, _,  = input_.size()
        keys = self.keys(input_).reshape((n, self.key_channels, h * w))
        queries = self.queries(input_).reshape(n, self.key_channels, h * w)
        values = self.values(input_).reshape((n, self.value_channels, h * w))
        
        head_key_channels = self.key_channels // self.head_count
        head_value_channels = self.value_channels // self.head_count
        
        attended_values = []
        for i in range(self.head_count):
            key = F.softmax(keys[
                :,
                i * head_key_channels: (i + 1) * head_key_channels,
                :
            ], dim=2)
            
            query = F.softmax(queries[
                :,
                i * head_key_channels: (i + 1) * head_key_channels,
                :
            ], dim=1)
                        
            value = values[
                :,
                i * head_value_channels: (i + 1) * head_value_channels,
                :
            ]            
            
            context = key @ value.transpose(1, 2) # dk*dv
            attended_value = (context.transpose(1, 2) @ query).reshape(n, head_value_channels, h, w) # n*dv            
            attended_values.append(attended_value)
                
        aggregated_values = torch.cat(attended_values, dim=1)
        attention = self.reprojection(aggregated_values)

        return attention


class EfficientTransformerBlock(nn.Module):
    """
        Input  -> x (Size: (b, (H*W), d)), H, W
        Output -> (b, (H*W), d)
    """
    def __init__(self, in_dim, key_dim, value_dim, head_count=1, token_mlp='mix'):
        super().__init__()
        self.norm1 = nn.LayerNorm(in_dim)
        self.attn = EfficientAttention(in_channels=in_dim, key_channels=key_dim,
                                       value_channels=value_dim, head_count=1)
        self.norm2 = nn.LayerNorm(in_dim)
        if token_mlp=='mix':
            self.mlp = MixFFN(in_dim, int(in_dim*4))  
        elif token_mlp=='mix_skip':
            self.mlp = MixFFN_skip(in_dim, int(in_dim*4)) 
        else:
            self.mlp = MLP_FFN(in_dim, int(in_dim*4))

    def forward(self, x: torch.Tensor, H, W) -> torch.Tensor:
        norm_1 = self.norm1(x)
        norm_1 = Rearrange('b (h w) d -> b d h w', h=H, w=W)(norm_1)
        
        attn = self.attn(norm_1)
        attn = Rearrange('b d h w -> b (h w) d')(attn)
        
        tx = x + attn
        mx = tx + self.mlp(self.norm2(tx), H, W)
        return mx   
    

class PatchExpand(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.expand = nn.Linear(dim, 2*dim, bias=False) if dim_scale==2 else nn.Identity()
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x): 
        """
        x: B, H*W, C
        """
        # print("x_shape-----",x.shape)
        H, W = self.input_resolution
        x = self.expand(x)
        
        B, L, C = x.shape
        # print(x.shape)
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=2, p2=2, c=C//4)
        x = x.view(B,-1,C//4)
        x= self.norm(x.clone())

        return x
    
class FinalPatchExpand_X4(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(dim, 16*dim, bias=False)
        self.output_dim = dim 
        self.norm = norm_layer(self.output_dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale, c=C//(self.dim_scale**2))
        x = x.view(B,-1,self.output_dim)
        x= self.norm(x.clone())

        return x
  

class MyDecoderLayer(nn.Module):
    def __init__(self, input_size, in_out_chan, head_count, token_mlp_mode, n_class=9,
                 norm_layer=nn.LayerNorm, is_last=False):
        super().__init__()
        dims = in_out_chan[0]
        out_dim = in_out_chan[1]
        key_dim = in_out_chan[2]
        value_dim = in_out_chan[3]
        if not is_last:
            self.concat_linear = nn.Linear(dims*2, out_dim)
            # transformer decoder
            self.layer_up = PatchExpand(input_resolution=input_size, dim=out_dim, dim_scale=2, norm_layer=norm_layer)
            self.last_layer = None
        else:
            self.concat_linear = nn.Linear(dims*4, out_dim)
            # transformer decoder
            self.layer_up = FinalPatchExpand_X4(input_resolution=input_size, dim=out_dim, dim_scale=4, norm_layer=norm_layer)
            # self.last_layer = nn.Linear(out_dim, n_class)
            self.last_layer = nn.Conv2d(out_dim, n_class,1)
            # self.last_layer = None

        self.layer_former_1 = EfficientTransformerBlock(out_dim, key_dim, value_dim, head_count, token_mlp_mode)
        self.layer_former_2 = EfficientTransformerBlock(out_dim, key_dim, value_dim, head_count, token_mlp_mode)
       

        def init_weights(self): 
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.LayerNorm):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)
                elif isinstance(m, nn.Conv2d):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

        init_weights(self)
      
    def forward(self, x1, x2=None):
        if x2 is not None:# skip connection exist
            b, h, w, c = x2.shape
            x2 = x2.view(b, -1, c)
            cat_x = torch.cat([x1, x2], dim=-1)
            cat_linear_x = self.concat_linear(cat_x)
            tran_layer_1 = self.layer_former_1(cat_linear_x, h, w)
            tran_layer_2 = self.layer_former_2(tran_layer_1, h, w)
            
            if self.last_layer:
                out = self.last_layer(self.layer_up(tran_layer_2).view(b, 4*h, 4*w, -1).permute(0,3,1,2)) 
            else:
                out = self.layer_up(tran_layer_2)
        else:
            # if len(x1.shape)>3:
            #     x1 = x1.permute(0,2,3,1)
            #     b, h, w, c = x1.shape
            #     x1 = x1.view(b, -1, c)
            out = self.layer_up(x1)
        return out

class OverlapPatchEmbeddings(nn.Module):
    def __init__(self, img_size=224, patch_size=7, stride=4, padding=1, in_ch=3, dim=768):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_ch, dim, patch_size, stride, padding)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        px = self.proj(x)
        _, _, H, W = px.shape
        fx = px.flatten(2).transpose(1, 2)
        nfx = self.norm(fx)
        return nfx, H, W  


## MSViT modules

class DWConv2d_BN(nn.Module):
    """
    Depthwise Separable Conv
    """

    def __init__(
        self,
        in_ch,
        out_ch,
        kernel_size=1,
        stride=1,
        norm_layer=nn.BatchNorm2d,
        act_layer=nn.Hardswish,
        bn_weight_init=1,
        norm_cfg="BN",
    ):
        super().__init__()

        # dw
        self.dwconv = nn.Conv2d(
            in_ch,
            out_ch,
            kernel_size,
            stride,
            (kernel_size - 1) // 2,
            groups=out_ch,
            bias=False,
        )
        # pw-linear
        self.pwconv = nn.Conv2d(out_ch, out_ch, 1, 1, 0, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = act_layer() if act_layer is not None else nn.Identity()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
            #     n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            #     m.weight.data.normal_(0, sqrt(2.0 / n))
            #     if m.bias is not None:
            #         m.bias.data.zero_()
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(bn_weight_init)
                m.bias.data.zero_()

    def forward(self, x):

        x = self.dwconv(x)
        x = self.pwconv(x)
        x = self.bn(x)
        x = self.act(x)

        return x

class Conv2d_BN(nn.Module):
    def __init__(
        self,
        in_ch,
        out_ch,
        kernel_size=1,
        stride=1,
        pad=0,
        dilation=1,
        groups=1,
        bn_weight_init=1,
        act_layer=None,
        norm_cfg="BN",
    ):
        super().__init__()
        # self.add_module('c', torch.nn.Conv2d(
        #     a, b, ks, stride, pad, dilation, groups, bias=False))
        self.conv = nn.Conv2d(
            in_ch, out_ch, kernel_size, stride, pad, dilation, groups, bias=False
        )
        self.bn = nn.BatchNorm2d(out_ch)

        torch.nn.init.constant_(self.bn.weight, bn_weight_init)
        torch.nn.init.constant_(self.bn.bias, 0)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                # Note that there is no bias due to BN
                # fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(mean=0.0, std=sqrt(2.0 / fan_out))

        self.act_layer = act_layer() if act_layer is not None else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act_layer(x)

        return x

class Conv3d_BN_concat(nn.Module):
    # input: attention list of length #path
    def __init__(
        self,
        in_ch,
        out_ch,
        kernel_size=1,
        stride=1,
        pad=0,
        dilation=1,
        groups=1,
        bn_weight_init=1,
        act_layer=None,
        norm_cfg="BN",
    ):
        super().__init__()
        # self.add_module('c', torch.nn.Conv2d(
        #     a, b, ks, stride, pad, dilation, groups, bias=False))
        # self.conv = nn.Conv2d(
        #     in_ch, out_ch, kernel_size, stride, pad, dilation, groups, bias=False
        # )
        self.bn = nn.BatchNorm2d(out_ch)

        torch.nn.init.constant_(self.bn.weight, bn_weight_init)
        torch.nn.init.constant_(self.bn.bias, 0)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                # Note that there is no bias due to BN
                # fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(mean=0.0, std=sqrt(2.0 / fan_out))

        # self.act_layer = act_layer() if act_layer is not None else nn.Identity()
        self.interact_concat = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=(4,1,1)),
            nn.ReLU()
        )

    def forward(self, x_list):
        
        b,c,h,w = x_list[0].shape
        out_3d = []
        for ip in range(len(x_list)):
            x = x_list[ip]
            x = x.unsqueeze_(dim=2)
            out_3d.append(x)
            # print(f'{ip} path extend to shape{x.shape}')

        x = torch.cat(out_3d, dim=2)
        # print(f'after concat: {x.shape}')
        x = torch.squeeze(self.interact_concat(x), dim=2)
        # print(f'after squeeze: {x.shape}')
        x = self.bn(x)

        return x

class CAM_Module(nn.Module):
    """ Channel attention module"""

    def __init__(self, in_dim):
        super().__init__()
        self.channel_in = in_dim

        self.gamma = Parameter(torch.zeros(1))
        self.softmax = Softmax(dim=-1)
        self.numpath = 3

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B  C  numPath H  W)
            returns :
                out : attention value + input feature
                attention: B X C X numPath X numPath
        """
        m_batchsize, C,numpath, height, width = x.size()
        proj_query = x.reshape(m_batchsize, C, numpath, -1) #b c 4 n
        # print(f'Shape of query: {proj_query.shape}')
        proj_key = x.reshape(m_batchsize, C,numpath, -1).permute(0, 1, 3, 2) #b c n 4
        # print(f'Shape of key: {proj_key.shape}') 
        energy = torch.matmul(proj_query, proj_key) # b c 4 4
        print(f'Shape of energy: {energy[0][0]}')
        max_energy_0 = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)
        energy_new = max_energy_0 - energy# The larger dependency of the channels, the lower energy_new value/ used to emphasizing the different information
        
        # min_energy_0 = torch.min(energy, -1, keepdim=True)[0].expand_as(energy)
        # energy_new = energy - min_energy_0
        print(f'energy_new: {energy_new[0][0]}')
        
        attention = self.softmax(energy_new)
        print(f'Shape of attention: {attention[0][0]}\n')
        proj_value = x.reshape(m_batchsize, C, numpath, -1)
        # print(f'Shape of proj_value: {proj_value.shape}')

        out = torch.matmul(attention, proj_value)# can be replace by torch.matmul
        out = out.reshape(m_batchsize, C, numpath, height, width)


#         logging.debug('cam device: {}, {}'.format(out.device, self.gamma.device))
        gamma = self.gamma.to(out.device)
        out = gamma * out + x
        return out


class CAM_Factorized_Module(nn.Module):
    """ Channel attention module"""

    def __init__(self, in_dim):
        super().__init__()
        self.channel_in = in_dim
        
        self.gamma = Parameter(torch.zeros(1))
        self.softmax = Softmax(dim=-1)
        self.numpath = 3
        self.num_heads = 8
        self.scale = (in_dim // self.num_heads) ** -0.5
        self.qkv = nn.Linear(in_dim, in_dim * 3)
        self.proj = nn.Linear(in_dim, in_dim)
        crpe_window={3: 2, 5: 3, 7: 3}
        # self.crpe = shared_crpe
        self.crpe = ConvRelPosEnc(Ch=in_dim // self.num_heads, h=self.num_heads, window=crpe_window)

    def forward(self, x):
        B, C, numpath, height, width = x.size()
        x1 = x.reshape(B,C,-1).permute(0,2,1)
        B,N,C = x1.size()
        # print(f'B:{B} N:{N} C:{C}')
        qkv = (
            self.qkv(x1)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
            .contiguous()
        ) 
        q, k, v = qkv[0], qkv[1], qkv[2]  # Shape: [B, h, N, Ch].
        # print(f'shape q: {q.shape}')
        # Factorized attention.
        k_softmax = k.softmax(dim=2)  # Softmax on dim N.
        k_softmax_T_dot_v = einsum(
            "b h n k, b h n v -> b h k v", k_softmax, v
        )  # Shape: [B, h, Ch, Ch].
        factor_att = einsum(
            "b h n k, b h k v -> b h n v", q, k_softmax_T_dot_v
        )  # Shape: [B, h, N, Ch].
        # print(f'attention: {factor_att.shape}')
        # Convolutional relative position encoding.
        # size=(height, width)
        # crpe = self.crpe(q, v, size=size)  # Shape: [B, h, N, Ch].
        # Merge and reshape.
        # out = self.scale * factor_att + crpe
        out = self.scale * factor_att 
        out = (
            out.transpose(1, 2).reshape(B, N, C).contiguous()
        )  # Shape: [B, h, N, Ch] -> [B, N, h, Ch] -> [B, N, C].

        # Output projection.
        out = self.proj(out) #[B, N, C]
        out = out.permute(0,2,1).reshape(B,C,numpath,height, width)#[B,C,N]
        # print(f'out shape: {out.shape}')
        gamma = self.gamma.to(out.device)
        out = gamma * out + x
        return out


class SE_Block(nn.Module):
    def __init__(self, in_ch, out_ch, r=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(in_ch, in_ch // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_ch // r, in_ch, bias=False),
            nn.Sigmoid()
        )
        self.conv = torch.nn.Conv2d(in_ch, out_ch, kernel_size = (1,1))
        self.act  = torch.nn.ReLU()
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        # print('start using se block')
        bs, c, _, _ = x.shape
        y = self.squeeze(x).view(bs, c)
        y = self.excitation(y).view(bs, c, 1, 1)
        x = x * y.expand_as(x)
        x = self.conv(x)
        x = self.act(self.bn(x))
        
        return x


class Conv3d_BN_channel_attention_concat(nn.Module):
    # input: attention list of length #path
    def __init__(
        self,
        in_ch,
        out_ch,
        kernel_size=1,
        stride=1,
        pad=0,
        dilation=1,
        groups=1,
        bn_weight_init=1,
        act_layer=None,
        norm_cfg="BN",
        cam='cam'
    ):
        super().__init__()
    
        self.bn = nn.BatchNorm2d(out_ch)

        torch.nn.init.constant_(self.bn.weight, bn_weight_init)
        torch.nn.init.constant_(self.bn.bias, 0)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
       
        self.interact_concat = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=(4,1,1)),
            nn.GELU()
        )
        if cam == 'cam':
            self.channelAttention = CAM_Module(in_dim = in_ch)
            # print('cam chosen')
        elif cam == 'cam_fact':
            self.channelAttention = CAM_Factorized_Module(in_dim = in_ch)
            # print('factorized module chosen')
        else:
            self.channelAttention = CAM_Factorized_Module(in_dim = in_ch)
   

        self.bn3d = nn.BatchNorm3d(in_ch)

    def forward(self, x_list):
        b,c,h,w = x_list[0].shape
        # print(f"b:{b} c:{c} h:{h} w:{w}")
     
        out_3d = []
        for ip in range(len(x_list)):
#             print(f'ip:{ip}')
            in_x = x_list[ip]  
#             print(f'in each {ip} iteration: {in_x.shape}')
            in_x = in_x.unsqueeze_(dim=2)
            out_3d.append(in_x)
#             print(f'{ip} path extend to shape: {in_x.shape}')

            x = torch.cat(out_3d, dim=2)
            x = self.bn3d(x)
        

        print(f'before channel attention: {x.shape}')
        x = self.channelAttention(x)
        x = self.bn3d(x)
        # print(f'after channel: {x.shape}')
        x = torch.squeeze(self.interact_concat(x), dim=2)
        # print(f'after squeeze: {x.shape}')
        x = self.bn(x)

        return x



class DWCPatchEmbed(nn.Module):
    """
    Depthwise Convolutional Patch Embedding layer
    Image to Patch Embedding
    """

    def __init__(
        self,
        in_chans=3,
        embed_dim=768,
        patch_size=16,
        stride=1,
        pad=0,
        act_layer=nn.Hardswish,
        norm_cfg='BN',
    ):
        super().__init__()
        self.stride = stride
        # TODO : confirm whether act_layer is effective or not
        self.patch_conv = DWConv2d_BN(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=stride,
            act_layer=nn.Hardswish,
            norm_cfg=norm_cfg,
        )

    def forward(self, x):
        x = self.patch_conv(x)
        # print(f'stride: {self.stride}')

        return x

class Patch_Embed_stage(nn.Module):

    def __init__(self, embed_dim, num_path=3, isPool=False, norm_cfg=dict(type="BN")):
        super(Patch_Embed_stage, self).__init__()

        self.patch_embeds = nn.ModuleList(
            [
                DWCPatchEmbed(
                    in_chans=embed_dim,
                    embed_dim=embed_dim,
                    patch_size=3,
                    stride=2 if isPool and idx == 0 else 1,
                    pad=1,
                    norm_cfg='BN',
                )
                for idx in range(num_path)
            ]
        )

        # scale

    def forward(self, x):
        att_inputs = []
        for pe in self.patch_embeds:
            x = pe(x)
            # print(f'patch_embedding:{x.shape}')
            att_inputs.append(x)

        return att_inputs

class ConvPosEnc(nn.Module):
    """Convolutional Position Encoding.
    Note: This module is similar to the conditional position encoding in CPVT.
    """

    def __init__(self, dim, k=3):
        super(ConvPosEnc, self).__init__()

        self.proj = nn.Conv2d(dim, dim, k, 1, k // 2, groups=dim)

    def forward(self, x, size):
        B, N, C = x.shape
        H, W = size

        feat = x.transpose(1, 2).contiguous().reshape(B, C, H, W)
        x = self.proj(feat) + feat
        x = x.flatten(2).transpose(1, 2).contiguous() #B C N->B N C

        return x


class ConvRelPosEnc(nn.Module):
    """Convolutional relative position encoding."""
    def __init__(self, Ch, h, window):
        """Initialization.

        Ch: Channels per head.
        h: Number of heads.
        window: Window size(s) in convolutional relative positional encoding.
                It can have two forms:
                1. An integer of window size, which assigns all attention heads
                   with the same window size in ConvRelPosEnc.
                2. A dict mapping window size to #attention head splits
                   (e.g. {window size 1: #attention head split 1, window size
                                      2: #attention head split 2})
                   It will apply different window size to
                   the attention head splits.
        """
        super().__init__()

        if isinstance(window, int):
            # Set the same window size for all attention heads.
            window = {window: h}
            self.window = window
        elif isinstance(window, dict):
            self.window = window
        else:
            raise ValueError()

        self.conv_list = nn.ModuleList()
        self.head_splits = []
        for cur_window, cur_head_split in window.items():
            dilation = 1  # Use dilation=1 at default.
            padding_size = (cur_window + (cur_window - 1) *
                            (dilation - 1)) // 2
            cur_conv = nn.Conv2d(
                cur_head_split * Ch,
                cur_head_split * Ch,
                kernel_size=(cur_window, cur_window),
                padding=(padding_size, padding_size),
                dilation=(dilation, dilation),
                groups=cur_head_split * Ch,
                )
            self.conv_list.append(cur_conv)
            self.head_splits.append(cur_head_split)
        self.channel_splits = [x * Ch for x in self.head_splits]

    def forward(self, q, v, size):
        """foward function"""
        B, h, N, Ch = q.shape
        H, W = size

        # We don't use CLS_TOKEN
        q_img = q
        v_img = v

        # Shape: [B, h, H*W, Ch] -> [B, h*Ch, H, W].
        v_img = rearrange(v_img, "B h (H W) Ch -> B (h Ch) H W", H=H, W=W)
        # Split according to channels.
        v_img_list = torch.split(v_img, self.channel_splits, dim=1)
        conv_v_img_list = [
            conv(x) for conv, x in zip(self.conv_list, v_img_list)
        ]
        conv_v_img = torch.cat(conv_v_img_list, dim=1)
        # Shape: [B, h*Ch, H, W] -> [B, h, H*W, Ch].
        conv_v_img = rearrange(conv_v_img, "B (h Ch) H W -> B h (H W) Ch", h=h)

        EV_hat_img = q_img * conv_v_img
        EV_hat = EV_hat_img
        return EV_hat


class FactorAtt_ConvRelPosEnc(nn.Module):
    """Factorized attention with convolutional relative position encoding class."""

    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        shared_crpe=None,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)  # Note: attn_drop is actually not used.
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # Shared convolutional relative position encoding.
        self.crpe = shared_crpe

    def forward(self, x, size):
        B, N, C = x.shape

        # Generate Q, K, V.
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
            .contiguous()
        )  # Shape: [3, B, h, N, Ch].
        q, k, v = qkv[0], qkv[1], qkv[2]  # Shape: [B, h, N, Ch].

        # Factorized attention.
        k_softmax = k.softmax(dim=2)  # Softmax on dim N.
        k_softmax_T_dot_v = einsum(
            "b h n k, b h n v -> b h k v", k_softmax, v
        )  # Shape: [B, h, Ch, Ch].
        factor_att = einsum(
            "b h n k, b h k v -> b h n v", q, k_softmax_T_dot_v
        )  # Shape: [B, h, N, Ch].

        # Convolutional relative position encoding.
        crpe = self.crpe(q, v, size=size)  # Shape: [B, h, N, Ch].

        # Merge and reshape.
        x = self.scale * factor_att + crpe
        x = (
            x.transpose(1, 2).reshape(B, N, C).contiguous()
        )  # Shape: [B, h, N, Ch] -> [B, N, h, Ch] -> [B, N, C].

        # Output projection.
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class MixFFN_skip(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.fc1 = nn.Linear(c1, c2)
        self.dwconv = DWConv(c2)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(c2, c1)
        self.norm1 = nn.LayerNorm(c2)
        self.norm2 = nn.LayerNorm(c2)
        self.norm3 = nn.LayerNorm(c2)
    def forward(self, x: torch.Tensor, H, W) -> torch.Tensor:
        ax = self.act(self.norm1(self.dwconv(self.fc1(x), H, W)+self.fc1(x)))
        out = self.fc2(ax)
        return out


class MHCABlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=3,
        drop_path=0.0,
        qkv_bias=True,
        qk_scale=None,
        norm_layer= 'LN',
        shared_cpe=None,
        shared_crpe=None,
    ):
        super().__init__()

        self.cpe = shared_cpe
        self.crpe = shared_crpe
        self.factoratt_crpe = FactorAtt_ConvRelPosEnc(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            shared_crpe=shared_crpe,
        )
        self.mlp = MixFFN_skip(dim, dim * mlp_ratio)
        # self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm1 = nn.LayerNorm(dim,eps=1e-6)
        self.norm2 = nn.LayerNorm(dim,eps=1e-6)

    def forward(self, x, size):
        # x.shape = [B, N, C]
        # print('inside the MHCABlock')
        H,W = size
        if self.cpe is not None:
            x = self.cpe(x, size)
        cur = self.norm1(x)
        x = x + self.factoratt_crpe(cur, size)

        cur = self.norm2(x)
        x = x + self.mlp(cur,H,W)
        return x


class MHCAEncoder(nn.Module):
    def __init__(
        self,
        dim,
        num_layers=1,
        num_heads=8,
        mlp_ratio=3,
        drop_path_list=[],
        qk_scale=None,
        crpe_window={3: 2, 5: 3, 7: 3},
    ):
        super().__init__()

        self.num_layers = num_layers
        self.cpe = ConvPosEnc(dim, k=3)
        self.crpe = ConvRelPosEnc(Ch=dim // num_heads, h=num_heads, window=crpe_window)
        self.MHCA_layers = nn.ModuleList(
            [
                MHCABlock(
                    dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    drop_path=drop_path_list[idx],
                    qk_scale=qk_scale,
                    shared_cpe=self.cpe,
                    shared_crpe=self.crpe,
                )
                for idx in range(self.num_layers)
            ]
        )

    def forward(self, x, size):
        H, W = size
        # print('inside the MHCAEncoder')
        B = x.shape[0]
        # x' shape : [B, N, C]
        for layer in self.MHCA_layers:
            x = layer(x, (H, W))
            # print("---MHCAEncoder---")

        # return x's shape : [B, N, C] -> [B, C, H, W]
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        return x


class ResBlock(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.Hardswish,
        norm_cfg="BN",
    ):
        super().__init__()

        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.conv1 = Conv2d_BN(
            in_features, hidden_features, act_layer=act_layer, norm_cfg=norm_cfg
        )
        self.dwconv = nn.Conv2d(
            hidden_features,
            hidden_features,
            3,
            1,
            1,
            bias=False,
            groups=hidden_features,
        )
        # self.norm = norm_layer(hidden_features)
        self.norm = nn.BatchNorm2d(hidden_features)
        self.act = act_layer()
        self.conv2 = Conv2d_BN(hidden_features, out_features, norm_cfg=norm_cfg)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
            # fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            # fan_out //= m.groups
            # m.weight.data.normal_(0, torch.sqrt(2.0 / fan_out))
            # if m.bias is not None:
            #     m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()


    def forward(self, x):
        identity = x
        feat = self.conv1(x)
        feat = self.dwconv(feat)
        feat = self.norm(feat)
        feat = self.act(feat)
        feat = self.conv2(feat)

        return identity + feat



class MHCA_stage(nn.Module):
    def __init__(
        self,
        embed_dim,
        out_embed_dim,
        num_layers=1,
        num_heads=8,
        mlp_ratio=3,
        num_path=4,
        norm_cfg="BN",
        drop_path_list=[],
        concat='normal'
    ):
        super().__init__()
        self.concat=concat
        self.mhca_blks = nn.ModuleList(
            [
                MHCAEncoder(
                    embed_dim,
                    num_layers,
                    num_heads,
                    mlp_ratio,
                    drop_path_list=drop_path_list,
                )
                for _ in range(num_path)
            ]
        )

        self.InvRes = ResBlock(
            in_features=embed_dim, out_features=embed_dim, norm_cfg=norm_cfg
        )
        if self.concat == 'normal':
            self.aggregate = Conv2d_BN(
                embed_dim * (num_path + 1),
                out_embed_dim,
                act_layer=nn.Hardswish,
                norm_cfg=norm_cfg,
            )
        elif self.concat == '3d':
            self.aggregate = Conv3d_BN_concat(
                embed_dim,
                out_embed_dim
            )
        elif self.concat == 'se':
            self.aggregate = SE_Block(in_ch = embed_dim*4, out_ch= out_embed_dim, r=16)
        else:
            self.aggregate = Conv3d_BN_channel_attention_concat(
                embed_dim,
                out_embed_dim,
                cam=self.concat# cam or cam_fact
            )


    def forward(self, inputs):
        # print(len(inputs))
        # print("---Res---")
        att_outputs = [self.InvRes(inputs[0])]
        # print(f'Resblock: {self.InvRes(inputs[0]).shape}')
        # 
        for x, encoder in zip(inputs, self.mhca_blks):
            # [B, C, H, W] -> [B, N, C]
            _, _, H, W = x.shape
            # print(f'H:{H} W:{W}')
            x = x.flatten(2).transpose(1, 2).contiguous()
            # print('going to the encoder')
            tmp = encoder(x, size=(H, W))
            # print(f'\n---attention path{tmp.shape}--')
            att_outputs.append(tmp)

        # out_concat = torch.cat(att_outputs, dim=1)
        # print(f'before cat: {att_outputs[0].shape}')
        # print(f'before cat: {att_outputs[1].shape}')
        # print(f'before cat: {att_outputs[2].shape}')
        if self.concat == 'normal' or 'se':
            out = self.aggregate( torch.cat(att_outputs, dim=1))
        else:
            out = self.aggregate(att_outputs)

        # print(f'\n after cat: {out.shape}')

        return out

#Jia: change the num_stages here because the stages are 4 originally
def dpr_generator(drop_path_rate, num_layers, num_stages):
    """
    Generate drop path rate list following linear decay rule
    """
    dpr_list = [x.item() for x in torch.linspace(0, drop_path_rate, sum(num_layers))]
    dpr = []
    cur = 0
    for i in range(num_stages):
        dpr_per_stage = dpr_list[cur : cur + num_layers[i]]
        dpr.append(dpr_per_stage)
        cur += num_layers[i]

    return dpr

class MSViT(nn.Module):
    def __init__(self, image_size, in_dim, key_dim, value_dim, layers, head_count=1, dil_conv=1, token_mlp='mix_skip',MSViT_config=1, concat='normal'):
        super().__init__()

        self.Hs=[56, 28, 14, 7]
        self.Ws=[56, 28, 14, 7]
        patch_sizes = [7, 3, 3, 3]
        strides = [4, 2, 2, 2]
        padding_sizes = [3, 1, 1, 1]
        # dil_conv = False #no dilation this version...
        if dil_conv:  
            dilation = 2 
            patch_sizes1 = [7, 5, 5, 5]
            patch_sizes2 = [0, 3, 3, 3]
            patch_sizes3 = [0, 1, 1, 1]
            dil_padding_sizes1 = [3, 0, 0, 0]    
            dil_padding_sizes2 = [0, 0, 0, 0]
            dil_padding_sizes3 = [0, 0, 0, 0]
            
        else:
            dilation = 1
            patch_sizes1 = [7, 3, 3, 3]
            patch_sizes2 = [5, 1, 1, 1]
            patch_sizes3 = [0, 5, 5, 5]
            dil_padding_sizes1 = [3, 1, 1, 1]
            dil_padding_sizes2 = [1, 0, 0, 0]
            dil_padding_sizes3 = [1, 2, 2, 2]


        # 1 by 1 convolution to alter the dimension
        self.conv1_1_s1 = nn.Conv2d(3*in_dim[0], in_dim[0], 1)
        self.conv1_1_s2 = nn.Conv2d(3*in_dim[1], in_dim[1], 1)
        self.conv1_1_s3 = nn.Conv2d(3*in_dim[2], in_dim[2], 1)
        self.conv1_1_s4 = nn.Conv2d(3*in_dim[3], in_dim[3], 1)

        # -------------MSViT codes---------------------------------------------------
        # Patch embeddings.
        if MSViT_config == 1:
            num_path = [3,3,3]
            # num_layers = [3,6,3]
            num_layers = [3,8,3]
            num_heads = [8,8,8]
        elif MSViT_config == 2:
            num_path = [3,3,3]
            num_layers = [3,8,3]
            num_heads = [8,8,8]
        # elif MSViT_config == 3:
        #     num_path = [3,3,3]
        #     num_layers = [3,8,3]
        #     num_heads = [4,4,4]
        #     # print(f"MSViT_config: {MSViT_config}")
        else: #config==2
            num_path = [3,3,3]
            num_layers = [3,8,3]
            num_heads = [8,8,8]


        # num_heads = [8,8,8]
        # print(f"num_path {num_path}")
        # print(f"\n num_layers {num_layers}")
        # print(f"\n num_heads {num_heads}")
        mlp_ratios = [4,4,4] # what is mlp_ratios?
        num_stages = 3
        drop_path_rate=0.0
        dpr = dpr_generator(drop_path_rate, num_layers, num_stages)
     
        self.patch_embed_stage2 = Patch_Embed_stage(
                    in_dim[0],
                    num_path=num_path[0],
                    isPool=True,
                    norm_cfg='BN',
                )

        self.patch_embed_stage3 = Patch_Embed_stage(
                    in_dim[1],
                    num_path=num_path[1],
                    isPool=True, 
                    norm_cfg='BN',
                )
        # if idx == 0 else True
        self.patch_embed_stage4 = Patch_Embed_stage(
                    in_dim[2],
                    num_path=num_path[2],
                    isPool=True,
                    norm_cfg='BN',
                )


        # Multi-Head Convolutional Self-Attention (MHCA)
        self.mhca_stage2 = MHCA_stage(
                    in_dim[0],
                    in_dim[1],
                    num_layers[0],
                    num_heads[0],
                    mlp_ratios[0],
                    num_path[0],
                    norm_cfg='BN',
                    drop_path_list=dpr[0],
                    concat=concat
                )

        self.mhca_stage3 = MHCA_stage(
                    in_dim[1],
                    in_dim[2],
                    num_layers[1],
                    num_heads[1],
                    mlp_ratios[1],
                    num_path[1],
                    norm_cfg='BN',
                    drop_path_list=dpr[1],
                    concat=concat
                )

        self.mhca_stage4 = MHCA_stage(
                    in_dim[2],
                    in_dim[3],
                    num_layers[2],
                    num_heads[2],
                    mlp_ratios[2],
                    num_path[2],
                    norm_cfg='BN',
                    drop_path_list=dpr[2],
                    concat=concat
                )

        # patch_embed
        # layers = [2, 2, 2, 2] dims = [64, 128, 320, 512]
        self.patch_embed1 = OverlapPatchEmbeddings(image_size, patch_sizes[0], strides[0], padding_sizes[0], 3, in_dim[0])
        
       
        # # transformer encoder
        self.cpe = ConvPosEnc(in_dim[0], k=3)
        self.block1 = nn.ModuleList([ 
            EfficientTransformerBlock(in_dim[0], key_dim[0], value_dim[0], head_count, token_mlp)
        for _ in range(layers[0])])
        self.norm1 = nn.LayerNorm(in_dim[0])

        
        
    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """

        def _init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                # trunc_normal_(m.weight, std=0.02)
                # if isinstance(m, nn.Linear) and m.bias is not None:
                #     nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
                # nn.init.constant_(m.bias, 0)
                # nn.init.constant_(m.weight, 1.0)
            elif isinstance(m, nn.Conv2d):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

       
            
        if pretrained is None:
            self.apply(_init_weights)
        else:
            raise TypeError("pretrained must be a str or None")


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        outs = []
        # stage conv stem
        # stage 1
        x, H, W = self.patch_embed1(x)
        # print(f'stage1 after embedding: {x.shape}')
        x = self.cpe(x, (H,W))
        # print(f'stage1 after cpe: {x.shape}')
        for blk in self.block1:
            x = blk(x, H, W)
        x = self.norm1(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

      

        # # stage 2
        # print("-------EN: Stage 2------\n\n")
        att_input = self.patch_embed_stage2(x)
        x = self.mhca_stage2(att_input)
        outs.append(x)

        # # stage 3
        # print("-------EN: Stage 3------\n\n")
        att_input = self.patch_embed_stage3(x)
        x = self.mhca_stage3(att_input)
        outs.append(x)

        # # stage 4
        # print("-------EN: Stage 4------\n\n")
        att_input = self.patch_embed_stage4(x)
        x = self.mhca_stage4(att_input)
        outs.append(x)

        return outs
    
class MSViT_4Stages(nn.Module):
    def __init__(self, image_size, in_dim, key_dim, value_dim, layers, head_count=1, dil_conv=1, token_mlp='mix_skip',MSViT_config=1, concat='normal'):
        super().__init__()

        self.Hs=[56, 28, 14, 7]
        self.Ws=[56, 28, 14, 7]
        patch_sizes = [7, 3, 3, 3]
        strides = [4, 2, 2, 2]
        padding_sizes = [3, 1, 1, 1]
        # dil_conv = False #no dilation this version...
        if dil_conv:  
            dilation = 2 
            patch_sizes1 = [7, 5, 5, 5]
            patch_sizes2 = [0, 3, 3, 3]
            patch_sizes3 = [0, 1, 1, 1]
            dil_padding_sizes1 = [3, 0, 0, 0]    
            dil_padding_sizes2 = [0, 0, 0, 0]
            dil_padding_sizes3 = [0, 0, 0, 0]
            
        else:
            dilation = 1
            patch_sizes1 = [7, 3, 3, 3]
            patch_sizes2 = [5, 1, 1, 1]
            patch_sizes3 = [0, 5, 5, 5]
            dil_padding_sizes1 = [3, 1, 1, 1]
            dil_padding_sizes2 = [1, 0, 0, 0]
            dil_padding_sizes3 = [1, 2, 2, 2]


        # 1 by 1 convolution to alter the dimension
        self.conv1_1_s1 = nn.Conv2d(3*in_dim[0], in_dim[0], 1)
        self.conv1_1_s2 = nn.Conv2d(3*in_dim[1], in_dim[1], 1)
        self.conv1_1_s3 = nn.Conv2d(3*in_dim[2], in_dim[2], 1)
        self.conv1_1_s4 = nn.Conv2d(3*in_dim[3], in_dim[3], 1)

        # -------------MSViT codes---------------------------------------------------
        
        # Patch embeddings.
        if MSViT_config == 1:
            num_path = [2,3,3,3]
            num_layers = [1, 3,8,3]
            num_heads = [8,8,8,8]
        elif MSViT_config == 2:
            num_path = [2,3,3,3]
            num_layers = [1, 3,8,3]
            num_heads = [8,8,8,8]
        else: #config==2
            num_path = [2,3,3,3]
            num_layers = [1, 3,8,3]
            num_heads = [8,8,8,8]


    
        mlp_ratios = [4, 4, 4, 4] # what is mlp_ratios?
        num_stages = 4
        drop_path_rate=0.0
        dpr = dpr_generator(drop_path_rate, num_layers, num_stages)

        self.stem = nn.Sequential(
            Conv2d_BN(
                3,
                in_dim[0] // 2,
                kernel_size=3,
                stride=2,
                pad=1,
                act_layer=nn.Hardswish,
            ),
            Conv2d_BN(
                in_dim[0] // 2,
                in_dim[0],
                kernel_size=3,
                stride=2,
                pad=1,
                act_layer=nn.Hardswish,
            ),
        )

        self.patch_embed_stage1 = Patch_Embed_stage(
                    in_dim[0],
                    num_path=num_path[0],
                    isPool=False,
                    norm_cfg='BN',
                )
     
        self.patch_embed_stage2 = Patch_Embed_stage(
                    in_dim[0],
                    num_path=num_path[1],
                    isPool=True,
                    norm_cfg='BN',
                )

        self.patch_embed_stage3 = Patch_Embed_stage(
                    in_dim[1],
                    num_path=num_path[2],
                    isPool=True, 
                    norm_cfg='BN',
                )
        # if idx == 0 else True
        self.patch_embed_stage4 = Patch_Embed_stage(
                    in_dim[2],
                    num_path=num_path[3],
                    isPool=True,
                    norm_cfg='BN',
                )


        # Multi-Head Convolutional Self-Attention (MHCA)
        self.mhca_stage1 = MHCA_stage(
                    in_dim[0],
                    in_dim[0],
                    num_layers[0],
                    num_heads[0],
                    mlp_ratios[0],
                    num_path[0],
                    norm_cfg='BN',
                    drop_path_list=dpr[0],
                    concat=concat
                )

        self.mhca_stage2 = MHCA_stage(
                    in_dim[0],
                    in_dim[1],
                    num_layers[1],
                    num_heads[1],
                    mlp_ratios[1],
                    num_path[1],
                    norm_cfg='BN',
                    drop_path_list=dpr[0],
                    concat=concat
                )

        self.mhca_stage3 = MHCA_stage(
                    in_dim[1],
                    in_dim[2],
                    num_layers[2],
                    num_heads[2],
                    mlp_ratios[2],
                    num_path[2],
                    norm_cfg='BN',
                    drop_path_list=dpr[1],
                    concat=concat
                )

        self.mhca_stage4 = MHCA_stage(
                    in_dim[2],
                    in_dim[3],
                    num_layers[3],
                    num_heads[3],
                    mlp_ratios[3],
                    num_path[3],
                    norm_cfg='BN',
                    drop_path_list=dpr[2],
                    concat=concat
                )

        # patch_embed
        # layers = [2, 2, 2, 2] dims = [64, 128, 320, 512]
        # self.patch_embed1 = OverlapPatchEmbeddings(image_size, patch_sizes[0], strides[0], padding_sizes[0], 3, in_dim[0])
        
       
        # # transformer encoder
        self.cpe = ConvPosEnc(in_dim[0], k=3)
        self.block1 = nn.ModuleList([ 
            EfficientTransformerBlock(in_dim[0], key_dim[0], value_dim[0], head_count, token_mlp)
        for _ in range(layers[0])])
        self.norm1 = nn.LayerNorm(in_dim[0])

        
        
    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """

        def _init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                # trunc_normal_(m.weight, std=0.02)
                # if isinstance(m, nn.Linear) and m.bias is not None:
                #     nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
                # nn.init.constant_(m.bias, 0)
                # nn.init.constant_(m.weight, 1.0)
            elif isinstance(m, nn.Conv2d):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

       
            
        if pretrained is None:
            self.apply(_init_weights)
        else:
            raise TypeError("pretrained must be a str or None")


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        outs = []
        # stage conv stem
        # stage 1
        x, H, W = self.patch_embed1(x)
        # print(f'stage1 after embedding: {x.shape}')
        x = self.cpe(x, (H,W))
        # print(f'stage1 after cpe: {x.shape}')
        for blk in self.block1:
            x = blk(x, H, W)
        x = self.norm1(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

      

        # # stage 2
        # print("-------EN: Stage 2------\n\n")
        att_input = self.patch_embed_stage2(x)
        x = self.mhca_stage2(att_input)
        outs.append(x)

        # # stage 3
        # print("-------EN: Stage 3------\n\n")
        att_input = self.patch_embed_stage3(x)
        x = self.mhca_stage3(att_input)
        outs.append(x)

        # # stage 4
        # print("-------EN: Stage 4------\n\n")
        att_input = self.patch_embed_stage4(x)
        x = self.mhca_stage4(att_input)
        outs.append(x)

        return outs
  

class MSTransception(nn.Module):
    def __init__(self, num_classes=9, head_count=1, dil_conv=1, token_mlp_mode="mix_skip", MSViT_config=1, concat='normal'):#, inception="135"
        super().__init__()
    
        # Encoder
        dims, key_dim, value_dim, layers = [[64, 128, 320, 512], [64, 128, 320, 512], [64, 128, 320, 512], [2, 2, 2, 2]]        
        self.backbone = MSViT_4Stages(image_size=224, in_dim=dims, key_dim=key_dim, value_dim=value_dim, layers=layers,
                            head_count=head_count, dil_conv=dil_conv, token_mlp=token_mlp_mode, MSViT_config=MSViT_config, concat=concat)
        # self.backbone = MiT_3inception_padding(image_size=224, in_dim=dims, key_dim=key_dim, value_dim=value_dim, layers=layers,
        #                     head_count=head_count, dil_conv=dil_conv, token_mlp=token_mlp_mode)

        # Here options:(1) MiT_3inception->3 stages;(2) MiT->4 stages; 
        # (3)MiT_3inception_padding: padding before transformer after patch embedding (follow depthconcat)
        # (4)MiT_3inception_3branches
        # Decoder
        d_base_feat_size = 7 #16 for 512 input size, and 7 for 224
        in_out_chan = [[32, 64, 64, 64],[144, 128, 128, 128],[288, 320, 320, 320],[512, 512, 512, 512]]  # [dim, out_dim, key_dim, value_dim]

        self.decoder_3 = MyDecoderLayer((d_base_feat_size, d_base_feat_size), in_out_chan[3], head_count, 
                                        token_mlp_mode, n_class=num_classes)
        self.decoder_2 = MyDecoderLayer((d_base_feat_size*2, d_base_feat_size*2), in_out_chan[2], head_count,
                                        token_mlp_mode, n_class=num_classes)
        self.decoder_1 = MyDecoderLayer((d_base_feat_size*4, d_base_feat_size*4), in_out_chan[1], head_count, 
                                        token_mlp_mode, n_class=num_classes) 
        self.decoder_0 = MyDecoderLayer((d_base_feat_size*8, d_base_feat_size*8), in_out_chan[0], head_count,
                                        token_mlp_mode, n_class=num_classes, is_last=True)

        
    def forward(self, x):
        #---------------Encoder-------------------------
        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1)

        output_enc = self.backbone(x)
        # return output_enc

        b,c,_,_ = output_enc[3].shape

        #---------------Decoder-------------------------     
        tmp_3 = self.decoder_3(output_enc[3].permute(0,2,3,1).reshape(b,-1,c))
        tmp_2 = self.decoder_2(tmp_3, output_enc[2].permute(0,2,3,1))
        tmp_1 = self.decoder_1(tmp_2, output_enc[1].permute(0,2,3,1))
        tmp_0 = self.decoder_0(tmp_1, output_enc[0].permute(0,2,3,1))

        return tmp_0
    

if __name__ == "__main__":
    #call Transception_res
    # normal 3d cam cam_fact se
    model = MSTransception(num_classes=9, head_count=8, dil_conv = 1, token_mlp_mode="mix_skip",MSViT_config=2, concat='se')
    tmp_0 = model(torch.rand(1, 3, 224, 224))
    
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model = model.to(device)
    # summary(model, (1, 3,224,224) )
    # stat(model.to('cpu'), (3, 224, 224))

    print(tmp_0.shape)
  