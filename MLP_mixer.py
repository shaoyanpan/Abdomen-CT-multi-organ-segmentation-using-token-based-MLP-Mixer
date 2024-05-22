

from typing import Tuple, Union
import torch
import torch.nn as nn
from einops import rearrange
from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks import UnetrBasicBlock, UnetrPrUpBlock, UnetUpBlock,UnetResBlock,TransformerBlock
from monai.networks.nets.swin_unetr import SwinTransformerBlock,get_window_size,compute_mask
from monai.networks.nets import ViT
from timm.models.layers import DropPath, to_3tuple, trunc_normal_
import numpy as np
from functools import partial
from einops.layers.torch import Rearrange, Reduce    


    
class FilterBasedTokenizer(nn.Module):
    def __init__(self, feature_map_cs: int, visual_tokens_cs: int = 1024, n_visual_tokens: int = 8) -> None:
        super().__init__()
        self.feature_map_cs, self.visual_tokens_cs, self.n_visual_tokens = feature_map_cs, visual_tokens_cs, n_visual_tokens
        self.W_a = nn.Conv1d(feature_map_cs, n_visual_tokens, kernel_size=1, bias=False)
        self.W_v = nn.Conv1d(feature_map_cs, visual_tokens_cs, kernel_size=1, bias=False)
        self.softmax = nn.Softmax(dim=2)
        self.cls_token = nn.Parameter(torch.zeros(1, visual_tokens_cs, 8))
        self.teacher_token = nn.Parameter(torch.zeros(1, visual_tokens_cs, 8))
        self.af1 = nn.LeakyReLU()
        self.pos_embedding = nn.Parameter(torch.zeros(1, visual_tokens_cs,(n_visual_tokens + 16)))
        self.norm = nn.LayerNorm(visual_tokens_cs)
        
    def forward(self, X: torch.Tensor):
        # X.shape = bs, feature_map_cs, HW
        X = torch.flatten(X, start_dim=2)
        A = self.W_a(X) # bs, n_visual_tokens, HW
        V = self.W_v(X)  # bs, visual_tokens_cs, HW
        attention_weights = self.softmax(A).permute(0, 2, 1) # bs, HW, n_visual_tokens
        out = self.norm(torch.bmm(V, attention_weights).permute(0,2,1)).permute(0,2,1)
        return out 


class RecurrentTokenizer(nn.Module):
    def __init__(self, feature_map_cs: int, visual_tokens_cs: int = 1024, n_visual_tokens: int = 8):
        super().__init__()
        self.feature_map_cs, self.visual_tokens_cs = feature_map_cs, visual_tokens_cs
        self.W_tr = nn.Conv1d(visual_tokens_cs, feature_map_cs, kernel_size=1, bias=False)
        self.W_v = nn.Conv1d(feature_map_cs, visual_tokens_cs, kernel_size=1, bias=False)
        self.softmax = nn.Softmax(dim=2)
        self.cls_token = nn.Parameter(torch.zeros(1, visual_tokens_cs, 8))
        self.teacher_token = nn.Parameter(torch.zeros(1, visual_tokens_cs, 8))
        self.af1 = nn.LeakyReLU()
        self.pos_embedding = nn.Parameter(torch.zeros(1, visual_tokens_cs,(n_visual_tokens + 16)))
        self.norm = nn.LayerNorm(visual_tokens_cs)

    def forward(self, X: torch.Tensor, T_in: torch.Tensor= None):
        # X.shape = bs, feature_map_cs, HW
        X = torch.flatten(X, start_dim=2)
        W_r = self.W_tr(T_in).permute(0, 2, 1)  # bs, n_visual_tokens, feature_map_cs
        proj = torch.bmm(W_r, X) / np.sqrt(self.feature_map_cs)  # bs, n_visual_tokens, HW
        attention_weights = self.softmax(proj).permute(0, 2, 1)  # bs, HW, n_visual_tokens
        values = self.W_v(X)  # bs, visual_tokens_cs, HW
        out = self.norm(torch.bmm(values, attention_weights).permute(0,2,1)).permute(0,2,1)
        return out 

class Projector(nn.Module):
    def __init__(self, feature_map_cs: int, visual_tokens_cs: int):
        super().__init__()
        self.W_q = nn.Conv1d(feature_map_cs//1, feature_map_cs//1, kernel_size=1, bias=False)
        self.W_k = nn.Conv1d(visual_tokens_cs, feature_map_cs//1, kernel_size=1, bias=False)
        self.W_v = nn.Conv1d(visual_tokens_cs, feature_map_cs//1, kernel_size=1, bias=False)
        self.W_s = nn.Conv1d(feature_map_cs, feature_map_cs//1, kernel_size=1, bias=False)
        self.softmax = nn.Softmax(dim=2)
        self.af1 = nn.LeakyReLU()
        self.norm = nn.LayerNorm(feature_map_cs)

    def forward(self, X_in: torch.Tensor, T: torch.Tensor):
        # X_in.shape = bs, feature_map_cs, HW
        T = T.permute(0, 2, 1)
        X_in = torch.flatten(X_in, start_dim=2)
        pixel_token_similarity = torch.bmm(self.W_q(X_in).permute(0, 2, 1), self.W_k(T))  # bs, HW, n_visual_tokens
        attention_weights = self.softmax(pixel_token_similarity).permute(0, 2, 1)  # bs, n_visual_tokens, HW
        values = self.W_v(T)  # bs, feature_map_cs, n_visual_tokens
        out = X_in+torch.bmm(values, attention_weights)
        out = self.norm(out.permute(0,2,1)).permute(0,2,1)
        return out  # bs, feature_map_cs, HW

class DownsampleBlock(nn.Module):
    def __init__(self,in_dim, out_dim, kernel_size = 7,stride = 2, scale = (0.5,0.5,1)):
        super(DownsampleBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_dim, out_dim, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, bias=False)
        self.bn1 = nn.InstanceNorm3d(out_dim)
        self.activation1 = nn.LeakyReLU()
        # self.activation2 = nn.GELU()
        
    def forward(self, x):

        out = self.conv1(x)
        out = self.activation1(self.bn1(out))
        return out

class UpsampleBlock(nn.Module):
    def __init__(self,in_dim, out_dim, kernel_size = 7, scale = (2,2,1)):
        super(UpsampleBlock, self).__init__()
        self.conv1 = nn.ConvTranspose3d(in_dim, out_dim, kernel_size=2, stride=2, padding=0, bias=False)
        self.bn1 = nn.InstanceNorm3d(out_dim)
        self.activation1 = nn.LeakyReLU()
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.activation1(self.bn1(out))

        return out    
        
    
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class MLPMixerBlock(nn.Module):

    def __init__(self, dim, num_patch, token_dim_factor, channel_dim_factor, dropout = 0.):
        super().__init__()

        self.token_mix = nn.Sequential(
            nn.LayerNorm(dim),
            Rearrange('b n d -> b d n'),
            FeedForward(num_patch, num_patch*token_dim_factor, dropout),
            Rearrange('b d n -> b n d')
        )

        self.channel_mix = nn.Sequential(
            nn.LayerNorm(dim),
            FeedForward(dim, dim*channel_dim_factor, dropout),
        )

    def forward(self, x):
        x = x + self.token_mix(x)
        x = x + self.channel_mix(x)
        return x

class MLPMixer(nn.Module):

    def __init__(self, dim, num_patch, token_dim_factor, channel_dim_factor,depth,dropout=0):
        super().__init__()
        self.mixer_blocks = nn.ModuleList([])
        for _ in range(depth):
            self.mixer_blocks.append(MLPMixerBlock(dim, num_patch, token_dim_factor, channel_dim_factor))

        self.layer_norm = nn.LayerNorm(dim)


    def forward(self, x):
        for mixer_block in self.mixer_blocks:
            x = mixer_block(x)
            x = self.layer_norm(x)
        return x



class MLP_MIXER(nn.Module):
    """
    UNETR based on: "Hatamizadeh et al.,
    UNETR: Transformers for 3D Medical Image Segmentation <https://arxiv.org/abs/2103.10504>"
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        depth: int,
        feature_size: int = 512,
        hidden_size: int = 256,
        mlp_dim: int = 512,
        dropout_rate: float = 0,
    ) -> None:
        """
        Args:
            in_channels: dimension of input channels.
            out_channels: dimension of output channels.
            img_size: dimension of input image.
            feature_size: dimension of network feature size.
            hidden_size: dimension of hidden layer.
            mlp_dim: dimension of feedforward layer.
            num_heads: number of attention heads.
            pos_embed: position embedding layer type.
            norm_name: feature normalization type and arguments.
            conv_block: bool argument to determine if convolutional block is used.
            res_block: bool argument to determine if residual block is used.
            dropout_rate: faction of the input units to drop.
        Examples::
            # for single channel input 4-channel output with patch size of (96,96,96), feature size of 32 and batch norm
            >>> net = UNETR(in_channels=1, out_channels=4, img_size=(96,96,96), feature_size=32, norm_name='batch')
            # for 4-channel input 3-channel output with patch size of (128,128,128), conv position embedding and instance norm
            >>> net = UNETR(in_channels=4, out_channels=3, img_size=(128,128,128), pos_embed='conv', norm_name='instance')
        """

        super().__init__()

        self.num_layers = depth
        self.num_tokens = feature_size
        self.dropout_rate = dropout_rate
        self.hidden_size = hidden_size
        self.mlp_size = mlp_dim

        dpr = [x.item() for x in torch.linspace(0, 0, 5)]

        self.model_encoder2 = MLPMixer(
            dim=self.mlp_size//1,
            num_patch=self.num_tokens//1,
            token_dim_factor=1,
            channel_dim_factor=4,
            depth=self.num_layers,
            dropout = dpr[2],
        )
                
        self.model_encoder3 = MLPMixer(
            dim=self.mlp_size//1,
            num_patch=self.num_tokens//2,
            token_dim_factor=1,
            channel_dim_factor=4,
            depth=self.num_layers,
            dropout = dpr[3],
        )
                        
        self.model_encoder4 = MLPMixer(
            dim=self.mlp_size,
            num_patch=self.num_tokens//4,
            token_dim_factor=1,
            channel_dim_factor=4,
            depth=self.num_layers,
            dropout = dpr[4],
        )
        
        self.model_decoder3 = MLPMixer(
            dim=self.mlp_size//1,
            num_patch=self.num_tokens//2,
            token_dim_factor=1,
            channel_dim_factor=4,
            depth=self.num_layers,
            dropout = dpr[3],
        )
        
        self.model_decoder2 = MLPMixer(
            dim=self.mlp_size//1,
            num_patch=self.num_tokens//1,
            token_dim_factor=1,
            channel_dim_factor=4,
            depth=self.num_layers,
            dropout = dpr[2],
        )
                

        self.conv1 = DownsampleBlock(1,self.hidden_size//16, kernel_size=3, stride=1, scale = (0.5,0.5,0.5))
        self.out = nn.Conv3d(hidden_size//16,out_channels, kernel_size=1, stride=1, padding=0, bias=False)         

        self.down_block3 = DownsampleBlock(self.hidden_size//4,
                                            self.hidden_size//2,kernel_size=3, scale = (0.5,0.5,0.5))
        self.down_block4 = DownsampleBlock(self.hidden_size//2,
                                            self.hidden_size//1,kernel_size=3, scale = (0.5,0.5,0.5))

        self.up_block3 = UpsampleBlock(self.hidden_size//2,
                                            self.hidden_size//4,kernel_size=3, scale = (2,2,2))
        self.up_block4 = UpsampleBlock(self.hidden_size,
                                            self.hidden_size//2,kernel_size=3)

        self.down_block1 = UnetResBlock(spatial_dims = 3,
                                  in_channels=self.hidden_size//16,
                                  out_channels=self.hidden_size//8,
                                  kernel_size=3,
                                  stride = 2,
                                  norm_name = "Instance",
                                  act_name=("leakyrelu", {"inplace": True, "negative_slope": 0.01}))
        self.down_block2 = UnetResBlock(spatial_dims = 3,
                                  in_channels=self.hidden_size//8,
                                  out_channels=self.hidden_size//4,
                                  kernel_size=3,
                                  stride = 2,
                                  norm_name = "Instance",
                                  act_name=("leakyrelu", {"inplace": True, "negative_slope": 0.01}))

        self.up_block2 =   UnetUpBlock(spatial_dims = 3,
                                  in_channels=self.hidden_size//4,
                                  out_channels=self.hidden_size//8,
                                  kernel_size=3,
                                  upsample_kernel_size = 2,
                                  stride = 2,
                                  norm_name = "Instance",
                                  act_name=("leakyrelu", {"inplace": True, "negative_slope": 0.01}))    
        self.up_block1 =   UnetUpBlock(spatial_dims = 3,
                                  in_channels=self.hidden_size//8,
                                  out_channels=self.hidden_size//16,
                                  kernel_size=3,
                                  upsample_kernel_size = 2,
                                  stride = 2,
                                  norm_name = "Instance",
                                  act_name=("leakyrelu", {"inplace": True, "negative_slope": 0.01}))              
        

        self.tokenizer2 = FilterBasedTokenizer(self.hidden_size//4,self.mlp_size,self.num_tokens//1)
        self.tokenizer3 = FilterBasedTokenizer(self.hidden_size//2,self.mlp_size,self.num_tokens//2)
        self.tokenizer4 = FilterBasedTokenizer(self.hidden_size//1,self.mlp_size,self.num_tokens//4)
        
  
        self.tokenizer22 = RecurrentTokenizer(self.hidden_size//4,self.mlp_size,self.num_tokens//2)
        self.tokenizer33 = RecurrentTokenizer(self.hidden_size//2,self.mlp_size,self.num_tokens//1)
        

        self.projector2 = Projector(self.hidden_size//4,self.mlp_size)
        self.projector3 = Projector(self.hidden_size//2,self.mlp_size)
        self.projector4 = Projector(self.hidden_size//1,self.mlp_size)
     
        self.projector22 = Projector(self.hidden_size//4,self.mlp_size)
        self.projector33 = Projector(self.hidden_size//2,self.mlp_size)

        
    def proj_feat(self, x, hidden_size, feat_size):
        x = x.view(x.size(0), feat_size[0], feat_size[1], feat_size[2], hidden_size)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        return x


    def forward(self, x_in):
        
        x = self.conv1(x_in)
        
        x_out1 =  self.down_block1(x)  

        x_down2 =  self.down_block2(x_out1)  
        token2 = self.tokenizer2(x_down2).flatten(start_dim=2).permute(0,2,1)
        token2 = self.model_encoder2(token2)

        x_out2 = rearrange(self.projector2(x_down2, token2),
                            'b c (h w l) -> b c h w l', 
                            h=x_down2.shape[2],w=x_down2.shape[3],l=x_down2.shape[4])

        x_down3 =  self.down_block3(x_out2)
        
        token3 = self.tokenizer3(x_down3).permute(0,2,1) 
        token3 = self.model_encoder3(token3)    
        x_out3 = rearrange(self.projector3(x_down3, token3),
                            'b c (h w l) -> b c h w l', 
                            h=x_down3.shape[2],w=x_down3.shape[3],l=x_down3.shape[4])        
        
        x_down4 =  self.down_block4(x_out3)
        
        token4 = self.tokenizer4(x_down4).permute(0,2,1) 
        token4 = self.model_encoder4(token4)         
        x_out4 = rearrange(self.projector4(x_down4, token4),
                            'b c (h w l) -> b c h w l', 
                            h=x_down4.shape[2],w=x_down4.shape[3],l=x_down4.shape[4])
        
        x_down44 =  self.up_block4(x_out4)
        
        token44 = self.tokenizer33(x_down44,token3.permute(0,2,1)).permute(0,2,1) 
        token44 = self.model_decoder3(token44)         
        x_out44 = rearrange(self.projector33(x_down44, token44),
                            'b c (h w l) -> b c h w l', 
                            h=x_down44.shape[2],w=x_down44.shape[3],l=x_down44.shape[4])

        x_down33 =  self.up_block3(x_out44)

        
        token33 = self.tokenizer22(x_down33,token2.permute(0,2,1)).permute(0,2,1) 
        token33 = self.model_decoder2(token33)          
        x_out33 = rearrange(self.projector22(x_down33, token33),
                            'b c (h w l) -> b c h w l', 
                            h=x_down33.shape[2],w=x_down33.shape[3],l=x_down33.shape[4])



        x_out22 =  self.up_block2(x_out33,x_out1)

        x_out11 =  self.up_block1(x_out22,x)
        
        logits = self.out(x_out11)
        return logits