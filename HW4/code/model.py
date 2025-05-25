"""
This model.py is primarily based on the implementation from 
*PromptIR: Prompting for All-in-One Blind Image Restoration*  
(https://github.com/va1shn9v/PromptIR/blob/main/net/model.py).

Building upon this foundation, I have added several modules—such as CBAM and ModelTrainer—
and experimented with modifying certain parameters.
"""

import numbers
import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from skimage.metrics import peak_signal_noise_ratio
from torchvision import models
from torchvision.models import VGG19_Weights
from einops import rearrange


##########################################################################
## PromptIR: Prompting for All-in-One Blind Image Restoration
## Vaishnav Potlapalli, Syed Waqas Zamir, Salman Khan, and Fahad Shahbaz Khan
## https://arxiv.org/abs/2306.13090


## Layer Norm
def to_3d(x):
    """
    Convert tensor from [B, C, H, W] to [B, H*W, C].
    """
    return rearrange(x, "b c h w -> b (h w) c")


def to_4d(x, h, w):
    """
    Convert tensor from [B, H*W, C] back to [B, C, H, W].
    """
    return rearrange(x, "b (h w) c -> b c h w", h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    """
    Normalizes input across the last dimension without using a bias term.
    """

    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        """
        Apply layer normalization without bias.
        """
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    """
    Layer Normalization with learnable scale (weight) and bias.
    Applies normalization over the last dimension.
    """

    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        """
        Apply layer normalization with bias.
        """
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    """
    Wrapper for applying either Bias-Free or With-Bias Layer Normalization.
    """

    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == "BiasFree":
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        """
        Apply layer normalization on 4D tensor.

        Returns:
            Tensor: Normalized tensor with same shape.
        """
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    """
    Gated-Dconv Feed-Forward Network (GDFN)
    """

    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(
            hidden_features * 2,
            hidden_features * 2,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=hidden_features * 2,
            bias=bias,
        )

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        """
        Forward pass through the Gated-Dconv Feed-Forward Network.
        """
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    """
    A multi-head attention mechanism applied to 2D feature maps.
    This implementation uses depthwise convolution to compute attention, and incorporates temperature scaling.
    """

    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            dim * 3,
            dim * 3,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=dim * 3,
            bias=bias,
        )
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        """
        Forward pass through the attention module.

        Returns:
            Tensor: Output tensor after applying the attention mechanism with shape [B, C, H, W].
        """
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        k = rearrange(k, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        v = rearrange(v, "b (head c) h w -> b head c (h w)", head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = attn @ v

        out = rearrange(
            out, "b head c (h w) -> b (head c) h w", head=self.num_heads, h=h, w=w
        )

        out = self.project_out(out)
        return out


class resblock(nn.Module):
    """
    A residual block for convolutional neural networks.
    It applies two convolutional layers with a skip connection that adds
    the input to the output (residual connection) to help mitigate vanishing gradients.
    """

    def __init__(self, dim):

        super(resblock, self).__init__()
        # self.norm = LayerNorm(dim, LayerNorm_type='BiasFree')

        self.body = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PReLU(),
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False),
        )

    def forward(self, x):
        """
        Forward pass for the residual block.
        """
        res = self.body((x))
        res += x
        return res


##########################################################################
## Resizing modules
class Downsample(nn.Module):
    """
    Downsampling block, which reduces the feature map resolution.
    """

    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(
                n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False
            ),
            nn.PixelUnshuffle(2),
        )

    def forward(self, x):
        """
        Forward pass for downsampling.
        """
        return self.body(x)


class Upsample(nn.Module):
    """
    Upsampling block, which increases the feature map resolution.
    """

    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(
                n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False
            ),
            nn.PixelShuffle(2),
        )

    def forward(self, x):
        """
        Forward pass for upsampling.
        """
        return self.body(x)


##########################################################################
class TransformerBlock(nn.Module):
    """
    Transformer Block
    """

    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        """
        Forward pass for TransformerBlock.
        """
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x


##########################################################################
class OverlapPatchEmbed(nn.Module):
    """
    Overlapped image patch embedding with 3x3 Conv
    """

    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(
            in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias
        )

    def forward(self, x):
        """
        Forward pass for OverlapPatchEmbed.
        """
        x = self.proj(x)

        return x


##########################################################################
##---------- Prompt Gen Module -----------------------
class PromptGenBlock(nn.Module):
    """
    A module that generates prompts conditioned on the input feature map.
    """

    def __init__(self, prompt_dim=128, prompt_len=5, prompt_size=96, lin_dim=192):
        super(PromptGenBlock, self).__init__()
        self.prompt_param = nn.Parameter(
            torch.rand(1, prompt_len, prompt_dim, prompt_size, prompt_size)
        )
        self.linear_layer = nn.Linear(lin_dim, prompt_len)
        self.conv3x3 = nn.Conv2d(
            prompt_dim, prompt_dim, kernel_size=3, stride=1, padding=1, bias=False
        )

    def forward(self, x):
        """
        Generate prompt based on the input feature map.
        """
        B, C, H, W = x.shape
        emb = x.mean(dim=(-2, -1))
        prompt_weights = F.softmax(self.linear_layer(emb), dim=1)
        prompt = prompt_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(
            -1
        ) * self.prompt_param.unsqueeze(0).repeat(B, 1, 1, 1, 1, 1).squeeze(1)
        prompt = torch.sum(prompt, dim=1)
        prompt = F.interpolate(prompt, (H, W), mode="bilinear")
        prompt = self.conv3x3(prompt)

        return prompt


##########################################################################


class ChannelAttention(nn.Module):
    """
    Channel Attention Module (CA) enhances important feature channels using
    both average-pooling and max-pooling, followed by a shared MLP.
    ref: https://zhuanlan.zhihu.com/p/99261200

    Args:
        in_planes (int): Number of input channels.
        ratio (int): Reduction ratio for the MLP. Default is 16.

    Forward Pass:
        x (Tensor): Input feature map of shape (B, C, H, W).

    Returns:
        Tensor: Attention-weighted feature map of the same shape as input.
    """

    def __init__(self, in_planes, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Forward pass of the Channel Attention Module.
        """
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(avg_out))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    """
    Spatial Attention Module (SA) enhances important spatial regions
    by computing attention using max-pooling and average-pooling along
    the channel axis.
    ref: https://zhuanlan.zhihu.com/p/99261200

    Args:
        kernel_size (int): Kernel size for the convolutional layer. Default is 7.

    Forward Pass:
        x (Tensor): Input feature map of shape (B, C, H, W).

    Returns:
        Tensor: Spatial attention map of shape (B, 1, H, W).
    """

    def __init__(self, kernel_size=7):
        super().__init__()

        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Forward pass of the Spatial Attention Module.
        """
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module (CBAM) applies both channel and spatial attention
    sequentially to refine feature representations.

    Args:
        in_planes (int): Number of input channels.
        ratio (int): Reduction ratio for channel attention. Default is 16.
        kernel_size (int): Kernel size for spatial attention. Default is 7.

    Forward Pass:
        x (Tensor): Input feature map of shape (B, C, H, W).

    Returns:
        Tensor: Attention-refined feature map of shape (B, C, H, W).
    """

    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super().__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        """
        Forward pass of the Convolutional Block Attention Module.
        """
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x


##---------- PromptIR -----------------------


class PromptIR(nn.Module):
    """
    PromptIR model for image restoration tasks.
    """

    def __init__(
        self,
        inp_channels=3,
        out_channels=3,
        dim=48,
        num_blocks=[6, 8, 8, 10],
        num_refinement_blocks=6,
        heads=[1, 2, 4, 8],
        ffn_expansion_factor=2.66,
        bias=False,
        LayerNorm_type="WithBias",  ## Other option 'BiasFree'
        decoder=False,
    ):

        super(PromptIR, self).__init__()

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.decoder = decoder

        if self.decoder:
            self.prompt1 = PromptGenBlock(
                prompt_dim=64, prompt_len=10, prompt_size=64, lin_dim=int(dim * 2**1)
            )
            self.prompt2 = PromptGenBlock(
                prompt_dim=128, prompt_len=10, prompt_size=32, lin_dim=int(dim * 2**2)
            )
            self.prompt3 = PromptGenBlock(
                prompt_dim=320, prompt_len=10, prompt_size=16, lin_dim=int(dim * 2**3)
            )

        self.chnl_reduce1 = nn.Conv2d(64, 64, kernel_size=1, bias=bias)
        self.chnl_reduce2 = nn.Conv2d(128, 128, kernel_size=1, bias=bias)
        self.chnl_reduce3 = nn.Conv2d(320, 256, kernel_size=1, bias=bias)

        self.reduce_noise_channel_1 = nn.Conv2d(dim + 64, dim, kernel_size=1, bias=bias)
        self.encoder_level1 = nn.Sequential(
            *[
                TransformerBlock(
                    dim=dim,
                    num_heads=heads[0],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                )
                for i in range(num_blocks[0])
            ]
        )

        self.down1_2 = Downsample(dim)  ## From Level 1 to Level 2

        self.reduce_noise_channel_2 = nn.Conv2d(
            int(dim * 2**1) + 128, int(dim * 2**1), kernel_size=1, bias=bias
        )
        self.encoder_level2 = nn.Sequential(
            *[
                TransformerBlock(
                    dim=int(dim * 2**1),
                    num_heads=heads[1],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                )
                for i in range(num_blocks[1])
            ]
        )

        self.down2_3 = Downsample(int(dim * 2**1))  ## From Level 2 to Level 3

        self.reduce_noise_channel_3 = nn.Conv2d(
            int(dim * 2**2) + 256, int(dim * 2**2), kernel_size=1, bias=bias
        )
        self.encoder_level3 = nn.Sequential(
            *[
                TransformerBlock(
                    dim=int(dim * 2**2),
                    num_heads=heads[2],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                )
                for i in range(num_blocks[2])
            ]
        )

        self.down3_4 = Downsample(int(dim * 2**2))  ## From Level 3 to Level 4
        self.latent = nn.Sequential(
            *[
                TransformerBlock(
                    dim=int(dim * 2**3),
                    num_heads=heads[3],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                )
                for i in range(num_blocks[3])
            ]
        )

        self.up4_3 = Upsample(int(dim * 2**2))  ## From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Conv2d(
            int(dim * 2**2) * 3 // 2, int(dim * 2**2), kernel_size=1, bias=bias
        )
        self.cbam3 = CBAM(int(dim * 2**2) * 3 // 2, ratio=16, kernel_size=7)
        self.noise_level3 = TransformerBlock(
            dim=int(dim * 2**3) + 320,
            num_heads=heads[2],
            ffn_expansion_factor=ffn_expansion_factor,
            bias=bias,
            LayerNorm_type=LayerNorm_type,
        )
        self.reduce_noise_level3 = nn.Conv2d(
            int(dim * 2**3) + 320, int(dim * 2**2), kernel_size=1, bias=bias
        )

        self.decoder_level3 = nn.Sequential(
            *[
                TransformerBlock(
                    dim=int(dim * 2**2),
                    num_heads=heads[2],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                )
                for i in range(num_blocks[2])
            ]
        )

        self.up3_2 = Upsample(int(dim * 2**2))  ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(
            int(dim * 2**2), int(dim * 2**1), kernel_size=1, bias=bias
        )
        self.cbam2 = CBAM(int(dim * 2**2), ratio=16, kernel_size=7)
        self.noise_level2 = TransformerBlock(
            dim=int(dim * 2**2) + 128,
            num_heads=heads[2],
            ffn_expansion_factor=ffn_expansion_factor,
            bias=bias,
            LayerNorm_type=LayerNorm_type,
        )
        self.reduce_noise_level2 = nn.Conv2d(
            int(dim * 2**2) + 128, int(dim * 2**2), kernel_size=1, bias=bias
        )

        self.decoder_level2 = nn.Sequential(
            *[
                TransformerBlock(
                    dim=int(dim * 2**1),
                    num_heads=heads[1],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                )
                for i in range(num_blocks[1])
            ]
        )

        self.up2_1 = Upsample(
            int(dim * 2**1)
        )  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)
        self.noise_level1 = TransformerBlock(
            dim=int(dim * 2**1) + 64,
            num_heads=heads[2],
            ffn_expansion_factor=ffn_expansion_factor,
            bias=bias,
            LayerNorm_type=LayerNorm_type,
        )
        self.reduce_noise_level1 = nn.Conv2d(
            int(dim * 2**1) + 64, int(dim * 2**1), kernel_size=1, bias=bias
        )
        self.cbam1 = CBAM(int(dim * 2**1), ratio=16, kernel_size=7)

        self.decoder_level1 = nn.Sequential(
            *[
                TransformerBlock(
                    dim=int(dim * 2**1),
                    num_heads=heads[0],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                )
                for i in range(num_blocks[0])
            ]
        )
        self.refinement = nn.Sequential(
            *[
                TransformerBlock(
                    dim=int(dim * 2**1),
                    num_heads=heads[0],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                )
                for i in range(num_refinement_blocks)
            ]
        )

        self.output = nn.Conv2d(
            int(dim * 2**1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias
        )

    def forward(self, inp_img, noise_emb=None):
        """
        Forward pass of the PromptIR model.
        """
        inp_enc_level1 = self.patch_embed(inp_img)

        out_enc_level1 = self.encoder_level1(inp_enc_level1)

        inp_enc_level2 = self.down1_2(out_enc_level1)

        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)

        out_enc_level3 = self.encoder_level3(inp_enc_level3)

        inp_enc_level4 = self.down3_4(out_enc_level3)
        latent = self.latent(inp_enc_level4)
        if self.decoder:
            dec3_param = self.prompt3(latent)

            latent = torch.cat([latent, dec3_param], 1)
            latent = self.noise_level3(latent)
            latent = self.reduce_noise_level3(latent)

        inp_dec_level3 = self.up4_3(latent)

        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.cbam3(inp_dec_level3)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)

        out_dec_level3 = self.decoder_level3(inp_dec_level3)
        if self.decoder:
            dec2_param = self.prompt2(out_dec_level3)
            out_dec_level3 = torch.cat([out_dec_level3, dec2_param], 1)
            out_dec_level3 = self.noise_level2(out_dec_level3)
            out_dec_level3 = self.reduce_noise_level2(out_dec_level3)

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.cbam2(inp_dec_level2)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)

        out_dec_level2 = self.decoder_level2(inp_dec_level2)
        if self.decoder:
            dec1_param = self.prompt1(out_dec_level2)
            out_dec_level2 = torch.cat([out_dec_level2, dec1_param], 1)
            out_dec_level2 = self.noise_level1(out_dec_level2)
            out_dec_level2 = self.reduce_noise_level1(out_dec_level2)

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        inp_dec_level1 = self.cbam1(inp_dec_level1)

        out_dec_level1 = self.decoder_level1(inp_dec_level1)

        out_dec_level1 = self.refinement(out_dec_level1)

        out_dec_level1 = self.output(out_dec_level1) + inp_img

        return out_dec_level1


class ModelTrainer:
    """
    This class is responsible for training, evaluating and testing the model.
    """

    def __init__(
        self,
        lr=1e-4,
        min_lr=1e-6,
        weight_decay=1e-4,
        factor=0.1,
        epochs=100,
        args=None,
    ):
        """
        Initializes the model trainer with the specified model name and number of classes.
        """
        self.device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = PromptIR(decoder=True)
        self.model.to(self.device)
        self.vgg = (
            models.vgg19(weights=VGG19_Weights.DEFAULT)
            .features[:16]
            .eval()
            .to(self.device)
        )  # e.g. relu3_1
        for param in self.vgg.parameters():
            param.requires_grad = False

        self.args = args
        self.lr = lr
        self.min_lr = min_lr
        self.weight_decay = weight_decay
        self.factor = factor
        self.epochs = epochs
        self.optim, self.scheduler = self.configure_optimizers()
        self.loss_fn = nn.L1Loss()

    def train_one_epoch(self, data_loader, epoch):
        """
        Trains the model for one epoch.

        Args:
            data_loader (DataLoader): DataLoader for training data.
            epoch (int): Current epoch number.

        Returns:
            float: Average loss for the epoch.
        """
        self.model.train()
        total_loss = 0
        total_l1_loss = 0
        total_perceptual_loss = 0
        pbar = tqdm(data_loader, ncols=120, desc=f"Epoch {epoch}")

        for img, target in pbar:

            img = img.to(self.device)
            target = target.to(self.device)

            self.optim.zero_grad()
            output = self.model(img, target)

            # calculate loss
            l1_loss = self.loss_fn(output, target)
            perceptual_loss = self.perceptual_loss(output, target)
            total_l1_loss += l1_loss.item()
            total_perceptual_loss += perceptual_loss.item()
            losses = l1_loss + 0.01 * perceptual_loss
            total_loss += losses.item()

            losses.backward()
            self.optim.step()

            # get current learning rate and update tqdm
            lr = self.optim.param_groups[0]["lr"]
            pbar.set_postfix(loss=losses.item(), lr=lr)

        return (
            total_loss / len(data_loader),
            total_l1_loss / len(data_loader),
            total_perceptual_loss / len(data_loader),
            lr,
        )

    def eval_one_epoch(self, data_loader, epoch, writer):
        """
        Evaluates the model for one epoch.

        Args:
            data_loader (DataLoader): DataLoader for evaluation data.
            epoch (int): Current epoch number.

        """

        self.model.eval()
        pbar = tqdm(data_loader, ncols=120, desc=f"Epoch {epoch}", unit="batch")
        total_loss = 0
        total_l1_loss = 0
        total_perceptual_loss = 0
        total_psnr = 0
        vis_inputs = []
        vis_targets = []
        vis_outputs = []

        with torch.no_grad():
            for i, (img, target) in enumerate(pbar):

                img = img.to(self.device)
                target = target.to(self.device)

                output = self.model(img)

                # calculate loss
                l1_loss = self.loss_fn(output, target)
                perceptual_loss = self.perceptual_loss(output, target)
                total_l1_loss += l1_loss.item()
                total_perceptual_loss += perceptual_loss.item()
                losses = l1_loss + 0.01 * perceptual_loss
                total_loss += losses.item()

                # calculate psnr
                psnr = self.compute_psnr(output, target)
                total_psnr += psnr
                pbar.set_postfix(loss=losses.item(), psnr=psnr)

                if i < 10:

                    def denorm(x):
                        return x.clamp(0, 1)

                    vis_inputs.append(denorm(img[0].cpu()))
                    vis_targets.append(denorm(target[0].cpu()))
                    vis_outputs.append(denorm(output[0].cpu()))

        # Update scheduler
        self.scheduler.step(total_loss / len(data_loader))

        rows = []
        for in_img, tgt_img, out_img in zip(vis_inputs, vis_targets, vis_outputs):
            row = torch.cat([in_img, tgt_img, out_img], dim=2)
            rows.append(row)

        for i in range(len(rows)):
            writer.add_image(f"Image_{i}", rows[i], epoch)

        return (
            total_loss / len(data_loader),
            total_l1_loss / len(data_loader),
            total_perceptual_loss / len(data_loader),
            total_psnr / len(data_loader),
        )

    def test(self, data_loader, npz_path):
        """
        Test the model on the test dataset and save the predictions to files.
        """
        output_dict = {}
        self.model.eval()
        pbar = tqdm(data_loader, ncols=120, desc="Predicting on data", unit="batch")

        with torch.no_grad():
            for img, img_filename in pbar:
                img = img.to(self.device)
                output = self.model(img)

                for i in range(output.shape[0]):
                    img_np = output[i].detach().cpu().numpy()
                    img_np = np.clip(img_np * 255.0, 0, 255).astype(np.uint8)  # 0-255
                    output_dict[img_filename[i]] = img_np

        # Save the predictions to a .npz file
        np.savez(npz_path, **output_dict)

    def configure_optimizers(self):
        """
        Sets the optimizer and scheduler for the model.

        Returns:
            tuple: Optimizer and scheduler
        """
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            betas=(0.9, 0.96),
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            min_lr=self.min_lr,
            factor=self.factor,
            patience=2,
        )

        return optimizer, scheduler

    def compute_psnr(self, recoverd, clean):
        """
        Compute PSNR between the recovered and clean images.
        ref: https://github.com/va1shn9v/PromptIR/blob/main/utils/val_utils.py
        """
        assert recoverd.shape == clean.shape
        recoverd = np.clip(recoverd.detach().cpu().numpy(), 0, 1)
        clean = np.clip(clean.detach().cpu().numpy(), 0, 1)

        recoverd = recoverd.transpose(0, 2, 3, 1)
        clean = clean.transpose(0, 2, 3, 1)
        psnr = 0

        for i in range(recoverd.shape[0]):
            psnr += peak_signal_noise_ratio(clean[i], recoverd[i], data_range=1)

        return psnr / recoverd.shape[0]

    # normalize
    def normalize(self, img):
        """
        Normalize image tensor using ImageNet mean and std
        """
        mean = torch.tensor([0.485, 0.456, 0.406], device=img.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=img.device).view(1, 3, 1, 1)
        return (img - mean) / std

    def perceptual_loss(self, output, target):
        """
        Compute perceptual loss using VGG features
        """
        output_vgg = self.vgg(self.normalize(output))
        target_vgg = self.vgg(self.normalize(target))
        return F.mse_loss(output_vgg, target_vgg)
