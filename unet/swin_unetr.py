# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# You may not use this file except in compliance with the License.
# Refer to the original MONAI code for full license details:
#     http://www.apache.org/licenses/LICENSE-2.0
# This file is modified from the original Swin UNETR implementation provided by MONAI.
# All modifications in this file are distributed under the same Apache 2.0 License terms.

from __future__ import annotations

import itertools
from collections.abc import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from torch.nn import LayerNorm
from typing_extensions import Final

from monai.networks.blocks import MLPBlock as Mlp
from monai.networks.blocks import (
    PatchEmbed,
    UnetOutBlock,
    UnetrBasicBlock,
    UnetrUpBlock,
)
from monai.networks.layers import DropPath, trunc_normal_
from monai.utils import ensure_tuple_rep, look_up_option, optional_import
from monai.utils.deprecate_utils import deprecated_arg

rearrange, _ = optional_import("einops", name="rearrange")


__all__ = [
    "SwinUNETR",
    "window_partition",
    "window_reverse",
    "WindowAttention",
    "SwinTransformerBlock",
    "PatchMerging",
    "PatchMergingV2",
    "MERGING_MODE",
    "BasicLayer",
    "SwinTransformer",
]


class SwinUNETR(nn.Module):
    """
    Swin UNETR model for semantic segmentation in 2D or 3D images.
    Modified based on MONAI's official implementation (Apache License 2.0).

    Args:
        in_channels (int): number of input channels. Default is 4 (e.g. RGB+texture).
        out_channels (int): number of output segmentation classes. Default is 2.
        embed_dim (int): patch embedding size. Default is 48.
        depths (Sequence[int]): number of layers in each stage. Default is [2, 2, 2, 2].
        window_size (int): window size for self-attention. Default is 16.
        patch_size (int): patch size for initial patch embedding. Default is 4.
        num_heads (Sequence[int]): number of attention heads in each stage. Default is (3, 6, 12, 24).
        norm_name (str): normalization type. Default is "instance".
        drop_rate (float): dropout rate. Default is 0.0.
        attn_drop_rate (float): attention dropout rate. Default is 0.0.
        dropout_path_rate (float): stochastic depth (drop path) rate. Default is 0.0.
        normalize (bool): whether to apply layer normalization to output features. Default is True.
        use_checkpoint (bool): use gradient checkpointing. Default is False.
        spatial_dims (int): spatial dimension, 2 for 2D, 3 for 3D. Default is 2.
        downsample (str): downsample method. Default is "merging".
        use_v2 (bool): whether to use PatchMergingV2. Default is False.
        img_size (int): (Deprecated) input size for checks. Default is 224 (if used).

    Returns:
        A torch.nn.Module instance for SwinUNETR.
    """

    @deprecated_arg(
        name="img_size",
        since="1.3",
        removed="1.5",
        msg_suffix="The img_size argument is not required anymore and checks on the input size are run during forward().",
    )
    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 2,
        embed_dim: int = 48,
        depths: tuple = (2, 2, 2, 2),
        window_size: int = 16,
        patch_size: int = 4,
        num_heads: tuple = (3, 6, 12, 24),
        norm_name: str = "instance",
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        dropout_path_rate: float = 0.0,
        normalize: bool = True,
        use_checkpoint: bool = False,
        spatial_dims: int = 2,
        downsample="merging",
        use_v2=False,
        img_size: int = 1024,  # deprecated usage
    ) -> None:
        super().__init__()

        if spatial_dims not in (2, 3):
            raise ValueError("`spatial_dims` must be 2 or 3.")

        # convert to tuple for 2D/3D usage
        patch_sizes = ensure_tuple_rep(patch_size, spatial_dims)
        window_sizes = ensure_tuple_rep(window_size, spatial_dims)
        self.patch_size = patch_sizes

        # optional check for input size, if you want
        # (remove this block if you don't need input size check)
        self._check_input_size(ensure_tuple_rep(img_size, spatial_dims))

        if not (0 <= drop_rate <= 1):
            raise ValueError("drop_rate should be between 0 and 1.")
        if not (0 <= attn_drop_rate <= 1):
            raise ValueError("attn_drop_rate should be between 0 and 1.")
        if not (0 <= dropout_path_rate <= 1):
            raise ValueError("dropout_path_rate should be between 0 and 1.")

        # embed_dim should be divisible by 12 in some official constraints
        if embed_dim % 12 != 0:
            raise ValueError("embed_dim should be divisible by 12.")

        self.n_channels = in_channels
        self.n_classes = out_channels
        self.normalize = normalize
        feature_size = embed_dim

        # create the Swin Transformer backbone
        self.swinViT = SwinTransformer(
            in_chans=self.n_channels,
            embed_dim=embed_dim,
            window_size=window_sizes,
            patch_size=self.patch_size,
            depths=depths,
            num_heads=num_heads,
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=dropout_path_rate,
            norm_layer=nn.LayerNorm,  # or partial(nn.LayerNorm, eps=1e-5) ...
            use_checkpoint=use_checkpoint,
            spatial_dims=spatial_dims,
            downsample=(
                look_up_option(downsample, MERGING_MODE)
                if isinstance(downsample, str)
                else downsample
            ),
            use_v2=use_v2,
        )

        # Encoder blocks (UnetrBasicBlock) for feature extraction at various scales
        self.encoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.n_channels,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )
        self.encoder2 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )
        self.encoder3 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=2 * feature_size,
            out_channels=2 * feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )
        self.encoder4 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=4 * feature_size,
            out_channels=4 * feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )
        self.encoder10 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=16 * feature_size,
            out_channels=16 * feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        # Decoder blocks (UnetrUpBlock)
        self.decoder5 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=16 * feature_size,
            out_channels=8 * feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )
        self.decoder4 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 8,
            out_channels=feature_size * 4,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )
        self.decoder3 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 4,
            out_channels=feature_size * 2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 2,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )
        self.decoder1 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=self.patch_size,
            norm_name=norm_name,
            res_block=True,
        )

        # Output projection layer
        self.out = UnetOutBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size,
            out_channels=self.n_classes,
        )

    def load_from(self, weights):
        """
        Load weights from a pre-trained model state dictionary (from external checkpoints).
        This function maps the keys from the external checkpoint to this module's parameters.
        """
        with torch.no_grad():
            self.swinViT.patch_embed.proj.weight.copy_(
                weights["state_dict"]["module.patch_embed.proj.weight"]
            )
            self.swinViT.patch_embed.proj.bias.copy_(
                weights["state_dict"]["module.patch_embed.proj.bias"]
            )
            for bname, block in self.swinViT.layers1[0].blocks.named_children():
                block.load_from(weights, n_block=bname, layer="layers1")
            self.swinViT.layers1[0].downsample.reduction.weight.copy_(
                weights["state_dict"]["module.layers1.0.downsample.reduction.weight"]
            )
            self.swinViT.layers1[0].downsample.norm.weight.copy_(
                weights["state_dict"]["module.layers1.0.downsample.norm.weight"]
            )
            self.swinViT.layers1[0].downsample.norm.bias.copy_(
                weights["state_dict"]["module.layers1.0.downsample.norm.bias"]
            )
            for bname, block in self.swinViT.layers2[0].blocks.named_children():
                block.load_from(weights, n_block=bname, layer="layers2")
            self.swinViT.layers2[0].downsample.reduction.weight.copy_(
                weights["state_dict"]["module.layers2.0.downsample.reduction.weight"]
            )
            self.swinViT.layers2[0].downsample.norm.weight.copy_(
                weights["state_dict"]["module.layers2.0.downsample.norm.weight"]
            )
            self.swinViT.layers2[0].downsample.norm.bias.copy_(
                weights["state_dict"]["module.layers2.0.downsample.norm.bias"]
            )
            for bname, block in self.swinViT.layers3[0].blocks.named_children():
                block.load_from(weights, n_block=bname, layer="layers3")
            self.swinViT.layers3[0].downsample.reduction.weight.copy_(
                weights["state_dict"]["module.layers3.0.downsample.reduction.weight"]
            )
            self.swinViT.layers3[0].downsample.norm.weight.copy_(
                weights["state_dict"]["module.layers3.0.downsample.norm.weight"]
            )
            self.swinViT.layers3[0].downsample.norm.bias.copy_(
                weights["state_dict"]["module.layers3.0.downsample.norm.bias"]
            )
            for bname, block in self.swinViT.layers4[0].blocks.named_children():
                block.load_from(weights, n_block=bname, layer="layers4")
            self.swinViT.layers4[0].downsample.reduction.weight.copy_(
                weights["state_dict"]["module.layers4.0.downsample.reduction.weight"]
            )
            self.swinViT.layers4[0].downsample.norm.weight.copy_(
                weights["state_dict"]["module.layers4.0.downsample.norm.weight"]
            )
            self.swinViT.layers4[0].downsample.norm.bias.copy_(
                weights["state_dict"]["module.layers4.0.downsample.norm.bias"]
            )

    @torch.jit.unused
    def _check_input_size(self, spatial_shape):
        """
        Check if input size is divisible by the required patch size^(depth=5).
        Raises ValueError if not divisible.
        """
        img_size = np.array(spatial_shape)
        remainder = (img_size % np.power(self.patch_size, 5)) > 0
        if remainder.any():
            wrong_dims = (np.where(remainder)[0] + 2).tolist()
            raise ValueError(
                f"Spatial dimensions {wrong_dims} of input image (shape: {spatial_shape}) must be divisible by patch_size^5."
            )

    def forward(self, x_in):
        """
        Forward pass of the SwinUNETR model.

        Args:
            x_in (torch.Tensor): Input tensor, shape [B, C, H, W] or [B, C, D, H, W].

        Returns:
            torch.Tensor: Segmentation logits of shape [B, num_classes, ...].
        """
        if not torch.jit.is_scripting():
            self._check_input_size(x_in.shape[2:])
        hidden_states_out = self.swinViT(x_in, self.normalize)

        # Low-level encoder features
        enc0 = self.encoder1(x_in)
        enc1 = self.encoder2(hidden_states_out[0])
        enc2 = self.encoder3(hidden_states_out[1])
        enc3 = self.encoder4(hidden_states_out[2])

        # Bottleneck features
        dec4 = self.encoder10(hidden_states_out[4])

        # Decoder pathway
        dec3 = self.decoder5(dec4, hidden_states_out[3])
        dec2 = self.decoder4(dec3, enc3)
        dec1 = self.decoder3(dec2, enc2)
        dec0 = self.decoder2(dec1, enc1)
        out = self.decoder1(dec0, enc0)

        # Final output head
        logits = self.out(out)
        return logits


def window_partition(x, window_size):
    """
    Partition input x into windows of size `window_size`.

    Args:
        x (torch.Tensor): Input feature map of shape [B, D, H, W, C] or [B, H, W, C].
        window_size (tuple): Local window size for each dimension.

    Returns:
        torch.Tensor: Windows reshaped to [num_windows * B, window_size.prod(), C].
    """
    x_shape = x.size()
    if len(x_shape) == 5:
        b, d, h, w, c = x_shape
        x = x.view(
            b,
            d // window_size[0],
            window_size[0],
            h // window_size[1],
            window_size[1],
            w // window_size[2],
            window_size[2],
            c,
        )
        windows = (
            x.permute(0, 1, 3, 5, 2, 4, 6, 7)
            .contiguous()
            .view(-1, window_size[0] * window_size[1] * window_size[2], c)
        )
    elif len(x_shape) == 4:
        b, h, w, c = x.shape
        x = x.view(
            b,
            h // window_size[0],
            window_size[0],
            w // window_size[1],
            window_size[1],
            c,
        )
        windows = (
            x.permute(0, 1, 3, 2, 4, 5)
            .contiguous()
            .view(-1, window_size[0] * window_size[1], c)
        )
    return windows


def window_reverse(windows, window_size, dims):
    """
    Reverse the window partition to the original spatial shape.

    Args:
        windows (torch.Tensor): Windowed tensor of shape [num_windows * B, window_size.prod(), C].
        window_size (tuple): The local window size.
        dims (list): Original shape of [B, D, H, W] or [B, H, W].

    Returns:
        torch.Tensor: Reshaped tensor to the original dimension.
    """
    if len(dims) == 4:
        b, d, h, w = dims
        x = windows.view(
            b,
            d // window_size[0],
            h // window_size[1],
            w // window_size[2],
            window_size[0],
            window_size[1],
            window_size[2],
            -1,
        )
        x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(b, d, h, w, -1)
    elif len(dims) == 3:
        b, h, w = dims
        x = windows.view(
            b,
            h // window_size[0],
            w // window_size[1],
            window_size[0],
            window_size[1],
            -1,
        )
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h, w, -1)
    return x


def get_window_size(x_size, window_size, shift_size=None):
    """
    Adjust window_size and shift_size based on actual input spatial size.
    If the input size is smaller than the window_size, use the smaller dimension.

    Args:
        x_size (tuple): Spatial dimension of input, e.g. (H, W) or (D, H, W).
        window_size (tuple): Window size for each spatial dimension.
        shift_size (tuple or None): Shift size to be used in SwinTransformer.

    Returns:
        tuple: (use_window_size) or (use_window_size, use_shift_size).
    """
    use_window_size = list(window_size)
    if shift_size is not None:
        use_shift_size = list(shift_size)
    for i in range(len(x_size)):
        if x_size[i] <= window_size[i]:
            use_window_size[i] = x_size[i]
            if shift_size is not None:
                use_shift_size[i] = 0

    if shift_size is None:
        return tuple(use_window_size)
    else:
        return tuple(use_window_size), tuple(use_shift_size)


class WindowAttention(nn.Module):
    """
    Window-based Multi-head Self-Attention with relative position bias.
    Based on: "Swin Transformer" (Liu et al. 2021).
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: Sequence[int],
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        # Generate a parameter table of relative positions
        mesh_args = torch.meshgrid.__kwdefaults__
        if len(self.window_size) == 3:
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros(
                    (2 * self.window_size[0] - 1)
                    * (2 * self.window_size[1] - 1)
                    * (2 * self.window_size[2] - 1),
                    num_heads,
                )
            )
            coords_d = torch.arange(self.window_size[0])
            coords_h = torch.arange(self.window_size[1])
            coords_w = torch.arange(self.window_size[2])
            if mesh_args is not None:
                coords = torch.stack(
                    torch.meshgrid(coords_d, coords_h, coords_w, indexing="ij")
                )
            else:
                coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w))
            coords_flatten = torch.flatten(coords, 1)
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()
            relative_coords[:, :, 0] += self.window_size[0] - 1
            relative_coords[:, :, 1] += self.window_size[1] - 1
            relative_coords[:, :, 2] += self.window_size[2] - 1
            relative_coords[:, :, 0] *= (2 * self.window_size[1] - 1) * (
                2 * self.window_size[2] - 1
            )
            relative_coords[:, :, 1] *= 2 * self.window_size[2] - 1
        elif len(self.window_size) == 2:
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros(
                    (2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads
                )
            )
            coords_h = torch.arange(self.window_size[0])
            coords_w = torch.arange(self.window_size[1])
            if mesh_args is not None:
                coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))
            else:
                coords = torch.stack(torch.meshgrid(coords_h, coords_w))
            coords_flatten = torch.flatten(coords, 1)
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()
            relative_coords[:, :, 0] += self.window_size[0] - 1
            relative_coords[:, :, 1] += self.window_size[1] - 1
            relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1

        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask):
        """
        Args:
            x (torch.Tensor): shape [B*nW, window_size*window_size, C].
            mask (torch.Tensor or None): attention mask for shifted windows.

        Returns:
            torch.Tensor: shape [B*nW, window_size*window_size, C].
        """
        b, n, c = x.shape
        qkv = (
            self.qkv(x)
            .reshape(b, n, 3, self.num_heads, c // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.clone()[:n, :n].reshape(-1)
        ].reshape(n, n, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nw = mask.shape[0]
            attn = attn.view(b // nw, nw, self.num_heads, n, n) + mask.unsqueeze(
                1
            ).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, n, n)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn).to(v.dtype)
        x = (attn @ v).transpose(1, 2).reshape(b, n, c)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    """
    Basic Swin Transformer block that applies:
    - LayerNorm -> WindowAttention -> Add&Norm -> FeedForward -> Add
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: Sequence[int],
        shift_size: Sequence[int],
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer: str = "GELU",
        norm_layer: type[LayerNorm] = nn.LayerNorm,
        use_checkpoint: bool = False,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.use_checkpoint = use_checkpoint

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim,
            window_size=self.window_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            hidden_size=dim,
            mlp_dim=mlp_hidden_dim,
            act=act_layer,
            dropout_rate=drop,
            dropout_mode="swin",
        )

    def forward_part1(self, x, mask_matrix):
        """
        First half of forward, containing WindowAttention with optional shift.
        """
        x_shape = x.size()
        x = self.norm1(x)

        if len(x_shape) == 5:
            b, d, h, w, c = x_shape
            window_size, shift_size = get_window_size(
                (d, h, w), self.window_size, self.shift_size
            )
            pad_l = pad_t = pad_d0 = 0
            pad_d1 = (window_size[0] - d % window_size[0]) % window_size[0]
            pad_b = (window_size[1] - h % window_size[1]) % window_size[1]
            pad_r = (window_size[2] - w % window_size[2]) % window_size[2]
            x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b, pad_d0, pad_d1))
            _, dp, hp, wp, _ = x.shape
            dims = [b, dp, hp, wp]
        elif len(x_shape) == 4:
            b, h, w, c = x_shape
            window_size, shift_size = get_window_size(
                (h, w), self.window_size, self.shift_size
            )
            pad_l = pad_t = 0
            pad_b = (window_size[0] - h % window_size[0]) % window_size[0]
            pad_r = (window_size[1] - w % window_size[1]) % window_size[1]
            x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
            _, hp, wp, _ = x.shape
            dims = [b, hp, wp]

        if any(i > 0 for i in shift_size):
            if len(x_shape) == 5:
                shifted_x = torch.roll(
                    x,
                    shifts=(-shift_size[0], -shift_size[1], -shift_size[2]),
                    dims=(1, 2, 3),
                )
            elif len(x_shape) == 4:
                shifted_x = torch.roll(
                    x, shifts=(-shift_size[0], -shift_size[1]), dims=(1, 2)
                )
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None

        x_windows = window_partition(shifted_x, window_size)
        attn_windows = self.attn(x_windows, mask=attn_mask)
        attn_windows = attn_windows.view(-1, *(window_size + (c,)))
        shifted_x = window_reverse(attn_windows, window_size, dims)

        if any(i > 0 for i in shift_size):
            if len(x_shape) == 5:
                x = torch.roll(
                    shifted_x,
                    shifts=(shift_size[0], shift_size[1], shift_size[2]),
                    dims=(1, 2, 3),
                )
            elif len(x_shape) == 4:
                x = torch.roll(
                    shifted_x, shifts=(shift_size[0], shift_size[1]), dims=(1, 2)
                )
        else:
            x = shifted_x

        # Remove padding
        if len(x_shape) == 5:
            if pad_d1 > 0 or pad_r > 0 or pad_b > 0:
                x = x[:, :d, :h, :w, :].contiguous()
        elif len(x_shape) == 4:
            if pad_r > 0 or pad_b > 0:
                x = x[:, :h, :w, :].contiguous()

        return x

    def forward_part2(self, x):
        """
        Second half of forward, containing MLP feedforward.
        """
        return self.drop_path(self.mlp(self.norm2(x)))

    def load_from(self, weights, n_block, layer):
        """
        Load external pretrained weights for a specific block within the SwinTransformer.
        """
        root = f"module.{layer}.0.blocks.{n_block}."
        block_names = [
            "norm1.weight",
            "norm1.bias",
            "attn.relative_position_bias_table",
            "attn.relative_position_index",
            "attn.qkv.weight",
            "attn.qkv.bias",
            "attn.proj.weight",
            "attn.proj.bias",
            "norm2.weight",
            "norm2.bias",
            "mlp.fc1.weight",
            "mlp.fc1.bias",
            "mlp.fc2.weight",
            "mlp.fc2.bias",
        ]
        with torch.no_grad():
            self.norm1.weight.copy_(weights["state_dict"][root + block_names[0]])
            self.norm1.bias.copy_(weights["state_dict"][root + block_names[1]])
            self.attn.relative_position_bias_table.copy_(
                weights["state_dict"][root + block_names[2]]
            )
            self.attn.relative_position_index.copy_(
                weights["state_dict"][root + block_names[3]]
            )
            self.attn.qkv.weight.copy_(weights["state_dict"][root + block_names[4]])
            self.attn.qkv.bias.copy_(weights["state_dict"][root + block_names[5]])
            self.attn.proj.weight.copy_(weights["state_dict"][root + block_names[6]])
            self.attn.proj.bias.copy_(weights["state_dict"][root + block_names[7]])
            self.norm2.weight.copy_(weights["state_dict"][root + block_names[8]])
            self.norm2.bias.copy_(weights["state_dict"][root + block_names[9]])
            self.mlp.linear1.weight.copy_(weights["state_dict"][root + block_names[10]])
            self.mlp.linear1.bias.copy_(weights["state_dict"][root + block_names[11]])
            self.mlp.linear2.weight.copy_(weights["state_dict"][root + block_names[12]])
            self.mlp.linear2.bias.copy_(weights["state_dict"][root + block_names[13]])

    def forward(self, x, mask_matrix):
        """
        Full forward pass of the SwinTransformerBlock, combining attention & MLP.
        """
        shortcut = x
        if self.use_checkpoint:
            x = checkpoint.checkpoint(
                self.forward_part1, x, mask_matrix, use_reentrant=False
            )
        else:
            x = self.forward_part1(x, mask_matrix)
        x = shortcut + self.drop_path(x)

        if self.use_checkpoint:
            x = x + checkpoint.checkpoint(self.forward_part2, x, use_reentrant=False)
        else:
            x = x + self.forward_part2(x)
        return x


class PatchMergingV2(nn.Module):
    """
    Patch merging layer V2 for downsampling, aggregates 8 patches (3D) or 4 patches (2D).
    """

    def __init__(
        self,
        dim: int,
        norm_layer: type[LayerNorm] = nn.LayerNorm,
        spatial_dims: int = 3,
    ) -> None:
        super().__init__()
        self.dim = dim
        if spatial_dims == 3:
            self.reduction = nn.Linear(8 * dim, 2 * dim, bias=False)
            self.norm = norm_layer(8 * dim)
        elif spatial_dims == 2:
            self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
            self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        Forward pass that rearranges spatial patches into the channel dimension, then projects down.
        """
        x_shape = x.size()
        if len(x_shape) == 5:
            b, d, h, w, c = x_shape
            pad_input = (h % 2 == 1) or (w % 2 == 1) or (d % 2 == 1)
            if pad_input:
                x = F.pad(x, (0, 0, 0, w % 2, 0, h % 2, 0, d % 2))
            x = torch.cat(
                [
                    x[:, i::2, j::2, k::2, :]
                    for i, j, k in itertools.product(range(2), range(2), range(2))
                ],
                -1,
            )
        elif len(x_shape) == 4:
            b, h, w, c = x_shape
            pad_input = (h % 2 == 1) or (w % 2 == 1)
            if pad_input:
                x = F.pad(x, (0, 0, 0, w % 2, 0, h % 2))
            x = torch.cat(
                [x[:, j::2, i::2, :] for i, j in itertools.product(range(2), range(2))],
                -1,
            )

        x = self.norm(x)
        x = self.reduction(x)
        return x


class PatchMerging(PatchMergingV2):
    """
    The original PatchMerging module from v0.9.0. Inherits from PatchMergingV2.
    """

    def forward(self, x):
        """
        Overriding forward to replicate the exact patch merging logic used in older versions of SwinUNETR.
        """
        x_shape = x.size()
        if len(x_shape) == 4:
            return super().forward(x)
        if len(x_shape) != 5:
            raise ValueError(f"expecting 5D x, got {x.shape}.")
        b, d, h, w, c = x_shape
        pad_input = (h % 2 == 1) or (w % 2 == 1) or (d % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, w % 2, 0, h % 2, 0, d % 2))
        x0 = x[:, 0::2, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, 0::2, :]
        x3 = x[:, 0::2, 0::2, 1::2, :]
        x4 = x[:, 1::2, 0::2, 1::2, :]
        x5 = x[:, 0::2, 1::2, 0::2, :]
        x6 = x[:, 0::2, 0::2, 1::2, :]
        x7 = x[:, 1::2, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3, x4, x5, x6, x7], -1)
        x = self.norm(x)
        x = self.reduction(x)
        return x


MERGING_MODE = {"merging": PatchMerging, "mergingv2": PatchMergingV2}


def compute_mask(dims, window_size, shift_size, device):
    """
    Compute attention mask used for shifted windows in SwinTransformer.
    """
    cnt = 0
    if len(dims) == 3:
        d, h, w = dims
        img_mask = torch.zeros((1, d, h, w, 1), device=device)
        for d_slice in (
            slice(-window_size[0]),
            slice(-window_size[0], -shift_size[0]),
            slice(-shift_size[0], None),
        ):
            for h_slice in (
                slice(-window_size[1]),
                slice(-window_size[1], -shift_size[1]),
                slice(-shift_size[1], None),
            ):
                for w_slice in (
                    slice(-window_size[2]),
                    slice(-window_size[2], -shift_size[2]),
                    slice(-shift_size[2], None),
                ):
                    img_mask[:, d_slice, h_slice, w_slice, :] = cnt
                    cnt += 1
    elif len(dims) == 2:
        h, w = dims
        img_mask = torch.zeros((1, h, w, 1), device=device)
        for h_slice in (
            slice(-window_size[0]),
            slice(-window_size[0], -shift_size[0]),
            slice(-shift_size[0], None),
        ):
            for w_slice in (
                slice(-window_size[1]),
                slice(-window_size[1], -shift_size[1]),
                slice(-shift_size[1], None),
            ):
                img_mask[:, h_slice, w_slice, :] = cnt
                cnt += 1

    mask_windows = window_partition(img_mask, window_size)
    mask_windows = mask_windows.squeeze(-1)
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(
        attn_mask == 0, float(0.0)
    )
    return attn_mask


class BasicLayer(nn.Module):
    """
    Basic Swin Transformer layer for one stage, consisting of multiple SwinTransformerBlocks + optional PatchMerging.
    """

    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        window_size: Sequence[int],
        drop_path: list,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        norm_layer: type[LayerNorm] = nn.LayerNorm,
        downsample: nn.Module | None = None,
        use_checkpoint: bool = False,
    ) -> None:
        super().__init__()
        self.window_size = window_size
        self.shift_size = tuple(i // 2 for i in window_size)
        self.no_shift = tuple(0 for i in window_size)
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList(
            [
                SwinTransformerBlock(
                    dim=dim,
                    num_heads=num_heads,
                    window_size=self.window_size,
                    shift_size=self.no_shift if (i % 2 == 0) else self.shift_size,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=(
                        drop_path[i] if isinstance(drop_path, list) else drop_path
                    ),
                    norm_layer=norm_layer,
                    use_checkpoint=use_checkpoint,
                )
                for i in range(depth)
            ]
        )
        self.downsample = downsample
        if callable(self.downsample):
            self.downsample = downsample(
                dim=dim, norm_layer=norm_layer, spatial_dims=len(self.window_size)
            )

    def forward(self, x):
        """
        Forward pass through multiple SwinTransformerBlocks, then optional downsample at the end of the stage.
        """
        x_shape = x.size()
        if len(x_shape) == 5:
            b, c, d, h, w = x_shape
            window_size, shift_size = get_window_size(
                (d, h, w), self.window_size, self.shift_size
            )
            x = rearrange(x, "b c d h w -> b d h w c")
            dp = int(np.ceil(d / window_size[0])) * window_size[0]
            hp = int(np.ceil(h / window_size[1])) * window_size[1]
            wp = int(np.ceil(w / window_size[2])) * window_size[2]
            attn_mask = compute_mask([dp, hp, wp], window_size, shift_size, x.device)

            for blk in self.blocks:
                x = blk(x, attn_mask)
            x = x.view(b, d, h, w, -1)
            if self.downsample is not None:
                x = self.downsample(x)
            x = rearrange(x, "b d h w c -> b c d h w")

        elif len(x_shape) == 4:
            b, c, h, w = x_shape
            window_size, shift_size = get_window_size(
                (h, w), self.window_size, self.shift_size
            )
            x = rearrange(x, "b c h w -> b h w c")
            hp = int(np.ceil(h / window_size[0])) * window_size[0]
            wp = int(np.ceil(w / window_size[1])) * window_size[1]
            attn_mask = compute_mask([hp, wp], window_size, shift_size, x.device)

            for blk in self.blocks:
                x = blk(x, attn_mask)
            x = x.view(b, h, w, -1)
            if self.downsample is not None:
                x = self.downsample(x)
            x = rearrange(x, "b h w c -> b c h w")

        return x


class SwinTransformer(nn.Module):
    """
    Core Swin Transformer backbone that produces multi-scale features.

    Args:
        in_chans: number of input channels.
        embed_dim: embedding dimension for patch embedding.
        window_size: local window size in each stage.
        patch_size: patch size.
        depths: list of layer depths for each stage.
        num_heads: list of head counts for each stage.
        mlp_ratio: ratio of MLP hidden dim to embedding dim.
        qkv_bias: whether to include a learnable bias to QKV.
        drop_rate: dropout rate for patch embedding & feed-forward layers.
        attn_drop_rate: attention dropout rate.
        drop_path_rate: drop path (stochastic depth) rate.
        norm_layer: normalization layer used in each block.
        patch_norm: whether to apply normalization after patch embedding.
        use_checkpoint: whether to use gradient checkpointing for memory efficiency.
        spatial_dims: number of spatial dimensions (2 or 3).
        downsample: module used for downsampling (PatchMerging).
        use_v2: whether to add a residual conv block in each stage's beginning (SwinUNETR v2).

    Returns:
        List of feature maps at each stage: x0_out, x1_out, x2_out, x3_out, x4_out.
    """

    def __init__(
        self,
        in_chans: int,
        embed_dim: int,
        window_size: Sequence[int],
        patch_size: Sequence[int],
        depths: Sequence[int],
        num_heads: Sequence[int],
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        norm_layer: type[LayerNorm] = nn.LayerNorm,
        patch_norm: bool = False,
        use_checkpoint: bool = False,
        spatial_dims: int = 3,
        downsample="merging",
        use_v2=False,
    ) -> None:
        super().__init__()
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.window_size = window_size
        self.patch_size = patch_size
        self.use_v2 = use_v2

        # Patch embedding
        self.patch_embed = PatchEmbed(
            patch_size=self.patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None,
            spatial_dims=spatial_dims,
        )
        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth schedule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # Build layers
        self.layers1 = nn.ModuleList()
        self.layers2 = nn.ModuleList()
        self.layers3 = nn.ModuleList()
        self.layers4 = nn.ModuleList()

        if self.use_v2:
            self.layers1c = nn.ModuleList()
            self.layers2c = nn.ModuleList()
            self.layers3c = nn.ModuleList()
            self.layers4c = nn.ModuleList()

        down_sample_mod = (
            look_up_option(downsample, MERGING_MODE)
            if isinstance(downsample, str)
            else downsample
        )

        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2**i_layer),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=self.window_size,
                drop_path=dpr[sum(depths[:i_layer]) : sum(depths[: i_layer + 1])],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                norm_layer=norm_layer,
                downsample=down_sample_mod,
                use_checkpoint=use_checkpoint,
            )
            if i_layer == 0:
                self.layers1.append(layer)
            elif i_layer == 1:
                self.layers2.append(layer)
            elif i_layer == 2:
                self.layers3.append(layer)
            elif i_layer == 3:
                self.layers4.append(layer)

            if self.use_v2:
                layerc = UnetrBasicBlock(
                    spatial_dims=3,
                    in_channels=embed_dim * 2**i_layer,
                    out_channels=embed_dim * 2**i_layer,
                    kernel_size=3,
                    stride=1,
                    norm_name="instance",
                    res_block=True,
                )
                if i_layer == 0:
                    self.layers1c.append(layerc)
                elif i_layer == 1:
                    self.layers2c.append(layerc)
                elif i_layer == 2:
                    self.layers3c.append(layerc)
                elif i_layer == 3:
                    self.layers4c.append(layerc)

        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))

    def proj_out(self, x, normalize=False):
        """
        Optionally apply layer normalization to output. Used to match upstream usage in forward().

        Args:
            x (torch.Tensor): feature map of shape [B, C, *spatial].
            normalize (bool): if True, apply layer_norm across the channel dimension.
        """
        if normalize:
            x_shape = x.size()
            if len(x_shape) == 5:
                n, ch, d, h, w = x_shape
                x = rearrange(x, "n c d h w -> n d h w c")
                x = F.layer_norm(x, [ch])
                x = rearrange(x, "n d h w c -> n c d h w")
            elif len(x_shape) == 4:
                n, ch, h, w = x_shape
                x = rearrange(x, "n c h w -> n h w c")
                x = F.layer_norm(x, [ch])
                x = rearrange(x, "n h w c -> n c h w")
        return x

    def forward(self, x, normalize=True):
        """
        Forward pass that processes x through each stage of the Swin transformer.

        Args:
            x (torch.Tensor): input tensor [B, C, *spatial].
            normalize (bool): if True, apply LN to each stage output.

        Returns:
            list of torch.Tensor: [x0_out, x1_out, x2_out, x3_out, x4_out] from each stage.
        """
        x0 = self.patch_embed(x)  # initial patch embedding
        x0 = self.pos_drop(x0)
        x0_out = self.proj_out(x0, normalize)

        # stage 1
        if self.use_v2:
            x0 = self.layers1c[0](x0.contiguous())
        x1 = self.layers1[0](x0.contiguous())
        x1_out = self.proj_out(x1, normalize)

        # stage 2
        if self.use_v2:
            x1 = self.layers2c[0](x1.contiguous())
        x2 = self.layers2[0](x1.contiguous())
        x2_out = self.proj_out(x2, normalize)

        # stage 3
        if self.use_v2:
            x2 = self.layers3c[0](x2.contiguous())
        x3 = self.layers3[0](x2.contiguous())
        x3_out = self.proj_out(x3, normalize)

        # stage 4
        if self.use_v2:
            x3 = self.layers4c[0](x3.contiguous())
        x4 = self.layers4[0](x3.contiguous())
        x4_out = self.proj_out(x4, normalize)

        return [x0_out, x1_out, x2_out, x3_out, x4_out]


def filter_swinunetr(key, value):
    """
    A filter function for selectively loading pretrained weights from third-party sources.
    Skips certain keys not used in MONAI's SwinUNETR.

    Args:
        key (str): the parameter name from the source state dict.
        value (torch.Tensor): the parameter values.

    Returns:
        tuple(str, torch.Tensor) or None: adjusted key-value pair or None if dropped.
    """
    if key in [
        "encoder.mask_token",
        "encoder.norm.weight",
        "encoder.norm.bias",
        "out.conv.conv.weight",
        "out.conv.conv.bias",
    ]:
        return None

    if key[:8] == "encoder.":
        if key[8:19] == "patch_embed":
            new_key = "swinViT." + key[8:]
        else:
            new_key = "swinViT." + key[8:18] + key[20:]
        return new_key, value
    else:
        return None
