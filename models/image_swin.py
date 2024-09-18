import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from .ptopk import PerturbedTopK
from .scorer import scorer
import time
path = './attention/'
class MixAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, dim_y, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.dim_y = dim_y
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.qkv_y = nn.Linear(dim_y, dim_y * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_y = nn.Linear(dim_y, dim_y)
        self.proj_drop = nn.Dropout(proj_drop)
        self.proj_qx = nn.Linear(dim, dim_y)
        self.proj_qy = nn.Linear(dim_y, dim)
        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, y, nuclei_mask = None, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        B_y, N_y, C_y = y.shape
        # nW = mask.shape[0]
        # nuclei_mask = numpy.zeros(B_ // nW, nW, self.num_heads, N, N)
        # replace_x = x.data
        # replace_x = replace_x.view(B_ // nW, nW, N, C)
        # replace_x = replace_x(B_ // nW, nW, N, 0)
        # for a, b, c in replace_x:
        #     if replace_x(a, b, c, 0) == 0:
        #         for head in range(0, self.num_heads):
        #             nuclei_mask[a, b, head, c, :] = float(-100.0)
        #             nuclei_mask[a, b, head, c, :] = float(-100.0)
        # nuclei_mask = numpy.zeros((B_, self.num_heads, N, N)) #nub_windows*B, heads, N, N
        # replace_x = x.data
        # print(replace_x.shape)
        # print(B_)
        # print(N)
        # for a in range(B_):
        #     for b in range(N):
        #         print(replace_x[a, b, 0])
        #         if replace_x[a, b, 0] == 0:
        #             print('success !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        #             for head in range(0, self.num_heads):
        #                 nuclei_mask[a, head, b, :] = float(-100.0)
        #                 nuclei_mask[a, head, :, b] = float(-100.0)


        # torch.where(nuclei_mask == float(0), float(1), nuclei_mask)
        kv = self.qkv(x).reshape(B_, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        kv_y = self.qkv_y(y).reshape(B_y, N_y, 2, self.num_heads, C_y // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]  # make torchscript happy (cannot use tensor as tuple)
        k_y, v_y = kv_y[0], kv_y[1]
        q = self.proj_qx(x).reshape(B_y, N_y, self.num_heads, C_y // self.num_heads).permute(0, 2, 1, 3)
        q_y = self.proj_qy(y).reshape(B_, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        q = q * self.scale
        q_y = q_y * self.scale
        attn_x = (q_y @ k.transpose(-2, -1))
        attn_y = (q @ k_y.transpose(-2, -1))
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn_x = attn_x + relative_position_bias.unsqueeze(0)
        attn_y = attn_y + relative_position_bias.unsqueeze(0)
        if torch.isnan(attn_x).any() or torch.isnan(attn_y).any():
            print("owarida")
        if nuclei_mask is not None:
            original_nuclei_mask = nuclei_mask
            nuclei_mask = nuclei_mask.unsqueeze(1)
            nuclei_multi_mask = nuclei_mask
            tuple_head = [nuclei_multi_mask]*self.num_heads
            tuple_head = tuple(tuple_head)
            nuclei_multi_mask = torch.cat(tuple_head, 1)
            tuple_mask = [nuclei_multi_mask]*(N)
            tuple_mask = tuple(tuple_mask)
            nuclei_multi_mask = torch.cat(tuple_mask, 2)
            nuclei_multi_mask = nuclei_multi_mask.squeeze(3)
            attn_x = attn_x.permute(0, 1, 3, 2)
            attn_x = attn_x.contiguous().view(-1, self.num_heads, N * N)
            attn_x = attn_x + nuclei_multi_mask
            attn_x = attn_x.view(-1, self.num_heads, N, N).permute(0, 1, 3, 2)
            attn_x = attn_x.contiguous().view(-1, self.num_heads, N * N)
            attn_x = attn_x + nuclei_multi_mask
            attn_x = attn_x.contiguous().view(-1, self.num_heads, N, N)
            attn_y = attn_y.permute(0, 1, 3, 2)
            attn_y = attn_y.contiguous().view(-1, self.num_heads, N * N)
            attn_y = attn_y + nuclei_multi_mask
            attn_y = attn_y.view(-1, self.num_heads, N, N).permute(0, 1, 3, 2)
            attn_y = attn_y.contiguous().view(-1, self.num_heads, N * N)
            attn_y = attn_y + nuclei_multi_mask
            attn_y = attn_y.contiguous().view(-1, self.num_heads, N, N)
        if mask is not None:
            nW = mask.shape[0]
            attn_x = attn_x.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)  # + nuclei_mask
            attn_y = attn_y.view(B_y // nW, nW, self.num_heads, N_y, N_y) + mask.unsqueeze(1).unsqueeze(0)  # + nuclei_mask
            attn_x = attn_x.view(-1, self.num_heads, N, N)
            attn_y = attn_y.view(-1, self.num_heads, N_y, N_y)
            attn_x = self.softmax(attn_x)
            attn_y = self.softmax(attn_y)
        else:
            attn_x = self.softmax(attn_x)
            attn_y = self.softmax(attn_y)
        attn_x = self.attn_drop(attn_x)
        attn_y = self.attn_drop(attn_y)
        x = (attn_x @ v).transpose(1, 2).reshape(B_, N, C)
        y = (attn_y @ v_y).transpose(1, 2).reshape(B_y, N_y, C_y)
        ticks = time.time()
        # torch.save(attn_x, path+str(ticks)+'_mix_'+'x_'+str(C)+'.pt')
        # torch.save(attn_y, path+str(ticks)+'_mix_'+'y_'+str(C_y)+'.pt')
        x = self.proj(x)
        y = self.proj_y(y)
        x = self.proj_drop(x)
        y = self.proj_drop(y)
        return x, y, original_nuclei_mask



class MixTransformerBlock(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, dim_y, input_resolution, num_heads, mix_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.dim_y = dim_y
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"
        self.norm1 = norm_layer(dim, eps=1e-4)
        self.norm3 = norm_layer(dim_y, eps=1e-4)
        self.attn = MixAttention(
            dim, dim_y, window_size=to_2tuple(self.window_size), num_heads=mix_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim, eps=1e-4)
        self.norm4 = norm_layer(dim_y, eps=1e-4)
        mlp_hidden_dim = int(dim * mlp_ratio)
        mlp_hidden_dim_y = int(dim_y * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.mlp_y = Mlp(in_features=dim_y, hidden_features=mlp_hidden_dim_y, act_layer=act_layer, drop=drop)
        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows, _ = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x, y, nuclei_mask):
        H, W = self.input_resolution
        # print(x.shape)
        B, L, C = x.shape
        B_y, L_y, C_y = y.shape
        # assert L == H * W, "input feature has wrong size"
        shortcut = x
        shortcut_y = y
        x = self.norm1(x)
        y = self.norm3(y)
        x = x.view(B, H, W, C)
        y = y.view(B_y, H, W, C_y)
        nuclei_mask = nuclei_mask.view(B, H, W, 1)
        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            shifted_y = torch.roll(y, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            nuclei_mask = torch.roll(nuclei_mask, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
            shifted_y = y
            nuclei_mask = nuclei_mask
        # partition windows
        x_windows, nuclei_mask = window_partition(shifted_x, self.window_size, nuclei_mask)  # nW*B, window_size, window_size, C
        y_windows, nuclei_mask = window_partition(shifted_y, self.window_size,
                                                  nuclei_mask)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C
        y_windows = y_windows.view(-1, self.window_size * self.window_size, C_y)  # nW*B, window_size*window_size, C
        nuclei_mask = nuclei_mask.view(-1, self.window_size * self.window_size, 1)
        # W-MSA/SW-MSA
        attn_windows_x, attn_windows_y, nuclei_mask = self.attn(x_windows, y_windows, nuclei_mask=nuclei_mask, mask=self.attn_mask)  # nW*B, window_size*window_size, C
        # merge windows
        attn_windows_x = attn_windows_x.view(-1, self.window_size, self.window_size, C)
        attn_windows_y = attn_windows_y.view(-1, self.window_size, self.window_size, C_y)
        nuclei_mask = nuclei_mask.view(-1, self.window_size, self.window_size, 1)
        shifted_x = window_reverse(attn_windows_x, self.window_size, H, W)  # B H' W' C
        shifted_y = window_reverse(attn_windows_y, self.window_size, H, W)  # B H' W' C
        nuclei_mask = window_reverse(nuclei_mask, self.window_size, H, W)
        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
            y = torch.roll(shifted_y, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
            nuclei_mask = torch.roll(nuclei_mask, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
            y = shifted_y
        x = x.view(B, H * W, C)
        y = y.view(B_y, H * W, C_y)
        nuclei_mask = nuclei_mask.view(B, H * W, 1)

        # FFN
        x = shortcut + self.drop_path(x)
        y = shortcut_y + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        y = y + self.drop_path(self.mlp_y(self.norm4(y)))
        return x, y, nuclei_mask

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, nuclei_mask = None, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        # nW = mask.shape[0]
        # nuclei_mask = numpy.zeros(B_ // nW, nW, self.num_heads, N, N)
        # replace_x = x.data
        # replace_x = replace_x.view(B_ // nW, nW, N, C)
        # replace_x = replace_x(B_ // nW, nW, N, 0)
        # for a, b, c in replace_x:
        #     if replace_x(a, b, c, 0) == 0:
        #         for head in range(0, self.num_heads):
        #             nuclei_mask[a, b, head, c, :] = float(-100.0)
        #             nuclei_mask[a, b, head, c, :] = float(-100.0)
        # nuclei_mask = numpy.zeros((B_, self.num_heads, N, N)) #nub_windows*B, heads, N, N
        # replace_x = x.data
        # print(replace_x.shape)
        # print(B_)
        # print(N)
        # for a in range(B_):
        #     for b in range(N):
        #         print(replace_x[a, b, 0])
        #         if replace_x[a, b, 0] == 0:
        #             print('success !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        #             for head in range(0, self.num_heads):
        #                 nuclei_mask[a, head, b, :] = float(-100.0)
        #                 nuclei_mask[a, head, :, b] = float(-100.0)


        # torch.where(nuclei_mask == float(0), float(1), nuclei_mask)
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)
        if torch.isnan(attn).any():
            print("owarida!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        if nuclei_mask is not None:
            original_nuclei_mask = nuclei_mask
            nuclei_mask = nuclei_mask.unsqueeze(1)
            nuclei_multi_mask = nuclei_mask
            tuple_head = [nuclei_multi_mask]*self.num_heads
            tuple_head = tuple(tuple_head)
            nuclei_multi_mask = torch.cat(tuple_head, 1)
            tuple_mask = [nuclei_multi_mask]*(N)
            tuple_mask = tuple(tuple_mask)
            nuclei_multi_mask = torch.cat(tuple_mask, 2)
            nuclei_multi_mask = nuclei_multi_mask.squeeze(3)
            attn = attn.permute(0, 1, 3, 2)
            attn = attn.contiguous().view(-1, self.num_heads, N*N)
            attn = attn + nuclei_multi_mask
            attn = attn.view(-1, self.num_heads, N, N).permute(0, 1, 3, 2)
            attn = attn.contiguous().view(-1, self.num_heads, N*N)
            attn = attn + nuclei_multi_mask
            attn = attn.contiguous().view(-1, self.num_heads, N, N)
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)  # + nuclei_mask
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)
        ticks = time.time()
        # torch.save(attn, path+str(ticks)+'_single_'+str(C)+'.pt')
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, original_nuclei_mask


def window_partition(x, window_size, nuclei_mask=None):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """

    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    if nuclei_mask is not None:
        nuclei_mask = nuclei_mask.view(B, H // window_size, window_size, W // window_size, window_size, 1)
        nuclei_mask = nuclei_mask.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, 1)
    return windows, nuclei_mask

def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"
        self.norm1 = norm_layer(dim, eps=1e-4)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim, eps=1e-4)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows, _ = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x, nuclei_mask):
        H, W = self.input_resolution
        # print(x.shape) #这里的输入是B H W C
        B, L, C = x.shape
        # assert L == H * W, "input feature has wrong size"
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)
        nuclei_mask = nuclei_mask.view(B, H, W, 1)
        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            nuclei_mask = torch.roll(nuclei_mask, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
            nuclei_mask = nuclei_mask
        # partition windows
        x_windows, nuclei_mask = window_partition(shifted_x, self.window_size, nuclei_mask)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C
        nuclei_mask = nuclei_mask.view(-1, self.window_size * self.window_size, 1)
        # W-MSA/SW-MSA
        attn_windows, nuclei_mask = self.attn(x_windows, nuclei_mask=nuclei_mask, mask=self.attn_mask)  # nW*B, window_size*window_size, C
        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        nuclei_mask = nuclei_mask.view(-1, self.window_size, self.window_size, 1)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C
        nuclei_mask = window_reverse(nuclei_mask, self.window_size, H, W)
        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
            nuclei_mask = torch.roll(nuclei_mask, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)
        nuclei_mask = nuclei_mask.view(B, H * W, 1)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x, nuclei_mask

class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x, nucleimask=None):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C
        x = self.norm(x)
        x = self.reduction(x)
        if nucleimask is not None:
            nucleimask = nucleimask.view(B, H, W, 1)

            n0 = nucleimask[:, 0::2, 0::2, :]  # B H/2 W/2 C
            n1 = nucleimask[:, 1::2, 0::2, :]  # B H/2 W/2 C
            n2 = nucleimask[:, 0::2, 1::2, :]  # B H/2 W/2 C
            n3 = nucleimask[:, 1::2, 1::2, :]  # B H/2 W/2 C
            n = torch.cat([n0, n1, n2, n3], -1)  # B H/2 W/2 4*C
            n = n.view(B, -1, 4)  # B H/2*W/2 4*C
            n, _ = torch.max(n, 2, keepdim=True)
        return x, n


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., depths_y=None, dim_y=None, mix_depth=None, mix_heads=None,
                 drop_path=0., drop_path_y=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.depths_y = depths_y
        self.use_checkpoint = use_checkpoint
        self.mix_depth = mix_depth
        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)])
        self.blocks_y = nn.ModuleList([
            SwinTransformerBlock(dim=dim_y, input_resolution=input_resolution,
                                 num_heads=mix_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path_y[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(depths_y)])
        self.mixblocks = nn.ModuleList([
            MixTransformerBlock(dim=dim, dim_y=dim_y, input_resolution=input_resolution,
                                 num_heads=num_heads, mix_heads=mix_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(mix_depth)])
        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
            self.downsample_y = downsample(input_resolution, dim=dim_y, norm_layer=norm_layer)
        else:
            self.downsample = None
            self.downsample_y = None

    def forward(self, x, y, nuclei_mask):
        for blk in self.blocks:
            if self.use_checkpoint:
                x, nuclei_mask = checkpoint.checkpoint(blk, x, nuclei_mask)
            else:
                x, nuclei_mask = blk(x, nuclei_mask)
        for blk_y in self.blocks_y:
            if self.use_checkpoint:
                y, nuclei_mask = checkpoint.checkpoint(blk_y, y, nuclei_mask)
            else:
                y, nuclei_mask = blk_y(y, nuclei_mask)
        for mixblk in self.mixblocks:
            if self.use_checkpoint:
                x, y, nuclei_mask = checkpoint.checkpoint(mixblk, x, y, nuclei_mask)
            else:
                x, y, nuclei_mask = mixblk(x, y, nuclei_mask)
        nuclei_mask_org = nuclei_mask
        if self.downsample is not None:
            x, nuclei_mask = self.downsample(x, nuclei_mask)
        if self.depths_y != 0 and self.downsample_y is not None:
            y, _ = self.downsample_y(y, nuclei_mask_org)

        return x, y, nuclei_mask


class Nuclei_swin(nn.Module):
    r""" Swin Transformer
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030

    Args:
        img_size (int | tuple(int)): Input image size. Default 224 不需要修改
        patch_size (int | tuple(int)): Patch size. Default: 4 这里应该默认为 1，具体到每个细胞核
        in_chans (int): Number of input image channels. Default: 3 这里是24或24+2（位置编码）
        num_classes (int): Number of classes for classification head. Default: 1000 我们输出的是risk 这里需要大改动，看学长代码源码
        embed_dim (int): Patch embedding dimension. Default: 96 一个patch一个像素所以是24或24+2（pixel=patch）
        depths (tuple(int)): Depth of each Swin Transformer layer. 同样是4 stage，可以先保持不变
        num_heads (tuple(int)): Number of attention heads in different layers. 先保持不变试试
        window_size (int): Window size. Default: 7 先保持不变
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4 MLP每层与embedding的比例，暂时不变
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True 建议添加，大小可能需要修改
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None 不知道什么意思，不设置先
        drop_rate (float): Dropout rate. Default: 0 默认就好
        attn_drop_rate (float): Attention dropout rate. Default: 0 默认就好
        drop_path_rate (float): Stochastic depth rate. Default: 0.1 默认就好
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm. 默认就好
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False 或许需要，先不用试试
        patch_norm (bool): If True, add normalization after patch embedding. Default: True 默认就好
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False 默认就好
    """

    def __init__(self,
                 embed_dim=12, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None, depths_y=None, dim_y=None, mix_depth=None, mix_heads=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, **kwargs):
        super().__init__()

        self.num_layers = len(depths)
        self.depth_y = depths_y
        self.depths = depths
        self.embed_dim = embed_dim
        self.embed_dim_y = []
        self.temp_dim = dim_y//2
        for i in range(len(depths_y)):
            if depths_y[i] != 0:
                self.temp_dim = self.temp_dim*2
                self.embed_dim_y.append(self.temp_dim)
            else:
                self.temp_dim = self.temp_dim
                self.embed_dim_y.append(self.temp_dim)
        self.ape = ape
        self.patch_norm = patch_norm
        if max(depths) != 0:
            self.num_features = 1024 + int(embed_dim * 2 ** (self.num_layers - 1))  # **是幂运算，这里指的是最后输出的特征维数，不建议变
        else:
            self.num_features = 1024
        self.num_features_x = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio
        num_patches = 50176
        patches_resolution = [32, 32]
        # self.patches_resolution = patches_resolution

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        dpr_y = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths_y))]
        # build layers
        self.layers = nn.ModuleList()
        self.scorenet = scorer()
        # self.ptopknet = PerturbedTopK(k=5)
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                               dim_y=self.embed_dim_y[i_layer],
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               depths_y=depths_y[i_layer],
                               mix_depth=mix_depth[i_layer],
                               num_heads=num_heads[i_layer],
                               mix_heads=mix_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               drop_path_y=dpr_y[sum(depths_y[:i_layer]):sum(depths_y[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                               use_checkpoint=use_checkpoint)
            #一个basiclayer/layer
            self.layers.append(layer)

        self.norm = norm_layer(self.num_features)
        # self.norm_x = norm_layer(self.num_features_x)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.avgpool2 = nn.AdaptiveAvgPool1d(1)
        # self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        head = []
        head.append(nn.Linear(self.num_features, self.num_features//2))
        head.append(nn.Linear(self.num_features // 2, self.num_features // 4))
        head.append(nn.Linear(self.num_features // 4, 1))
        self.head = nn.Sequential(*head)
        # head_x = []
        # head_x.append(nn.Linear(self.num_features_x, self.num_features_x // 2))
        # head_x.append(nn.Linear(self.num_features_x // 2, self.num_features_x // 4))
        # head_x.append(nn.Linear(self.num_features_x // 4, 1))
        self.head = nn.Sequential(*head)
        # self.head_x = nn.Sequential(*head_x)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_features(self, x, y, nuclei_mask):
        B, W, H, C = x.shape
        B_y, W_y, H_y, C_y = y.shape
        tempk_x = x
        tempk_y = y
        tempk_m = nuclei_mask

        k_y = y
        k_y = k_y.view(B_y, W_y, H_y, C_y)
        k_y = k_y.permute(0, 3, 1, 2)
        scores = self.scorenet(k_y)
        scores = scores.squeeze()
        B_s, W_s, H_s = scores.shape
        scores = scores.view(B_s, W_s*H_s)
        ptopknet = PerturbedTopK(k=5)
        indices = ptopknet(scores)
        sorted, true_indice = torch.sort(indices, descending=True)
        true_indice = true_indice[:, :, 0].squeeze()
        # print(true_indice)
        b_patch = torch.empty(1, 5, 4, 4, 512).cuda()
        for i in range(B):
            patch = torch.empty(1, 4, 4, 512).cuda()
            for j in range(5):
                k_indice = true_indice[i, j]
                w_indice = k_indice % 7
                h_indice = k_indice // 7
                ele_y = tempk_y[i, 4*h_indice:4*h_indice+4, 4*w_indice:4*w_indice+4, :].unsqueeze(0)
                patch = torch.cat((patch, ele_y), 0)
            patch = patch[1:, :].unsqueeze(0)
            b_patch = torch.cat((b_patch, patch), 0)
        b_patch = b_patch[1:, :, :, :, :]
        b_nuclei = torch.empty(1, 5, 32, 32, 18).cuda()
        for i in range(B):
            patch_x = torch.empty(1, 32, 32, 18).cuda()
            for j in range(5):
                k_indice = true_indice[i, j]
                w_indice = k_indice % 7
                h_indice = k_indice // 7
                ele_x = tempk_x[i, 32*h_indice:32*h_indice+32, 32*w_indice:32*w_indice+32, :].unsqueeze(0)
                patch_x = torch.cat((patch_x, ele_x), 0)
            patch_x = patch_x[1:, :].unsqueeze(0)
            b_nuclei = torch.cat((b_nuclei, patch_x), 0)
        b_nuclei = b_nuclei[1:, :, :, :, :]
        b_musk = torch.empty(1, 5, 32, 32, 1).cuda()
        for i in range(B):
            patch_m = torch.empty(1, 32, 32, 1).cuda()
            for j in range(5):
                k_indice = true_indice[i, j]
                w_indice = k_indice % 7
                h_indice = k_indice // 7
                ele_m = tempk_m[i, 32 * h_indice:32 * h_indice + 32, 32 * w_indice:32 * w_indice + 32, :].unsqueeze(0)
                patch_m = torch.cat((patch_m, ele_m), 0)
            patch_m = patch_m[1:, :].unsqueeze(0)
            b_musk = torch.cat((b_musk, patch_m), 0)
        b_musk = b_musk[1:, :, :, :, :]
        # end here
        b_patch = b_patch.view(B*5, 16, 512)
        b_nuclei = b_nuclei.view(B*5, 1024, 18)
        b_musk = b_musk.view(B * 5, 1024, 1)
        x = b_nuclei
        y = b_patch
        nuclei_mask = b_musk
        if self.ape:
            x = x + self.absolute_pos_embed
            y = y + self.absolute_pos_embed
        x = self.pos_drop(x)
        y = self.pos_drop(y)
        for layer in self.layers:
            x, y, nuclei_mask = layer(x, y, nuclei_mask)  #
        x = torch.cat((x, y), 2)
        x = self.norm(x)  # B L C
        # if max(self.depth_y) != 0:
        #     if max(self.depths) == 0:
        #         x = self.norm(y)
        #     else:
        #         x = torch.cat((x, y), 2)
        #         x = self.norm(x)  # B L C
        # else:
        #     x = self.norm_x(x)
        x = self.avgpool(x.transpose(1, 2))  # B C 1
        x = torch.flatten(x, 1)
        x = x.view(B, 5, -1)
        x = self.avgpool2(x.transpose(1, 2))
        x = torch.flatten(x, 1)
        return x

    def forward(self, x, y, nuclei_mask):
        x = self.forward_features(x, y, nuclei_mask)
        x = self.head(x)
        # if max(self.depth_y) != 0:
        #     x = self.head(x)
        # else:
        #     x = self.head_x(x)
        return x
