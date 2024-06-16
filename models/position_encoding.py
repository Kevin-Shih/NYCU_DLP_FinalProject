import torch.nn as nn
import torch
import math


class PositionEncodingSuperGule(nn.Module):
    def __init__(self,d_model):
        super().__init__()
        self.d_model=d_model
        self.kenc = KeypointEncoder(d_model, [32, 64])

    def forward(self,x):
        # x : N,C,H,W
        y_position = torch.ones((x.shape[2], x.shape[3])).cumsum(0).float().unsqueeze(0).to(x)
        x_position = torch.ones((x.shape[2], x.shape[3])).cumsum(1).float().unsqueeze(0).to(x)
        xy_position = torch.cat([x_position, y_position]) - 1
        xy_position = xy_position.view(2, -1).permute(1, 0).repeat(x.shape[0], 1, 1)
        xy_position_n = normalize_keypoints(xy_position, x.shape)
        ret = x + self.kenc(xy_position_n).view(x.shape)
        return ret


class PositionEncodingSine(nn.Module):
    """
    This is a sinusoidal position encoding that generalized to 2-dimensional images
    """

    def __init__(self, d_model, max_shape=(600, 600), temp_bug_fix=True):
        """
        Args:
            max_shape (tuple): for 1/4 featmap, the max length of 600 corresponds to 2400 pixels
            temp_bug_fix (bool): As noted in this [issue](https://github.com/zju3dv/LoFTR/issues/41),
                the original implementation of LoFTR includes a bug in the pos-enc impl, which has little impact
                on the final performance. For now, we keep both impls for backward compatability.
                We will remove the buggy impl after re-training all variants of our released models.
        """
        super().__init__()

        pe = torch.zeros((d_model, *max_shape))
        y_position = torch.ones(max_shape).cumsum(0).float().unsqueeze(0)
        x_position = torch.ones(max_shape).cumsum(1).float().unsqueeze(0)
        if temp_bug_fix:
            div_term = torch.exp(torch.arange(0, d_model//2, 2).float() * (-math.log(10000.0) / (d_model//2)))
        else:  # a buggy implementation (for backward compatability only)
            div_term = torch.exp(torch.arange(0, d_model//2, 2).float() * (-math.log(10000.0) / d_model//2))
        div_term = div_term[:, None, None]  # [C//4, 1, 1]
        pe[0::4, :, :] = torch.sin(x_position * div_term)
        pe[1::4, :, :] = torch.cos(x_position * div_term)
        pe[2::4, :, :] = torch.sin(y_position * div_term)
        pe[3::4, :, :] = torch.cos(y_position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0), persistent=False)  # [1, C, H, W]
        # self.register_buffer('pe11', pe.unsqueeze(0))  # [1, C, H, W]

    def forward(self, x):
        """
        Args:
            x: [N, C, H, W]
        """
        return x + self.pe[:, :, :x.size(2), :x.size(3)]


def MLP(channels: list, do_bn=True):
    """ Multi-layer perceptron """
    n = len(channels)
    layers = []
    for i in range(1, n):
        layers.append(
            nn.Conv1d(channels[i - 1], channels[i], kernel_size=1, bias=True))
        if i < (n-1):
            if do_bn:
                layers.append(nn.BatchNorm1d(channels[i]))
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


def normalize_keypoints(kpts, image_shape):
    """ Normalize keypoints locations based on image image_shape"""
    _, _, height, width = image_shape
    one = kpts.new_tensor(1)
    size = torch.stack([one*width, one*height])[None]
    center = size / 2
    scaling = size.max(1, keepdim=True).values * 0.7
    return (kpts - center[:, None, :]) / scaling[:, None, :]


class KeypointEncoder(nn.Module):
    """ Joint encoding of visual appearance and location using MLPs"""
    def __init__(self, feature_dim, layers):
        super().__init__()

        self.encoder = MLP([2] + layers + [feature_dim])
        nn.init.constant_(self.encoder[-1].bias, 0.0)

    def forward(self, kpts):
        inputs = kpts.transpose(1, 2)
        return self.encoder(inputs)


def get_proposal_pos_embed(proposals, hidden_dim):
    assert hidden_dim % proposals.shape[-1] == 0
    num_pos_feats = int(hidden_dim / proposals.shape[-1])
    temperature = 10000
    scale = 2 * math.pi

    dim_t = torch.arange(num_pos_feats, dtype=proposals.dtype, device=proposals.device)
    dim_t = temperature ** (2 * (dim_t.div(2, rounding_mode="floor")) / num_pos_feats)
    proposals = proposals * scale
    proposals = proposals.unbind(-1)

    pos = []
    for proposal in proposals:
        proposal = proposal[..., None] / dim_t
        proposal = torch.stack(
            (proposal[..., 0::2].sin(), proposal[..., 1::2].cos()), dim=-1
        ).flatten(-2)
        pos.append(proposal)
    pos = torch.cat(pos, dim=-1)

    return pos

class FixedBoxEmbedding(nn.Module): 
    def __init__(self, hidden_dim, temperature=10000, normalize=False):
        super(FixedBoxEmbedding, self).__init__()
        self.hidden_dim = hidden_dim
        self.temperature = temperature
        self.normalize = normalize

    def forward(self, x, mask=None, ref_size=4) -> torch.Tensor:
        eps = 1e-6
        if mask is not None:
            not_mask = ~mask
            y_embed = not_mask.cumsum(1, dtype=x.dtype)
            x_embed = not_mask.cumsum(2, dtype=x.dtype)

            size_h = not_mask[:, :, 0].sum(dim=-1, dtype=x.dtype)
            size_w = not_mask[:, 0, :].sum(dim=-1, dtype=x.dtype)
        else:
            size_h, size_w = x.shape[-2:]
            y_embed = torch.arange(1, size_h + 1, dtype=x.dtype, device=x.device)
            x_embed = torch.arange(1, size_w + 1, dtype=x.dtype, device=x.device)
            # y_embed, x_embed = torch.meshgrid(y_embed, x_embed, indexing="ij")
            # omit indexing to suit torch 1.9.1
            y_embed, x_embed = torch.meshgrid(y_embed, x_embed)
            x_embed = x_embed.unsqueeze(0).repeat(x.shape[0], 1, 1)
            y_embed = y_embed.unsqueeze(0).repeat(x.shape[0], 1, 1)

            size_h = torch.tensor([size_h] * x.shape[0], dtype=x.dtype, device=x.device)
            size_w = torch.tensor([size_w] * x.shape[0], dtype=x.dtype, device=x.device)

        if self.normalize:
            y_embed = (y_embed - 0.5) / (y_embed[:, -1:, :] + eps)
            x_embed = (x_embed - 0.5) / (x_embed[:, :, -1:] + eps)

        h_embed = ref_size / size_h
        w_embed = ref_size / size_w

        h_embed = h_embed.unsqueeze(1).unsqueeze(2).expand_as(x_embed)
        w_embed = w_embed.unsqueeze(1).unsqueeze(2).expand_as(x_embed)

        center_embed = torch.stack([x_embed, y_embed], dim=-1)
        size_embed = torch.stack([w_embed, h_embed], dim=-1)
        center = get_proposal_pos_embed(center_embed, self.hidden_dim)
        size = get_proposal_pos_embed(size_embed, self.hidden_dim)
        box = center + size

        return box.permute(0, 3, 1, 2)
