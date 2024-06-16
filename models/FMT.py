import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from .utils import flatten_with_shape, split_with_shape

'''
- We provide two different positional encoding methods as shown below.
- You can easily switch different pos-enc in the __init__() function of FMT.
- In our experiments, PositionEncodingSuperGule usually cost more GPU memory.
'''
from .position_encoding import PositionEncodingSuperGule, PositionEncodingSine, FixedBoxEmbedding
from e2edet.module.ops import BoxAttnFunction, InstanceAttnFunction

class LinearAttention(nn.Module):
    def __init__(self, eps=1e-6):
        super(LinearAttention, self).__init__()
        self.feature_map = lambda x: torch.nn.functional.elu(x) + 1
        self.eps = eps

    def forward(self, queries, keys, values):
        Q = self.feature_map(queries)
        K = self.feature_map(keys)

        # Compute the KV matrix, namely the dot product of keys and values so
        # that we never explicitly compute the attention matrix and thus
        # decrease the complexity
        KV = torch.einsum("nshd,nshm->nhmd", K, values)

        # Compute the normalizer
        Z = 1/(torch.einsum("nlhd,nhd->nlh", Q, K.sum(dim=1))+self.eps)

        # Finally compute and return the new values
        V = torch.einsum("nlhd,nhmd,nlh->nlhm", Q, KV, Z)

        return V.contiguous()

class BoxAttentionLayer(nn.Module):
    def __init__(self, d_model, num_level, num_head, kernel_size=2):
        super(BoxAttentionLayer, self).__init__()
        assert d_model % num_head == 0, "d_model should be divided by num_head"

        self.im2col_step = 64
        self.d_model = d_model
        self.num_head = num_head
        self.num_level = num_level
        self.head_dim = d_model // num_head
        self.kernel_size = kernel_size
        self.num_point = kernel_size ** 2

        self.linear_box_weight = nn.Parameter(
            torch.zeros(num_level * num_head * 4, d_model)
        )
        self.linear_box_bias = nn.Parameter(torch.zeros(num_head * num_level * 4))

        self.linear_attn_weight = nn.Parameter(
            torch.zeros(num_head * num_level * self.num_point, d_model)
        )
        self.linear_attn_bias = nn.Parameter(
            torch.zeros(num_head * num_level * self.num_point)
        )

        self.value_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self._create_kernel_indices(kernel_size, "kernel_indices")
        self._reset_parameters()

    def _create_kernel_indices(self, kernel_size, module_name):
        if kernel_size % 2 == 0:
            start_idx = -kernel_size // 2
            end_idx = kernel_size // 2

            indices = torch.linspace(start_idx + 0.5, end_idx - 0.5, kernel_size)
        else:
            start_idx = -(kernel_size - 1) // 2
            end_idx = (kernel_size - 1) // 2

            indices = torch.linspace(start_idx, end_idx, kernel_size)
        i, j = torch.meshgrid(indices, indices, indexing="ij")
        # omit indexing to suit torch 1.9.1
        # i, j = torch.meshgrid(indices, indices)
        kernel_indices = torch.stack([j, i], dim=-1).view(-1, 2) / self.kernel_size
        self.register_buffer(module_name, kernel_indices)

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.constant_(self.out_proj.bias, 0.0)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.constant_(self.value_proj.bias, 0.0)
        nn.init.constant_(self.linear_attn_weight, 0.0)
        nn.init.constant_(self.linear_attn_bias, 0.0)
        nn.init.constant_(self.linear_box_weight, 0.0)
        nn.init.uniform_(self.linear_box_bias)

    def _where_to_attend(self, query, v_valid_ratios, ref_windows):
        b, l = ref_windows.shape[:2]

        offset_boxes = F.linear(query, self.linear_box_weight, self.linear_box_bias)
        offset_boxes = offset_boxes.view(b, l, self.num_head, self.num_level, 4)

        if ref_windows.dim() == 3:
            ref_windows = ref_windows.unsqueeze(2).unsqueeze(3)
        else:
            ref_windows = ref_windows.unsqueeze(3)

        boxes = ref_windows + offset_boxes / 8 * ref_windows[..., [2, 3, 2, 3]]
        center, size = boxes.unsqueeze(-2).split(2, dim=-1)

        grid = center + self.kernel_indices * torch.relu(size)
        if v_valid_ratios is not None:
            grid = grid * v_valid_ratios

        return grid.contiguous()

    def forward(
        self, query, value, v_shape, v_mask, v_start_index, v_valid_ratios, ref_windows
    ):
        b, l1 = query.shape[:2]
        l2 = value.shape[1]

        value = self.value_proj(value)
        if v_mask is not None:
            value = value.masked_fill(v_mask[..., None], float(0))
        value = value.view(b, l2, self.num_head, self.head_dim)

        attn_weights = F.linear(query, self.linear_attn_weight, self.linear_attn_bias)
        attn_weights = F.softmax(attn_weights.view(b, l1, self.num_head, -1), dim=-1)
        attn_weights = attn_weights.view(
            b, l1, self.num_head, self.num_level, self.kernel_size, self.kernel_size
        )

        sampled_grid = self._where_to_attend(query, v_valid_ratios, ref_windows)
        output = BoxAttnFunction.apply(
            value, v_shape, v_start_index, sampled_grid, attn_weights, self.im2col_step
        )
        output = self.out_proj(output)

        return output

class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(AttentionLayer, self).__init__()

        # Fill d_keys and d_values
        d_keys = d_keys or (d_model//n_heads)
        d_values = d_values or (d_model//n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values):
        # Extract the dimensions into local variables
        N, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        # Project the queries/keys/values
        queries = self.query_projection(queries).view(N, L, H, -1)
        keys = self.key_projection(keys).view(N, S, H, -1)
        values = self.value_projection(values).view(N, S, H, -1)

        # Compute the attention
        new_values = self.inner_attention(
            queries,
            keys,
            values,
        ).view(N, L, -1)

        # Project the output and return
        return self.out_projection(new_values)


class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_keys=None, d_values=None, d_ff=None, dropout=0.0,
                 activation="relu"):
        super(EncoderLayer, self).__init__()

        d_keys = d_keys or (d_model//n_heads)
        inner_attention = LinearAttention()
        attention = AttentionLayer(inner_attention, d_model, n_heads, d_keys, d_values)

        d_ff = d_ff or 2 * d_model
        self.attention = attention
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = getattr(F, activation)

    def forward(self, x, source):
        # Normalize the masks
        N = x.shape[0]
        L = x.shape[1]

        # Run self attention and add it to the input
        x = x + self.dropout(self.attention(
            x, source, source,
        ))

        # Run the fully connected part of the layer
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.linear1(y)))
        y = self.dropout(self.linear2(y))

        return self.norm2(x+y)

class BoxEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_keys=None, d_values=None, d_ff=None, dropout=0.0,
                 activation="relu"):
        super(BoxEncoderLayer, self).__init__()
        attention = BoxAttentionLayer(d_model, num_level= 3, num_head= n_heads)

        d_ff = d_ff or 2 * d_model
        self.attention = attention
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = getattr(F, activation)

    # x(query): w/ pos encoding
    # source: w/o pos encoding
    def forward(self, x, source, src_shape, src_start_index, ref_windows):
        # Normalize the masks
        N = x.shape[0]
        L = x.shape[1]

        # Run self attention and add it to the input
        # query, value, v_shape, v_mask, v_start_index, v_valid_ratios, ref_windows
        x = x + self.dropout(self.attention(
            x, source, src_shape, None, src_start_index, None, ref_windows
        )[0])

        # Run the fully connected part of the layer
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.linear1(y)))
        y = self.dropout(self.linear2(y))

        return self.norm2(x+y)

class FMT(nn.Module):
    def __init__(self, config):
        super(FMT, self).__init__()

        self.d_model = config['d_model']
        self.nhead = config['nhead']
        self.layer_names = config['layer_names']
        encoder_layer = EncoderLayer(config['d_model'], config['nhead'])
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(len(self.layer_names))])
        self._reset_parameters()

        # self.pos_encoding = PositionEncodingSuperGule(config['d_model'])
        self.pos_encoding = PositionEncodingSine(config['d_model'])

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, ref_feature=None, src_feature=None, feat="ref"):
        """
        Args:
            ref_feature(torch.Tensor): [N, C, H, W]
            src_feature(torch.Tensor): [N, C, H, W]
        """

        assert ref_feature is not None

        if feat == "ref": # only self attention layer

            assert self.d_model == ref_feature.size(1)
            _, _, H, _ = ref_feature.shape

            ref_feature = einops.rearrange(self.pos_encoding(ref_feature), 'n c h w -> n (h w) c')

            ref_feature_list = []
            for layer, name in zip(self.layers, self.layer_names): # every self attention layer
                if name == 'self':
                    ref_feature = layer(ref_feature, ref_feature)
                    ref_feature_list.append(einops.rearrange(ref_feature, 'n (h w) c -> n c h w', h=H))
            return ref_feature_list

        elif feat == "src":

            assert self.d_model == ref_feature[0].size(1)
            _, _, H, _ = ref_feature[0].shape

            ref_feature = [einops.rearrange(_, 'n c h w -> n (h w) c') for _ in ref_feature]

            src_feature = einops.rearrange(self.pos_encoding(src_feature), 'n c h w -> n (h w) c')

            for i, (layer, name) in enumerate(zip(self.layers, self.layer_names)):
                if name == 'self':
                    src_feature = layer(src_feature, src_feature)
                elif name == 'cross':
                    src_feature = layer(src_feature, ref_feature[i // 2])
                else:
                    raise KeyError
            return einops.rearrange(src_feature, 'n (h w) c -> n c h w', h=H)
        else:
            raise ValueError("Wrong feature name")

class BoxFMT(nn.Module):
    def __init__(self, config, ):
        super(BoxFMT, self).__init__()

        self.d_model = config['d_model']
        self.nhead = config['nhead']
        self.layer_names = config['layer_names']
        encoder_layer = BoxEncoderLayer(config['d_model'], config['nhead'])
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(len(self.layer_names))])
        self._reset_parameters()
        self.ref_size = config['ref_size']
        self.ref_shape = None
        self.ref_start_index = None
        self.ref_windows = None

        # self.pos_encoding = PositionEncodingSuperGule(config['d_model'])
        # self.pos_encoding = PositionEncodingSine(config['d_model'])

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _create_ref_windows(self, tensor_list, mask_list):
        ref_windows = []

        eps = 1e-6
        for i, tensor in enumerate(tensor_list):
            if mask_list is not None:
                not_mask = ~(mask_list[i])
                y_embed = not_mask.cumsum(1, dtype=tensor.dtype)
                x_embed = not_mask.cumsum(2, dtype=tensor.dtype)

                size_h = not_mask[:, :, 0].sum(dim=-1, dtype=tensor.dtype)
                size_w = not_mask[:, 0, :].sum(dim=-1, dtype=tensor.dtype)
            else:
                size_h, size_w = tensor.shape[-2:]
                y_embed = torch.arange(
                    1, size_h + 1, dtype=tensor.dtype, device=tensor.device
                )
                x_embed = torch.arange(
                    1, size_w + 1, dtype=tensor.dtype, device=tensor.device
                )
                y_embed, x_embed = torch.meshgrid(y_embed, x_embed, indexing="ij")
                # omit indexing to suit torch 1.9.1
                # y_embed, x_embed = torch.meshgrid(y_embed, x_embed)
                x_embed = x_embed.unsqueeze(0).repeat(tensor.shape[0], 1, 1)
                y_embed = y_embed.unsqueeze(0).repeat(tensor.shape[0], 1, 1)

                size_h = torch.tensor(
                    [size_h] * tensor.shape[0], dtype=tensor.dtype, device=tensor.device
                )
                size_w = torch.tensor(
                    [size_w] * tensor.shape[0], dtype=tensor.dtype, device=tensor.device
                )

            y_embed = (y_embed - 0.5) / (y_embed[:, -1:, :] + eps)
            x_embed = (x_embed - 0.5) / (x_embed[:, :, -1:] + eps)
            center = torch.stack([x_embed, y_embed], dim=-1).flatten(1, 2)

            h_embed = self.ref_size / size_h
            w_embed = self.ref_size / size_w

            size = torch.stack([w_embed, h_embed], dim=-1)
            size = size.unsqueeze(1).expand_as(center)

            ref_box = torch.cat([center, size], dim=-1)
            ref_windows.append(ref_box)

        ref_windows = torch.cat(ref_windows, dim=1)

        return ref_windows

    def forward(self, ref_feature=None, src_feature=None, pos= None, feat="ref"):
        """
        Args:
            ref_feature(torch.Tensor): [N, C, H, W] 
            src_feature(torch.Tensor): [N, C, H, W]
        """
        
        assert ref_feature is not None       
        
        # the shape of ref. and src. views images should be same, so ref_windows can share
        # ref_windows = self._create_ref_windows(ref_feature, None)

        # pre-processing for ref. image in box-attention
        # should be "ref_feature" instead of "ref" to forward through 4 layers
        # ref_feature: [(B, L_1, C), ..., (B, L_T, C)] Note. L_1~L_T is  the flatten length of that level(stage) i.e. L_1 = H_1*W_1
        # ->(B, L_1 ... L_T, C),  (B, C)
        
        # ref, _, ref_shape = flatten_with_shape(ref_feature, None)
        

        if feat == "ref": # only self attention layer
            # print(self.d_model, len(ref_feature), ref_feature[0].size())
            assert self.d_model == ref_feature[0].size(1)
            _, _, H, _ = ref_feature[0].shape
            # ref_feature = einops.rearrange(ref_feature, 'n c h w -> n (h w) c')
            self.ref_windows = self._create_ref_windows(ref_feature, None)
            ref_feature, _, self.ref_shape = flatten_with_shape(ref_feature, None)
            self.ref_start_index = torch.cat([self.ref_shape.new_zeros(1), self.ref_shape.prod(1).cumsum(0)[:-1]])
            
            ref_feature_list = []
            for layer, name in zip(self.layers, self.layer_names):  # every self attention layer
                layer:BoxEncoderLayer
                if name == 'self':
                    ref_feature = layer(ref_feature + pos,          # x(query): (flatten ref_feature/last layer output) + pos encoding
                                        ref_feature,                # source(value): ref_feature as value w/o pos encoding
                                        self.ref_shape,             # key: defined by ref. windows and value i.e. values from box of interest 
                                        self.ref_start_index,
                                        self.ref_windows
                                        ) # (x, source, src_shape, src_start_index, ref_windows)
                    ref_feature, _ = split_with_shape(ref_feature, None, self.ref_shape)
                    ref_feature = list(ref_feature)
                    for i in range(len(ref_feature)):
                        ref_feature[i] = einops.rearrange(ref_feature[i], 'n (h w) c -> n c h w', h=H<<i)
                    ref_feature_list.append(ref_feature) # ref_feature_list: [Layer, Level_t(stage_t), (B, C, H, W)]
                    ref_feature, _, _ = flatten_with_shape(ref_feature, None)
                    # ref_feature_list.append(einops.rearrange(ref_feature, 'n (h w) c -> n c h w', h=H))
            return ref_feature_list

        elif feat == "src":
            # pre-processing for box-attention
            #### not sure how to define q, k, v ####
            # should be "src_feature" instead of "src" to forward through 4 layers
            # src_feature: [(B, L_1, C), ..., (B, L_T, C)] Note. L_1~L_T is  the flatten length of that level(stage) i.e. L_1 = H_1*W_1
            _, _, H, _ = src_feature[0].shape
            # _, _, H, _ = ref_feature[0][0].shape
            src_feature, _, src_shape = flatten_with_shape(src_feature, None)
            src_start_index = torch.cat([src_shape.new_zeros(1), src_shape.prod(1).cumsum(0)[:-1]])
            
            assert self.d_model == ref_feature[0][0].size(1)
            # ref_feature = [einops.rearrange(_, 'n c h w -> n (h w) c') for _ in ref_feature]
            # src_feature = einops.rearrange(self.pos_encoding(src_feature), 'n c h w -> n (h w) c')

            for i, (layer, name) in enumerate(zip(self.layers, self.layer_names)):
                if name == 'self':
                    src_feature = layer(src_feature + pos,      # x(query): (flatten src_feature/last layer output) + pos encoding
                                        src_feature,            # source(value): src_feature as value w/o pos encoding
                                        src_shape,              # key: defined by ref. windows and value i.e. values from box of interest 
                                        src_start_index,
                                        self.ref_windows
                                        ) # (x, source, src_shape, src_start_index, ref_windows)
                elif name == 'cross':
                    layered_ref_feature, _, _= flatten_with_shape(ref_feature[i // 2], None)
                    src_feature = layer(src_feature + pos,      # x(query): (flatten src_feature/last layer output) + pos encoding
                                        layered_ref_feature,    # source(value): ref_feature_list[stage_t] as value w/o pos encoding, [stages_t, (B, L_t, C)]
                                        self.ref_shape,              # key: defined by ref. windows and value i.e. values from box of interest 
                                        self.ref_start_index,
                                        self.ref_windows
                                        ) # (x, source, src_shape, src_start_index, ref_windows
                else:
                    raise KeyError
           
            src_feature, _ = split_with_shape(src_feature, None, src_shape)
            src_feature = list(src_feature)[0]
            src_feature = einops.rearrange(src_feature, 'n (h w) c -> n c h w', h=H)
            return src_feature
        else:
            raise ValueError("Wrong feature name")

class FMT_with_pathway(nn.Module):
    def __init__(self,
            base_channels=8,
            FMT_config={
                'd_model': 32,
                'nhead': 8,
                'layer_names': ['self', 'cross'] * 4}):

        super(FMT_with_pathway, self).__init__()

        self.FMT = FMT(FMT_config)

        self.dim_reduction_1 = nn.Conv2d(base_channels * 4, base_channels * 2, 1, bias=False)
        self.dim_reduction_2 = nn.Conv2d(base_channels * 2, base_channels * 1, 1, bias=False)

        self.smooth_1 = nn.Conv2d(base_channels * 2, base_channels * 2, 3, padding=1, bias=False)
        self.smooth_2 = nn.Conv2d(base_channels * 1, base_channels * 1, 3, padding=1, bias=False)

    def _upsample_add(self, x, y):
        """_upsample_add. Upsample and add two feature maps.

        :param x: top feature map to be upsampled.
        :param y: lateral feature map.
        """

        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True) + y


    def forward(self, features):
        """forward.

        :param features: multiple views and multiple stages features
        """

        for nview_idx, feature_multi_stages in enumerate(features):
            if nview_idx == 0: # ref view
                ref_fea_t_list = self.FMT(feature_multi_stages["stage1"].clone(), feat="ref")
                feature_multi_stages["stage1"] = ref_fea_t_list[-1]
                feature_multi_stages["stage2"] = self.smooth_1(self._upsample_add(self.dim_reduction_1(feature_multi_stages["stage1"]), feature_multi_stages["stage2"]))
                feature_multi_stages["stage3"] = self.smooth_2(self._upsample_add(self.dim_reduction_2(feature_multi_stages["stage2"]), feature_multi_stages["stage3"]))

            else: # src view
                feature_multi_stages["stage1"] = self.FMT([_.clone() for _ in ref_fea_t_list], feature_multi_stages["stage1"].clone(), feat="src")
                feature_multi_stages["stage2"] = self.smooth_1(self._upsample_add(self.dim_reduction_1(feature_multi_stages["stage1"]), feature_multi_stages["stage2"]))
                feature_multi_stages["stage3"] = self.smooth_2(self._upsample_add(self.dim_reduction_2(feature_multi_stages["stage2"]), feature_multi_stages["stage3"]))

        return features


class BoxFMT_with_pathway(nn.Module):
    def __init__(self,
            base_channels=8,
            FMT_config={
                'd_model': 32,
                'nhead': 8,
                'layer_names': ['self', 'cross'] * 4,
                'ref_size': 4}, 
            ):
        super(BoxFMT_with_pathway, self).__init__()

        self.BoxFMT = BoxFMT(FMT_config, )

        self.dim_reduction_1 = nn.Conv2d(base_channels * 4, base_channels * 2, 1, bias=False)
        self.dim_reduction_2 = nn.Conv2d(base_channels * 2, base_channels * 1, 1, bias=False)

        self.smooth_1 = nn.Conv2d(base_channels * 4, base_channels * 2, 3, padding=1, bias=False)
        self.smooth_2 = nn.Conv2d(base_channels * 2, base_channels * 1, 3, padding=1, bias=False)
        self.pos_encoding = FixedBoxEmbedding(FMT_config['d_model'], normalize=True)
        self.ref_size = FMT_config['ref_size']

    def _upsample_add(self, x, y):
        """_upsample_add. Upsample and add two feature maps.

        :param x: top feature map to be upsampled.
        :param y: lateral feature map.
        """

        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True) + y


    def forward(self, features):
        """forward.

        :param features: multiple views and multiple stages features
        """
        for nview_idx, feature_multi_stages in enumerate(features):
            """
            Append every stage feature to multi_stage_features
            Mask will be ignored
            """
            multi_stage_features = []
            pos_encodings = []
            # dict -> list
            for i, (_, feature) in enumerate(feature_multi_stages.items()):
                multi_stage_features.append(feature.clone())
                pos_encodings.append(self.pos_encoding(feature, None, self.ref_size).type_as(feature))
            src_pos = []
            if pos_encodings[0] is not None:
                for pe in pos_encodings[0:1]:
                    b, c = pe.shape[:2]
                    pe = pe.view(b, c, -1).transpose(1, 2)
                    src_pos.append(pe)
                src_pos = torch.cat(src_pos, dim=1)

            # multi_stage_features: should be [stages, (B, C, H, W)]
            # multi_stage_ref_feat_list: should be [4 layers attention, stages, ...]
            if nview_idx == 0: # ref view
                multi_stage_ref_feat_list = self.BoxFMT(multi_stage_features[0:1], pos=src_pos[0:1], feat="ref") # expect output is multi stage
                # print(len(multi_stage_ref_feat_list), len(multi_stage_ref_feat_list[0]), multi_stage_ref_feat_list[0][0].shape)
                feature_multi_stages["stage1"] = multi_stage_ref_feat_list[-1][0] # get last layer, stage1
                # feature_multi_stages["stage2"] = self.smooth_1(self._upsample_add(feature_multi_stages["stage1"], multi_stage_ref_feat_list[-1][1])) # upsample get last layer, stage2
                feature_multi_stages["stage2"] = self.smooth_1(self._upsample_add(feature_multi_stages["stage1"], multi_stage_features[1])) # upsample get last layer, stage2
                # feature_multi_stages["stage3"] = self.smooth_2(self._upsample_add(feature_multi_stages["stage2"], self.dim_reduction_1(multi_stage_ref_feat_list[-1][2]))) # upsample get last layer, stage3
                feature_multi_stages["stage3"] = self.smooth_2(self._upsample_add(feature_multi_stages["stage2"], self.dim_reduction_1(multi_stage_features[2]))) # upsample get last layer, stage3
                
                feature_multi_stages["stage1"] = F.interpolate(feature_multi_stages["stage1"], scale_factor=4, mode='nearest')
                feature_multi_stages["stage2"] = F.interpolate(feature_multi_stages["stage2"], scale_factor=4, mode='nearest')
                feature_multi_stages["stage3"] = F.interpolate(feature_multi_stages["stage3"], scale_factor=4, mode='nearest')

            else: # src view
                # _ enumerate through 4 layers i.e. _: [stages, ...]
                # [_.clone() for _ in multi_stage_ref_feat_list]: [(layer1)[stages, ...], (layer2)[stages, ...], (layer3)[stages, ...], (layer4)[stages, ...]]
                clone_multi_stage_ref_feat_list=[]
                for layer_feat in multi_stage_ref_feat_list:
                    temp = [_.clone() for _ in layer_feat]
                    clone_multi_stage_ref_feat_list.append(temp)
                feature_multi_stages["stage1"] = self.BoxFMT(clone_multi_stage_ref_feat_list, multi_stage_features[0:1], pos=src_pos[0:1], feat="src")
                feature_multi_stages["stage2"] = self.smooth_1(self._upsample_add(feature_multi_stages["stage1"], multi_stage_features[1])) # upsample get last layer, stage2
                feature_multi_stages["stage3"] = self.smooth_2(self._upsample_add(feature_multi_stages["stage2"], self.dim_reduction_1(multi_stage_features[2]))) # upsample get last layer, stage3

                feature_multi_stages["stage1"] = F.interpolate(feature_multi_stages["stage1"], scale_factor=4, mode='nearest')
                feature_multi_stages["stage2"] = F.interpolate(feature_multi_stages["stage2"], scale_factor=4, mode='nearest')
                feature_multi_stages["stage3"] = F.interpolate(feature_multi_stages["stage3"], scale_factor=4, mode='nearest')
        return features