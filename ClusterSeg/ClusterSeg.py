from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import copy
import logging
import math
from os.path import join as pjoin
import torch
import torch.nn as nn
import numpy as np
from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from torch.nn import init
from torch.autograd import Variable
import torch.nn.functional as F
from scipy import ndimage
try:
    from itertools import ifilterfalse
except ImportError:  # py3k
    from itertools import filterfalse as ifilterfalse


# download pretrained vit model from https://console.cloud.google.com/storage/vit_models/
ATTENTION_Q = "MultiHeadDotProductAttention_1/query"
ATTENTION_K = "MultiHeadDotProductAttention_1/key"
ATTENTION_V = "MultiHeadDotProductAttention_1/value"
ATTENTION_OUT = "MultiHeadDotProductAttention_1/out"
FC_0 = "MlpBlock_3/Dense_0"
FC_1 = "MlpBlock_3/Dense_1"
ATTENTION_NORM = "LayerNorm_0"
MLP_NORM = "LayerNorm_2"


def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}


class Attention(nn.Module):
    def __init__(self, config, vis):
        super(Attention, self).__init__()
        self.vis = vis
        self.num_attention_heads = config.transformer["num_heads"]
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(config.hidden_size, self.all_head_size)
        self.key = Linear(config.hidden_size, self.all_head_size)
        self.value = Linear(config.hidden_size, self.all_head_size)

        self.out = Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])

        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights


class Mlp(nn.Module):
    def __init__(self, config):
        super(Mlp, self).__init__()
        self.fc1 = Linear(config.hidden_size, config.transformer["mlp_dim"])
        self.fc2 = Linear(config.transformer["mlp_dim"], config.hidden_size)
        self.act_fn = ACT2FN["gelu"]
        self.dropout = Dropout(config.transformer["dropout_rate"])

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Embeddings(nn.Module):
    def __init__(self, config, in_channels):
        super(Embeddings, self).__init__()
        self.config = config
        img_size = self.config.img_size
        if config.patches.get("grid") is not None:  # ResNet
            grid_size = config.patches["grid"]
            patch_size = (img_size[0] // 16 // grid_size[0], img_size[1] // 16 // grid_size[1])
            patch_size_real = (patch_size[0] * 16, patch_size[1] * 16)
            n_patches = (img_size[0] // patch_size_real[0]) * (img_size[1] // patch_size_real[1])
        else:
            patch_size = _pair(config.patches["size"])
            n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
        self.patch_embeddings = Conv2d(in_channels=in_channels,
                                       out_channels=config.hidden_size,
                                       kernel_size=patch_size,
                                       stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, config.hidden_size))
        self.dropout = Dropout(config.transformer["dropout_rate"])

    def forward(self, x):
        x = self.patch_embeddings(x)  # (B, hidden. n_patches^(1/2), n_patches^(1/2))
        x = x.flatten(2)
        x = x.transpose(-1, -2)  # (B, n_patches, hidden)

        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings


class Block(nn.Module):
    def __init__(self, config, vis):
        super(Block, self).__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = Mlp(config)
        self.attn = Attention(config, vis)

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x, weights

    def load_from(self, weights, n_block):
        ROOT = "Transformer/encoderblock_{}".format(n_block)
        with torch.no_grad():
            query_weight = np2th(weights["{}/{}/kernel".format(ROOT, ATTENTION_Q)]).view(self.hidden_size, self.hidden_size).t()
            key_weight = np2th(weights["{}/{}/kernel".format(ROOT, ATTENTION_K)]).view(self.hidden_size, self.hidden_size).t()
            value_weight = np2th(weights["{}/{}/kernel".format(ROOT, ATTENTION_V)]).view(self.hidden_size, self.hidden_size).t()
            out_weight = np2th(weights["{}/{}/kernel".format(ROOT, ATTENTION_OUT)]).view(self.hidden_size, self.hidden_size).t()

            query_bias = np2th(weights["{}/{}/bias".format(ROOT, ATTENTION_Q)]).view(-1)
            key_bias = np2th(weights["{}/{}/bias".format(ROOT, ATTENTION_K)]).view(-1)
            value_bias = np2th(weights["{}/{}/bias".format(ROOT, ATTENTION_V)]).view(-1)
            out_bias = np2th(weights["{}/{}/bias".format(ROOT, ATTENTION_OUT)]).view(-1)

            self.attn.query.weight.copy_(query_weight)
            self.attn.key.weight.copy_(key_weight)
            self.attn.value.weight.copy_(value_weight)
            self.attn.out.weight.copy_(out_weight)
            self.attn.query.bias.copy_(query_bias)
            self.attn.key.bias.copy_(key_bias)
            self.attn.value.bias.copy_(value_bias)
            self.attn.out.bias.copy_(out_bias)

            mlp_weight_0 = np2th(weights["{}/{}/kernel".format(ROOT, FC_0)]).t()
            mlp_weight_1 = np2th(weights["{}/{}/kernel".format(ROOT, FC_1)]).t()
            mlp_bias_0 = np2th(weights["{}/{}/bias".format(ROOT, FC_0)]).t()
            mlp_bias_1 = np2th(weights["{}/{}/bias".format(ROOT, FC_1)]).t()

            self.ffn.fc1.weight.copy_(mlp_weight_0)
            self.ffn.fc2.weight.copy_(mlp_weight_1)
            self.ffn.fc1.bias.copy_(mlp_bias_0)
            self.ffn.fc2.bias.copy_(mlp_bias_1)

            self.attention_norm.weight.copy_(np2th(weights["{}/{}/scale".format(ROOT, ATTENTION_NORM)]))
            self.attention_norm.bias.copy_(np2th(weights["{}/{}/bias".format(ROOT, ATTENTION_NORM)]))
            self.ffn_norm.weight.copy_(np2th(weights["{}/{}/scale".format(ROOT, MLP_NORM)]))
            self.ffn_norm.bias.copy_(np2th(weights["{}/{}/bias".format(ROOT, MLP_NORM)]))


class Encoder(nn.Module):
    def __init__(self, config, vis):
        super(Encoder, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)
        for _ in range(config.transformer["num_layers"]):
            layer = Block(config, vis)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):
        attn_weights = []
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states)
            if self.vis:
                attn_weights.append(weights)
        encoded = self.encoder_norm(hidden_states)
        return encoded, attn_weights


class Transformer(nn.Module):
    def __init__(self, config, in_channels, vis):
        super(Transformer, self).__init__()
        self.embeddings = Embeddings(config, in_channels)
        self.encoder = Encoder(config, vis)

    def forward(self, input_ids):
        embedding_output = self.embeddings(input_ids)
        encoded, attn_weights = self.encoder(embedding_output)  # (B, n_patch, hidden)
        return encoded, attn_weights


class Conv2dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        relu = nn.ReLU(inplace=True)
        bn = nn.BatchNorm2d(out_channels)
        super(Conv2dReLU, self).__init__(conv, bn, relu)


class ConvMore(nn.Module):
    def __init__(self, config):
        super(ConvMore, self).__init__()
        self.config = config
        self.head_channels = config.head_channels
        self.conv = Conv2dReLU(
            config.hidden_size,
            self.head_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=True
        )

    def forward(self, hidden_states):
        B, n_patch, hidden = hidden_states.size()  # reshape from (B, n_patch, hidden) to (B, h, w, hidden)
        h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        x = hidden_states.permute(0, 2, 1)
        x = x.contiguous().view(B, hidden, h, w)
        x = self.conv(x)
        return x


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def init_weights(net, init_type='normal'):
    if init_type == 'normal':
        net.apply(weights_init_normal)
    elif init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)


class unetConv2(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm, n=2, ks=3, stride=1, padding=1):
        super(unetConv2, self).__init__()
        self.n = n
        self.ks = ks
        self.stride = stride
        self.padding = padding
        s = stride
        p = padding
        if is_batchnorm:
            for i in range(1, n + 1):
                conv = nn.Sequential(nn.Conv2d(in_size, out_size, ks, s, p),
                                     nn.BatchNorm2d(out_size),
                                     nn.ReLU(inplace=True), )
                setattr(self, 'conv%d' % i, conv)
                in_size = out_size
        else:
            for i in range(1, n + 1):
                conv = nn.Sequential(nn.Conv2d(in_size, out_size, ks, s, p),
                                     nn.ReLU(inplace=True), )
                setattr(self, 'conv%d' % i, conv)
                in_size = out_size
        # initialise the blocks
        for m in self.children():
            init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        x = inputs
        for i in range(1, self.n + 1):
            conv = getattr(self, 'conv%d' % i)
            x = conv(x)
        return x


class unetUp_origin(nn.Module):
    def __init__(self, in_size, out_size, is_deconv=False, n_concat=2):
        super(unetUp_origin, self).__init__()
        # self.conv = unetConv2(out_size*2, out_size, False)
        if is_deconv:
            self.conv = unetConv2(in_size + (n_concat - 2) * out_size, out_size, False)
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=4, stride=2, padding=1)
        else:
            self.conv = unetConv2(in_size + (n_concat - 2) * out_size, out_size, False)
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)

        # initialise the blocks
        for m in self.children():
            if m.__class__.__name__.find('unetConv2') != -1: continue
            init_weights(m, init_type='kaiming')

    def forward(self, inputs0, *input):
        outputs0 = self.up(inputs0)
        for i in range(len(input)):
            outputs0 = torch.cat([outputs0, input[i]], 1)
        return self.conv(outputs0)


class ClusterSeg(nn.Module):
    def __init__(self, config, img_size=512, num_classes=2, in_channels=3, vis=False):
        super(ClusterSeg, self).__init__()
        self.config = config
        self.img_size = img_size
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.vis = vis
        self.classifier = config.classifier
        self.is_deconv = config.is_deconv
        self.is_batchnorm = config.is_batchnorm
        self.is_ds = config.is_ds
        # self.encoder = Encoder(config, vis)

        filters = [32, 64, 128, 256, 512]

        self.transformer = Transformer(config, in_channels=filters[3], vis=vis)
        self.convmore = ConvMore(config)

        self.conv00 = unetConv2(self.in_channels, filters[0], self.is_batchnorm)
        self.maxpool0 = nn.MaxPool2d(kernel_size=2)
        self.conv10 = unetConv2(filters[0], filters[1], self.is_batchnorm)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.conv20 = unetConv2(filters[1], filters[2], self.is_batchnorm)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.conv30 = unetConv2(filters[2], filters[3], self.is_batchnorm)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)

        # upsampling
        # mask path
        self.up_concat01 = unetUp_origin(filters[1], filters[0], self.is_deconv)
        self.up_concat11 = unetUp_origin(filters[2], filters[1], self.is_deconv)
        self.up_concat21 = unetUp_origin(filters[3], filters[2], self.is_deconv)
        self.up_concat31 = unetUp_origin(filters[4], filters[3], self.is_deconv)

        self.up_concat02 = unetUp_origin(filters[1], filters[0], self.is_deconv, 3)
        self.up_concat12 = unetUp_origin(filters[2], filters[1], self.is_deconv, 3)
        self.up_concat22 = unetUp_origin(filters[3], filters[2], self.is_deconv, 3)

        self.up_concat03 = unetUp_origin(filters[1], filters[0], self.is_deconv, 4)
        self.up_concat13 = unetUp_origin(filters[2], filters[1], self.is_deconv, 4)

        self.up_concat04 = unetUp_origin(filters[1], filters[0], self.is_deconv, 5)

        # boundary path
        self.upb_concat01 = unetUp_origin(filters[1], filters[0], self.is_deconv)
        self.upb_concat11 = unetUp_origin(filters[2], filters[1], self.is_deconv)
        self.upb_concat21 = unetUp_origin(filters[3], filters[2], self.is_deconv)
        self.upb_concat31 = unetUp_origin(filters[4], filters[3], self.is_deconv)

        self.upb_concat02 = unetUp_origin(filters[1], filters[0], self.is_deconv, 3)
        self.upb_concat12 = unetUp_origin(filters[2], filters[1], self.is_deconv, 3)
        self.upb_concat22 = unetUp_origin(filters[3], filters[2], self.is_deconv, 3)

        self.upb_concat03 = unetUp_origin(filters[1], filters[0], self.is_deconv, 4)
        self.upb_concat13 = unetUp_origin(filters[2], filters[1], self.is_deconv, 4)

        self.upb_concat04 = unetUp_origin(filters[1], filters[0], self.is_deconv, 5)

        # three class path
        self.upt_concat31 = unetUp_origin(filters[4], filters[3], self.is_deconv)
        self.upt_concat22 = unetUp_origin(filters[3], filters[2], self.is_deconv, 3)
        self.upt_concat13 = unetUp_origin(filters[2], filters[1], self.is_deconv, 4)
        self.upt_concat04 = unetUp_origin(filters[1], filters[0], self.is_deconv, 5)

        # final conv (without any concat)
        self.final_4 = nn.Conv2d(filters[0], self.num_classes, 1)
        self.finalb_1 = nn.Conv2d(filters[0], self.num_classes, 1)
        self.finalb_2 = nn.Conv2d(filters[0], self.num_classes, 1)
        self.finalb_3 = nn.Conv2d(filters[0], self.num_classes, 1)
        self.finalb_4 = nn.Conv2d(filters[0], self.num_classes, 1)
        self.finalt_4 = nn.Conv2d(filters[0], self.num_classes, 1)

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m)
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m)

    def forward(self, inputs):
        X_00 = self.conv00(inputs)  # [B, 32, s, s]
        maxpool0 = self.maxpool0(X_00)  # [B, 32, s/2, s/2]
        X_10 = self.conv10(maxpool0)  # [B, 64, s/2, s/2]
        maxpool1 = self.maxpool1(X_10)  # [B, 64, s/4, s/4]
        X_20 = self.conv20(maxpool1)  # [B, 128, s/4, s/4]
        maxpool2 = self.maxpool2(X_20)  # [B, 128, s/8, s/8]
        X_30 = self.conv30(maxpool2)  # [B, 256, s/8, s/8]
        maxpool3 = self.maxpool3(X_30)  # [B, 256, s/16, s/16]
        x, attn_weights = self.transformer(maxpool3)  # [B, s/16 * s/16, 768]
        X_40 = self.convmore(x)  # [B, 512, s/16, s/16]

        # column 1
        X_01 = self.up_concat01(X_10, X_00)
        X_11 = self.up_concat11(X_20, X_10)
        X_21 = self.up_concat21(X_30, X_20)
        X_31 = self.up_concat31(X_40, X_30)

        Y_01 = self.upb_concat01(X_10, X_00)
        Y_11 = self.upb_concat11(X_20, X_10)
        Y_21 = self.upb_concat21(X_30, X_20)
        Y_31 = self.upb_concat31(X_40, X_30)

        Z_31 = self.upt_concat31(X_40, X_30)

        # column 2
        X_02 = self.up_concat02(X_11, X_00, X_01)
        X_12 = self.up_concat12(X_21, X_10, X_11)
        X_22 = self.up_concat22(X_31, X_20, X_21)

        Y_02 = self.upb_concat02(Y_11, X_00, Y_01)
        Y_12 = self.upb_concat12(Y_21, X_10, Y_11)
        Y_22 = self.upb_concat22(Y_31, X_20, Y_21)

        Z_22 = self.upt_concat22(Z_31, X_20, Y_21)

        # column 3
        X_03 = self.up_concat03(X_12, X_00, X_01, X_02)
        X_13 = self.up_concat13(X_22, X_10, X_11, X_12)

        Y_03 = self.upb_concat03(Y_12, X_00, Y_01, Y_02)
        Y_13 = self.upb_concat13(Y_22, X_10, Y_11, Y_12)

        Z_13 = self.upt_concat13(Z_22, X_10, Y_11, Y_12)

        # column 4
        X_04 = self.up_concat04(X_13, X_00, X_01, X_02, X_03)
        Y_04 = self.upb_concat04(Y_13, X_00, Y_01, Y_02, Y_03)
        Z_04 = self.upt_concat04(Z_13, X_00, Y_01, Y_02, Y_03)

        # final column
        final_m = self.final_4(X_04)
        final_b1 = self.finalb_1(Y_01)
        final_b2 = self.finalb_2(Y_02)
        final_b3 = self.finalb_3(Y_03)
        final_b4 = self.finalb_4(Y_04)
        final_t = self.finalt_4(Z_04)
        return F.softmax(final_m, dim=1), F.softmax((0.5 * final_b1 + 0.75 * final_b2 + 1.25 * final_b3 + 1.5 * final_b4) / 4, dim=1), F.softmax(final_t, dim=1)

    def load_from(self, weights):
        with torch.no_grad():
            self.transformer.encoder.encoder_norm.weight.copy_(np2th(weights["Transformer/encoder_norm/scale"]))
            self.transformer.encoder.encoder_norm.bias.copy_(np2th(weights["Transformer/encoder_norm/bias"]))

            posemb = np2th(weights["Transformer/posembed_input/pos_embedding"])

            posemb_new = self.transformer.embeddings.position_embeddings
            if posemb.size() == posemb_new.size():
                self.transformer.embeddings.position_embeddings.copy_(posemb)
            elif posemb.size()[1] - 1 == posemb_new.size()[1]:
                posemb = posemb[:, 1:]
                self.transformer.embeddings.position_embeddings.copy_(posemb)
            else:
                ntok_new = posemb_new.size(1)
                if self.classifier == "seg":
                    _, posemb_grid = posemb[:, :1], posemb[0, 1:]
                gs_old = int(np.sqrt(len(posemb_grid)))
                gs_new = int(np.sqrt(ntok_new))
                print('load_pretrained: grid-size from %s to %s' % (gs_old, gs_new))
                posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)
                zoom = (gs_new / gs_old, gs_new / gs_old, 1)
                posemb_grid = ndimage.zoom(posemb_grid, zoom, order=1)  # th2np
                posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
                posemb = posemb_grid
                self.transformer.embeddings.position_embeddings.copy_(np2th(posemb))

            for bname, block in self.transformer.encoder.named_children():
                for uname, unit in block.named_children():
                    unit.load_from(weights, n_block=uname)

