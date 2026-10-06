"""ESDNet (Yu et al., ECCV 2022, "Towards Efficient and Scale-Robust Ultra-High-Definition
Image Demoireing"), reused here as the de-weathering / low-light enhancement network.

Module attribute names must stay as they are: ``weights/deweathering_model.pth`` is a
state_dict keyed by them.

Input constraint: height and width must be multiples of 32 (pixel-unshuffle by 2, two
stride-2 downsamplings, and the SAM block's 1/4-scale branch).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ESDNet(nn.Module):
    def __init__(self, en_feature_num, en_inter_num, de_feature_num, de_inter_num, sam_number=1):
        super().__init__()
        self.encoder = Encoder(feature_num=en_feature_num, inter_num=en_inter_num, sam_number=sam_number)
        self.decoder = Decoder(en_num=en_feature_num, feature_num=de_feature_num, inter_num=de_inter_num,
                               sam_number=sam_number)

    def forward(self, x):
        """x: (B, 3, H, W) in [0, 1] -> (out_full, out_half, out_quarter)."""
        h, w = x.shape[-2:]
        if h % 32 or w % 32:
            raise ValueError(f"ESDNet input height/width must be multiples of 32, got {h}x{w}")
        y_1, y_2, y_3 = self.encoder(x)
        return self.decoder(y_1, y_2, y_3)


class Decoder(nn.Module):
    def __init__(self, en_num, feature_num, inter_num, sam_number):
        super().__init__()
        self.preconv_3 = conv_relu(4 * en_num, feature_num, 3, padding=1)
        self.decoder_3 = Decoder_Level(feature_num, inter_num, sam_number)

        self.preconv_2 = conv_relu(2 * en_num + feature_num, feature_num, 3, padding=1)
        self.decoder_2 = Decoder_Level(feature_num, inter_num, sam_number)

        self.preconv_1 = conv_relu(en_num + feature_num, feature_num, 3, padding=1)
        self.decoder_1 = Decoder_Level(feature_num, inter_num, sam_number)

    def forward(self, y_1, y_2, y_3):
        x_3 = self.preconv_3(y_3)
        out_3, feat_3 = self.decoder_3(x_3)

        x_2 = self.preconv_2(torch.cat([y_2, feat_3], dim=1))
        out_2, feat_2 = self.decoder_2(x_2)

        x_1 = self.preconv_1(torch.cat([y_1, feat_2], dim=1))
        out_1 = self.decoder_1(x_1, feat=False)

        return out_1, out_2, out_3


class Encoder(nn.Module):
    def __init__(self, feature_num, inter_num, sam_number):
        super().__init__()
        self.conv_first = nn.Sequential(
            nn.Conv2d(12, feature_num, kernel_size=5, stride=1, padding=2, bias=True),
            nn.ReLU(inplace=True),
        )
        self.encoder_1 = Encoder_Level(feature_num, inter_num, level=1, sam_number=sam_number)
        self.encoder_2 = Encoder_Level(2 * feature_num, inter_num, level=2, sam_number=sam_number)
        self.encoder_3 = Encoder_Level(4 * feature_num, inter_num, level=3, sam_number=sam_number)

    def forward(self, x):
        x = F.pixel_unshuffle(x, 2)
        x = self.conv_first(x)

        out_feature_1, down_feature_1 = self.encoder_1(x)
        out_feature_2, down_feature_2 = self.encoder_2(down_feature_1)
        out_feature_3 = self.encoder_3(down_feature_2)

        return out_feature_1, out_feature_2, out_feature_3


class Encoder_Level(nn.Module):
    def __init__(self, feature_num, inter_num, level, sam_number):
        super().__init__()
        self.rdb = RDB(in_channel=feature_num, d_list=(1, 2, 1), inter_num=inter_num)
        self.sam_blocks = nn.ModuleList(
            SAM(in_channel=feature_num, d_list=(1, 2, 3, 2, 1), inter_num=inter_num) for _ in range(sam_number)
        )
        if level < 3:
            self.down = nn.Sequential(
                nn.Conv2d(feature_num, 2 * feature_num, kernel_size=3, stride=2, padding=1, bias=True),
                nn.ReLU(inplace=True),
            )
        self.level = level

    def forward(self, x):
        out_feature = self.rdb(x)
        for sam_block in self.sam_blocks:
            out_feature = sam_block(out_feature)
        if self.level < 3:
            return out_feature, self.down(out_feature)
        return out_feature


class Decoder_Level(nn.Module):
    def __init__(self, feature_num, inter_num, sam_number):
        super().__init__()
        self.rdb = RDB(feature_num, (1, 2, 1), inter_num)
        self.sam_blocks = nn.ModuleList(
            SAM(in_channel=feature_num, d_list=(1, 2, 3, 2, 1), inter_num=inter_num) for _ in range(sam_number)
        )
        self.conv = conv(in_channel=feature_num, out_channel=12, kernel_size=3, padding=1)

    def forward(self, x, feat=True):
        x = self.rdb(x)
        for sam_block in self.sam_blocks:
            x = sam_block(x)
        out = F.pixel_shuffle(self.conv(x), 2)
        if feat:
            return out, F.interpolate(x, scale_factor=2, mode="bilinear")
        return out


class DB(nn.Module):
    """Dense block: dilated conv layers with dense concatenation, then a 1x1 projection."""

    def __init__(self, in_channel, d_list, inter_num):
        super().__init__()
        self.d_list = d_list
        self.conv_layers = nn.ModuleList()
        c = in_channel
        for d in d_list:
            self.conv_layers.append(
                conv_relu(in_channel=c, out_channel=inter_num, kernel_size=3, dilation_rate=d, padding=d)
            )
            c = c + inter_num
        self.conv_post = conv(in_channel=c, out_channel=in_channel, kernel_size=1)

    def forward(self, x):
        t = x
        for conv_layer in self.conv_layers:
            t = torch.cat([conv_layer(t), t], dim=1)
        return self.conv_post(t)


class RDB(DB):
    """Residual dense block: DB plus a skip connection."""

    def forward(self, x):
        return super().forward(x) + x


class SAM(nn.Module):
    """Semantic-aligned multi-scale block: three DBs at 1x / 0.5x / 0.25x fused by CSAF."""

    def __init__(self, in_channel, d_list, inter_num):
        super().__init__()
        self.basic_block = DB(in_channel=in_channel, d_list=d_list, inter_num=inter_num)
        self.basic_block_2 = DB(in_channel=in_channel, d_list=d_list, inter_num=inter_num)
        self.basic_block_4 = DB(in_channel=in_channel, d_list=d_list, inter_num=inter_num)
        self.fusion = CSAF(3 * in_channel)

    def forward(self, x):
        x_2 = F.interpolate(x, scale_factor=0.5, mode="bilinear")
        x_4 = F.interpolate(x, scale_factor=0.25, mode="bilinear")

        y_0 = self.basic_block(x)
        y_2 = F.interpolate(self.basic_block_2(x_2), scale_factor=2, mode="bilinear")
        y_4 = F.interpolate(self.basic_block_4(x_4), scale_factor=4, mode="bilinear")

        return x + self.fusion(y_0, y_2, y_4)


class CSAF(nn.Module):
    """Cross-scale attention fusion (squeeze-excitation over the three scale branches)."""

    def __init__(self, in_chnls, ratio=4):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d((1, 1))
        self.compress1 = nn.Conv2d(in_chnls, in_chnls // ratio, 1, 1, 0)
        self.compress2 = nn.Conv2d(in_chnls // ratio, in_chnls // ratio, 1, 1, 0)
        self.excitation = nn.Conv2d(in_chnls // ratio, in_chnls, 1, 1, 0)

    def forward(self, x0, x2, x4):
        out = torch.cat([self.squeeze(x0), self.squeeze(x2), self.squeeze(x4)], dim=1)
        out = F.relu(self.compress1(out))
        out = F.relu(self.compress2(out))
        out = torch.sigmoid(self.excitation(out))
        w0, w2, w4 = torch.chunk(out, 3, dim=1)
        return x0 * w0 + x2 * w2 + x4 * w4


class conv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, dilation_rate=1, padding=0, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=kernel_size,
                              stride=stride, padding=padding, bias=True, dilation=dilation_rate)

    def forward(self, x_input):
        return self.conv(x_input)


class conv_relu(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, dilation_rate=1, padding=0, stride=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=kernel_size,
                      stride=stride, padding=padding, bias=True, dilation=dilation_rate),
            nn.ReLU(inplace=True),
        )

    def forward(self, x_input):
        return self.conv(x_input)
