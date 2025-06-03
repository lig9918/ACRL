import torch
import torch.nn as nn
from torch.nn import init
from .resnet_agw import resnet50 as resnet50_agw
import numpy as np
import torchvision.transforms as transforms
import torch.nn.functional as F
from ChannelAug import ChannelAdap, ChannelAdapGray, ChannelRandomErasing, ChannelExchange, Gray, RGB2HSV
from clustercontrast.utils.data import transforms as T
# import pywt

class Normalize(nn.Module):
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out


class Non_local(nn.Module):
    def __init__(self, in_channels, reduc_ratio=2):
        super(Non_local, self).__init__()

        self.in_channels = in_channels
        self.inter_channels = reduc_ratio // reduc_ratio

        self.g = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1,
                      padding=0),
        )

        self.W = nn.Sequential(
            nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels,
                      kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.in_channels),
        )
        nn.init.constant_(self.W[1].weight, 0.0)
        nn.init.constant_(self.W[1].bias, 0.0)

        self.theta = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                               kernel_size=1, stride=1, padding=0)

        self.phi = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        '''
                :param x: (b, c, t, h, w)
                :return:
                '''

        batch_size = x.size(0)
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        N = f.size(-1)
        # f_div_C = torch.nn.functional.softmax(f, dim=-1)
        f_div_C = f / N

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


# #####################################################################

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.zeros_(m.bias.data)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.01)
        init.zeros_(m.bias.data)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0, 0.001)
        if m.bias:
            init.zeros_(m.bias.data)


class visible_module(nn.Module):
    def __init__(self, arch='resnet50'):
        super(visible_module, self).__init__()

        model_v = resnet50_agw(pretrained=True,
                               last_conv_stride=1, last_conv_dilation=1)
        # avg pooling to global pooling
        self.visible = model_v

    def forward(self, x):
        x = self.visible.conv1(x)
        x = self.visible.bn1(x)
        x = self.visible.relu(x)
        x = self.visible.maxpool(x)
        return x


class thermal_module(nn.Module):
    def __init__(self, arch='resnet50'):
        super(thermal_module, self).__init__()

        model_t = resnet50_agw(pretrained=True,
                               last_conv_stride=1, last_conv_dilation=1)
        # avg pooling to global pooling
        self.thermal = model_t

    def forward(self, x):
        x = self.thermal.conv1(x)
        x = self.thermal.bn1(x)
        x = self.thermal.relu(x)
        x = self.thermal.maxpool(x)
        return x


class base_resnet(nn.Module):
    def __init__(self, arch='resnet50'):
        super(base_resnet, self).__init__()

        model_base = resnet50_agw(pretrained=True,
                                  last_conv_stride=1, last_conv_dilation=1)
        # avg pooling to global pooling
        model_base.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.base = model_base

    def forward(self, x):
        x = self.base.layer1(x)
        x = self.base.layer2(x)
        x = self.base.layer3(x)
        x = self.base.layer4(x)
        return x

#
# class cift(nn.Module):
#     def __init__(self):
#         super(cift, self).__init__()
#         self.l2norm = Normalize(2)
#         # self.c = nn.Parameter(torch.ones(128, 1))
#         # self.k = nn.Parameter(torch.zeros(128, 2048))
#         self.c = nn.Parameter(torch.ones(192, 1))
#         self.k = nn.Parameter(torch.zeros(192, 2048))
#         self.b = nn.Parameter(torch.tensor(0.5), requires_grad=True)
#
#     def forward(self, x, num_rgb=64, mode='r2i'):
#         num = x.shape[0]
#
#         num_ir = num - num_rgb
#         s = self.l2norm(x) @ self.l2norm(x).T
#         s_hom = (s / 0.4).exp()
#         s_het = (s / 0.2).exp()
#         mask_hom = torch.zeros_like(s)
#         mask_het = torch.zeros_like(s)
#         cmask = torch.ones_like(s) - torch.eye(num).type_as(x)
#         if mode == 'r2i':
#             mask_hom[num_rgb:, num_rgb:] = 1  # 右下全1
#             mask_het[:num_rgb, num_rgb:] = 1  # 左下全1
#             mask_het[:num_rgb, :num_rgb] = torch.eye(num_rgb).type_as(x)
#         if mode == 'i2r':
#             mask_hom[:num_rgb, :num_rgb] = 1
#             mask_het[num_rgb:, :num_rgb] = 1
#             mask_het[num_rgb:, num_rgb:] = torch.eye(num_ir).type_as(x)
#         mask = mask_hom + mask_het
#         s_hom = s_hom * mask
#         topk_hom = s_hom.topk(k=4 + 1, dim=1)[0][:, -1]
#         topk_hom = ((s_hom - topk_hom.unsqueeze(1)) > 0).float()
#         a_hom = s_hom * topk_hom
#         a = a_hom / (a_hom.sum(dim=-1, keepdim=True))
#         gx = a @ x
#
#         if self.training:
#             t = 10
#             for i in range(t):
#                 b = self.b
#                 cx = (torch.ones_like(x).normal_(0, 1) * self.c.clamp(0) + self.k)
#                 cx = (1 - b) * x + b * cx
#                 cs = self.l2norm(cx) @ self.l2norm(cx).T
#
#                 cs_hom = (cs / 0.4).exp()
#                 cs_hom = cs_hom * mask * cmask
#                 ctopk_hom = cs_hom.topk(k=4 + 1, dim=1)[0][:, -1]
#                 ctopk_hom = ((cs_hom - ctopk_hom.unsqueeze(1)) > 0).float()
#                 ca_hom = cs_hom * ctopk_hom
#                 ca = ca_hom / (ca_hom.sum(dim=-1, keepdim=True))
#                 if i == 0:
#                     cgx = ca @ x / t
#                     c = ca @ cx / t
#                 else:
#                     cgx = cgx + ca @ x / t
#                     c = c + ca @ x / t
#             return gx, cgx, b,c
#         else:
#             return gx, b


class embed_net_ori(nn.Module):
    def __init__(self, num_classes=1000, no_local='on', gm_pool='on', arch='resnet50'):
        super(embed_net_ori, self).__init__()

        self.thermal_module = thermal_module(arch=arch)
        self.visible_module = visible_module(arch=arch)
        self.base_resnet = base_resnet(arch=arch)
        self.non_local = no_local
        if self.non_local == 'on':
            layers = [3, 4, 6, 3]
            non_layers = [0, 2, 3, 0]
            self.NL_1 = nn.ModuleList(
                [Non_local(256) for i in range(non_layers[0])])
            self.NL_1_idx = sorted([layers[0] - (i + 1) for i in range(non_layers[0])])
            self.NL_2 = nn.ModuleList(
                [Non_local(512) for i in range(non_layers[1])])
            self.NL_2_idx = sorted([layers[1] - (i + 1) for i in range(non_layers[1])])
            self.NL_3 = nn.ModuleList(
                [Non_local(1024) for i in range(non_layers[2])])
            self.NL_3_idx = sorted([layers[2] - (i + 1) for i in range(non_layers[2])])
            self.NL_4 = nn.ModuleList(
                [Non_local(2048) for i in range(non_layers[3])])
            self.NL_4_idx = sorted([layers[3] - (i + 1) for i in range(non_layers[3])])

        pool_dim = 2048
        self.num_features = pool_dim
        self.l2norm = Normalize(2)
        self.bottleneck = nn.BatchNorm1d(pool_dim)
        self.bottleneck.bias.requires_grad_(False)  # no shift

        self.classifier = nn.Linear(pool_dim, num_classes, bias=False)

        self.bottleneck.apply(weights_init_kaiming)
        self.classifier.apply(weights_init_classifier)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.gm_pool = gm_pool
        # self.cift = cift()
        self.c = nn.Parameter(torch.ones(192, 1))
        self.k = nn.Parameter(torch.zeros(192, 2048))

        # self.ca = ChannelAttention(64)
        # self.ca1 = ChannelAttention(256)
        # self.ca2 = ChannelAttention(512)
        # self.ca3 = ChannelAttention(1024)
        # self.ca4 = ChannelAttention(2048)

    # 应用数据增强到 x1

    def forward(self, x1, x2, modal=0, label_1=None, label_2=None):
        single_size = x1.size(0)
        # print(x1.size(0))
        # print( x2.size(0))
        if modal == 0:
            x1 = self.visible_module(x1)
            x2 = self.thermal_module(x2)
            x = torch.cat((x1, x2), 0)
            label = torch.cat((label_1, label_2), -1)
        elif modal == 1:
            x = self.visible_module(x1)
        elif modal == 2:
            x = self.thermal_module(x2)

        # shared block
        if self.non_local == 'on':
            NL1_counter = 0
            if len(self.NL_1_idx) == 0: self.NL_1_idx = [-1]
            for i in range(len(self.base_resnet.base.layer1)):
                x = self.base_resnet.base.layer1[i](x)
                if i == self.NL_1_idx[NL1_counter]:
                    _, C, H, W = x.shape
                    x = self.NL_1[NL1_counter](x)
                    NL1_counter += 1
            # x = x * self.ca1(x)
            # Layer 2
            NL2_counter = 0
            if len(self.NL_2_idx) == 0: self.NL_2_idx = [-1]
            for i in range(len(self.base_resnet.base.layer2)):
                x = self.base_resnet.base.layer2[i](x)
                if i == self.NL_2_idx[NL2_counter]:
                    _, C, H, W = x.shape
                    x = self.NL_2[NL2_counter](x)
                    NL2_counter += 1
            # Layer 3
            NL3_counter = 0
            if len(self.NL_3_idx) == 0: self.NL_3_idx = [-1]
            for i in range(len(self.base_resnet.base.layer3)):
                x = self.base_resnet.base.layer3[i](x)

                if i == self.NL_3_idx[NL3_counter]:
                    _, C, H, W = x.shape
                    x = self.NL_3[NL3_counter](x)
                    NL3_counter += 1
            # Layer 4
            NL4_counter = 0
            if len(self.NL_4_idx) == 0: self.NL_4_idx = [-1]
            for i in range(len(self.base_resnet.base.layer4)):
                x = self.base_resnet.base.layer4[i](x)

                if i == self.NL_4_idx[NL4_counter]:
                    _, C, H, W = x.shape
                    x = self.NL_4[NL4_counter](x)
                    NL4_counter += 1
            # x = x * self.ca4(x)

        else:
            x = self.base_resnet(x)
        if self.gm_pool == 'on':
            b, c, h, w = x.shape
            x = x.view(b, c, -1)
            p = 3.0
            x_pool = (torch.mean(x ** p, dim=-1) + 1e-12) ** (1 / p)

        else:
            x_pool = self.avgpool(x)
            x_pool = x_pool.view(x_pool.size(0), x_pool.size(1))


        feat = self.bottleneck(x_pool)
        c = self.c
        k = self.k
        f_r2i = self.bottleneck(x_pool)
        f_i2r = self.bottleneck(x_pool)

        # if self.training:
        #     # gf_r2i, cgf_r2i, k, c = self.cift(f_r2i, 96, 'r2i')
        #     # gf_i2r, cgf_i2r, k, c = self.cift(f_i2r, 96, 'i2r')
        #     gf_r2i, cgf_r2i, k, c = self.cift(f_r2i, 96, 'r2i')
        #     gf_i2r, cgf_i2r, k, c = self.cift(f_i2r, 96, 'i2r')
        #     gy_r2i = self.classifier(gf_r2i)
        #     gy_i2r = self.classifier(gf_i2r)
        #
        #     cgy_r2i = self.classifier(gf_r2i - cgf_r2i)
        #     cgy_i2r = self.classifier(gf_i2r - cgf_i2r)
        #
        #     c_r2i = self.classifier(cgf_r2i - c)
        #     c_i2r = self.classifier(cgf_i2r - c)
        #
        #     return feat, feat[:single_size], feat[single_size:], label_1, label_2, x_pool[:single_size], x_pool[
        #                                                                                                  single_size:], gy_r2i, gy_i2r, cgy_r2i, cgy_i2r, k,c_r2i,c_i2r
        # else:
        #     return self.l2norm(feat)


        if self.training:
            return feat,feat[:single_size],feat[single_size:],label_1,label_2,x_pool[:single_size],x_pool[single_size:], c, k
            # x_pool#, self.classifier(feat)
        else:
            # return self.l2norm(x_pool), self.l2norm(feat),
            return self.l2norm(feat)#self.l2norm(x_pool)#,

def agw(pretrained=False, no_local='on', **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = embed_net_ori(no_local='on', gm_pool='on')  # without no-local -> resnet with non-local->agw

    return model