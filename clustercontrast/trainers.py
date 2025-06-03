from __future__ import print_function, absolute_import
from audioop import cross
import time
from .utils.meters import AverageMeter
import torch.nn as nn
import torch
from torch.nn import functional as F
import math
import torchvision.transforms as transforms
from ChannelAug import ChannelExchange
from clustercontrast.models.agw import embed_net_ori


def pdist_torch(emb1, emb2):
    '''
    compute the eucilidean distance matrix between embeddings1 and embeddings2
    using gpu
    '''
    m, n = emb1.shape[0], emb2.shape[0]
    emb1_pow = torch.pow(emb1, 2).sum(dim=1, keepdim=True).expand(m, n)
    emb2_pow = torch.pow(emb2, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist_mtx = emb1_pow + emb2_pow
    dist_mtx = dist_mtx.addmm_(1, -2, emb1, emb2.t())
    # dist_mtx = dist_mtx.clamp(min = 1e-12)
    dist_mtx = dist_mtx.clamp(min=1e-12).sqrt()
    return dist_mtx


def softmax_weights(dist, mask):
    max_v = torch.max(dist * mask, dim=1, keepdim=True)[0]
    diff = dist - max_v
    Z = torch.sum(torch.exp(diff) * mask, dim=1, keepdim=True) + 1e-6  # avoid division by zero
    W = torch.exp(diff) * mask / Z
    return W


def normalize(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
      x: pytorch Variable
    Returns:
      x: pytorch Variable, same shape as input
    """
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x


def l2norm(x, power=2):
    norm = x.pow(power).sum(1, keepdim=True).pow(1. / power)
    x = x.div(norm)
    return x


def cift(x, num_rgb=64, mode='r2i', c=None, k=None, b=None, training=True):
    # l2norm = normalize2()
    num = x.shape[0]
    num_ir = num - num_rgb
    s = l2norm(x, 2) @ l2norm(x, 2).T
    s_hom = (s / 0.4).exp()
    s_het = (s / 0.2).exp()
    mask_hom = torch.zeros_like(s)
    mask_het = torch.zeros_like(s)
    cmask = torch.ones_like(s) - torch.eye(num).type_as(x)
    if mode == 'r2i':
        mask_hom[num_rgb:, num_rgb:] = 1  # 右下全1
        mask_het[:num_rgb, num_rgb:] = 1  # 左下全1
        mask_het[:num_rgb, :num_rgb] = torch.eye(num_rgb).type_as(x)
    if mode == 'i2r':
        mask_hom[:num_rgb, :num_rgb] = 1
        mask_het[num_rgb:, :num_rgb] = 1
        mask_het[num_rgb:, num_rgb:] = torch.eye(num_ir).type_as(x)
    mask = mask_hom + mask_het
    s_hom = s_hom * mask
    topk_hom = s_hom.topk(k=4 + 1, dim=1)[0][:, -1]
    topk_hom = ((s_hom - topk_hom.unsqueeze(1)) > 0).float()
    a_hom = s_hom * topk_hom
    a = a_hom / (a_hom.sum(dim=-1, keepdim=True))
    gx = a @ x
    if training:
        t = 10
        for i in range(t):

            cx = (torch.ones_like(x).normal_(0, 1) * c.clamp(0) + k)
            cx = (1 - b) * x + b * cx
            cs = l2norm(cx, 2) @ l2norm(cx, 2).T

            cs_hom = (cs / 0.4).exp()
            cs_hom = cs_hom * mask * cmask
            ctopk_hom = cs_hom.topk(k=4 + 1, dim=1)[0][:, -1]
            ctopk_hom = ((cs_hom - ctopk_hom.unsqueeze(1)) > 0).float()
            ca_hom = cs_hom * ctopk_hom
            ca = ca_hom / (ca_hom.sum(dim=-1, keepdim=True))
            if i == 0:
                cgx = ca @ x / t
                c = ca @ cx / t
            else:
                cgx = cgx + ca @ x / t
                c = c + ca @ x / t
        return gx, cgx, b, c
    else:
        return gx, b


class ClusterContrastTrainer(object):
    def __init__(self, encoder, memory=None):
        super(ClusterContrastTrainer, self).__init__()
        self.encoder = encoder
        self.memory_ir = memory
        self.memory_rgb = memory
        # self.c = nn.Parameter(torch.ones(384, 1))
        # self.k = nn.Parameter(torch.zeros(384, 2048))
        self.b = nn.Parameter(torch.tensor(0.5), requires_grad=True)
        # self.tri = TripletLoss_ADP(alpha = 1, gamma = 1, square = 1)

    def train(self, epoch, data_loader_ir, data_loader_rgb, optimizer, print_freq=10, train_iters=400, i2r=None,
              r2i=None):
        self.encoder.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        criterion_tri = OriTripletLoss(256, 0.3)  # (batchsize, margin)

        end = time.time()
        for i in range(train_iters):
            # load data
            inputs_ir = data_loader_ir.next()
            inputs_rgb = data_loader_rgb.next()
            data_time.update(time.time() - end)

            # process inputs
            inputs_ir, labels_ir, indexes_ir = self._parse_data_ir(inputs_ir)
            inputs_rgb, inputs_rgb1, labels_rgb, indexes_rgb = self._parse_data_rgb(inputs_rgb)
            # KL any?

            # forward
            inputs_rgb = torch.cat((inputs_rgb, inputs_rgb1), 0)
            labels_rgb = torch.cat((labels_rgb, labels_rgb), -1)
            # _, f_out_rgb, f_out_ir, labels_rgb, labels_ir, pool_rgb, pool_ir, gy_r2i, gy_i2r, cgy_r2i, cgy_i2r, k, c_r2i, c_i2r = self._forward(inputs_rgb, inputs_ir, label_1=labels_rgb, label_2=labels_ir, modal=0)

            feat, f_out_rgb, f_out_ir, labels_rgb, labels_ir, pool_rgb, pool_ir, c, k = self._forward(inputs_rgb,
                                                                                                      inputs_ir,
                                                                                                      label_1=labels_rgb,
                                                                                                      label_2=labels_ir,
                                                                                                      modal=0)
            f_r2i = feat
            f_i2r = feat
            # # print(feat.shape)
            # # print(feat_h.shape)
            b = self.b
            # gf_r2i, cgf_r2i, k, c = cift(f_r2i, 256, 'r2i', c, k, b)
            # gf_i2r, cgf_i2r, k, c = cift(f_i2r, 128, 'i2r', c, k, b)
            gf_r2i, cgf_r2i, k, c = cift(f_r2i, 192, 'r2i', c, k, b)
            gf_i2r, cgf_i2r, k, c = cift(f_i2r, 192, 'i2r', c, k, b)
            gy_r2i = self.encoder.module.classifier(gf_r2i)
            gy_i2r = self.encoder.module.classifier(gf_i2r)
            cgy_r2i = self.encoder.module.classifier(gf_r2i - cgf_r2i)
            cgy_i2r = self.encoder.module.classifier(gf_i2r - cgf_i2r)
            loss_ir = self.memory_ir(f_out_ir, labels_ir)
            loss_rgb = self.memory_rgb(f_out_rgb, labels_rgb)
            criterion_id = nn.CrossEntropyLoss()
            criterion_id.to('cuda')
            labels = torch.cat((labels_rgb, labels_ir), -1)
            criterion = nn.L1Loss()
            # criterion_hcc = hcc(margin_euc=0.6, margin_kl=6)

            # if k.dim() > 0:
            #     k = k.sum()
            # cross contrastive learning
            if r2i:
                rgb2ir_labels = torch.tensor([r2i[key.item()] for key in labels_rgb]).cuda()
                ir2rgb_labels = torch.tensor([i2r[key.item()] for key in labels_ir]).cuda()
                loss3 = (criterion_id(cgy_r2i, labels) / 2 + criterion_id(cgy_i2r, labels) / 2) + k * k

                alternate = True
                if alternate:
                    # accl
                    if epoch % 2 == 1:
                        cross_loss = 1 * self.memory_rgb(f_out_ir, ir2rgb_labels.long())
                    else:
                        cross_loss = 1 * self.memory_ir(f_out_rgb, rgb2ir_labels.long())
                else:
                    cross_loss = self.memory_rgb(f_out_ir, ir2rgb_labels.long()) + self.memory_ir(f_out_rgb,
                                                                                                  rgb2ir_labels.long())
                    # Unidirectional
                    # cross_loss = self.memory_rgb(f_out_ir, ir2rgb_labels.long())
                    # cross_loss = self.memory_ir(f_out_rgb, rgb2ir_labels.long())
            else:
                cross_loss = torch.tensor(0.0)
                loss3 = torch.tensor(0.0)
                loss_h = torch.tensor(0.0)
                # loss4 = torch.tensor(0.0)
            new_loss_rgb = loss_rgb
            new_cross_loss = cross_loss
            # loss = loss_ir + new_loss_rgb + 0.25 * new_cross_loss
            loss = loss_ir + new_loss_rgb + 0.25 * new_cross_loss + loss3 + k * k  # total loss
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

            losses.update(loss.item())

            # print log
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss {:.3f} ({:.3f})\t'
                      'Loss ir {:.3f}\t'
                      'Loss rgb {:.3f}\t'
                      'Loss cross {:.3f}\t'

                      #   'Loss tri rgb {:.3f}\t'
                      #   'Loss tri ir {:.3f}\t'
                      .format(epoch, i + 1, len(data_loader_rgb),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg, loss_ir.item(), new_loss_rgb.item(), new_cross_loss.item()
                              #   , loss_tri_rgb
                              # , loss_tri_ir
                              ))

    def _parse_data_rgb(self, inputs):
        imgs, imgs1, _, pids, _, indexes = inputs
        return imgs.cuda(), imgs1.cuda(), pids.cuda(), indexes.cuda()

    def _parse_data_ir(self, inputs):
        imgs, _, pids, _, indexes = inputs
        return imgs.cuda(), pids.cuda(), indexes.cuda()

    def _forward(self, x1, x2, x1_c=None, x2_c=None, label_1=None, label_2=None, modal=0):
        return self.encoder(x1, x2, modal=modal, label_1=label_1, label_2=label_2)


class OriTripletLoss(nn.Module):
    """Triplet loss with hard positive/negative mining.

    Reference:
    Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py.

    Args:
    - margin (float): margin for triplet.
    """

    def __init__(self, batch_size, margin=0.3):
        super(OriTripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets):
        """
        Args:
        - inputs: feature matrix with shape (batch_size, feat_dim)
        - targets: ground truth labels with shape (num_classes)
        """
        n = inputs.size(0)

        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability

        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)

        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)

        # compute accuracy
        correct = torch.ge(dist_an, dist_ap).sum().item()
        return loss, correct