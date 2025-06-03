from __future__ import absolute_import

from torchvision.transforms import *
import cv2
# from PIL import Image
import random
import math
# import numpy as np
# import torch
from PIL import Image
import torchvision.transforms as transforms
import os.path as osp
from clustercontrast import datasets
import numpy as np


class ChannelExchange(object):
    """ Adaptive selects a channel or two channels.
    Args:
         probability: The probability that the Random Erasing operation will be performed.
         sl: Minimum proportion of erased area against input image.
         sh: Maximum proportion of erased area against input image.
         r1: Minimum aspect ratio of erased area.
         mean: Erasing value.
    """

    def __init__(self, gray=2):
        self.gray = gray

    def __call__(self, img):
        idx = random.randint(0, self.gray)
        if idx == 0:
            # random select R Channel
            img[1, :, :] = img[0, :, :]
            img[2, :, :] = img[0, :, :]
        elif idx == 1:
            # random select B Channel
            img[0, :, :] = img[1, :, :]
            img[2, :, :] = img[1, :, :]
        elif idx == 2:
            # random select G Channel
            img[0, :, :] = img[2, :, :]
            img[1, :, :] = img[2, :, :]
        else:
            tmp_img = 0.2989 * img[0, :, :] + 0.5870 * img[1, :, :] + 0.1140 * img[2, :, :]
            img[0, :, :] = tmp_img
            img[1, :, :] = tmp_img
            img[2, :, :] = tmp_img
        return img


class ChannelAdap(object):
    """ Adaptive selects a channel or two channels.
    Args:
         probability: The probability that the Random Erasing operation will be performed.
         sl: Minimum proportion of erased area against input image.
         sh: Maximum proportion of erased area against input image.
         r1: Minimum aspect ratio of erased area.
         mean: Erasing value.
    """

    def __init__(self, probability=0.5):
        self.probability = probability

    def __call__(self, img):

        # if random.uniform(0, 1) > self.probability:
        # return img

        idx = random.randint(0, 3)

        if idx == 0:
            # random select R Channel
            img[1, :, :] = img[0, :, :]
            img[2, :, :] = img[0, :, :]
        elif idx == 1:
            # random select B Channel
            img[0, :, :] = img[1, :, :]
            img[2, :, :] = img[1, :, :]
        elif idx == 2:
            # random select G Channel
            img[0, :, :] = img[2, :, :]
            img[1, :, :] = img[2, :, :]
        else:
            img = img

        return img


class ChannelAdapGray(object):
    """ Adaptive selects a channel or two channels.
    Args:
         probability: The probability that the Random Erasing operation will be performed.
         sl: Minimum proportion of erased area against input image.
         sh: Maximum proportion of erased area against input image.
         r1: Minimum aspect ratio of erased area.
         mean: Erasing value.
    """

    def __init__(self, probability=0.5):
        self.probability = probability

    def __call__(self, img):

        # if random.uniform(0, 1) > self.probability:
        # return img

        idx = random.randint(0, 3)

        if idx == 0:
            # random select R Channel
            img[1, :, :] = img[0, :, :]
            img[2, :, :] = img[0, :, :]
        elif idx == 1:
            # random select B Channel
            img[0, :, :] = img[1, :, :]
            img[2, :, :] = img[1, :, :]
        elif idx == 2:
            # random select G Channel
            img[0, :, :] = img[2, :, :]
            img[1, :, :] = img[2, :, :]
        else:
            if random.uniform(0, 1) > self.probability:
                # return img
                img = img
            else:
                tmp_img = 0.2989 * img[0, :, :] + 0.5870 * img[1, :, :] + 0.1140 * img[2, :, :]
                img[0, :, :] = tmp_img
                img[1, :, :] = tmp_img
                img[2, :, :] = tmp_img

        # img.show()
        return img


class Gray(object):
    """ Adaptive selects a channel or two channels.
    Args:
         probability: The probability that the Random Erasing operation will be performed.
         sl: Minimum proportion of erased area against input image.
         sh: Maximum proportion of erased area against input image.
         r1: Minimum aspect ratio of erased area.
         mean: Erasing value.
    """

    def __init__(self, probability=0.5):
        self.probability = probability

    def __call__(self, img):
        # if random.uniform(0, 1) > self.probability:
        # return img

        tmp_img = 0.2989 * img[0, :, :] + 0.5870 * img[1, :, :] + 0.1140 * img[2, :, :]
        img[0, :, :] = tmp_img
        img[1, :, :] = tmp_img
        img[2, :, :] = tmp_img
        return img


class ChannelRandomErasing(object):
    """ Randomly selects a rectangle region in an image and erases its pixels.
        'Random Erasing Data Augmentation' by Zhong et al.
        See https://arxiv.org/pdf/1708.04896.pdf
    Args:
         probability: The probability that the Random Erasing operation will be performed.
         sl: Minimum proportion of erased area against input image.
         sh: Maximum proportion of erased area against input image.
         r1: Minimum aspect ratio of erased area.
         mean: Erasing value.
    """

    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3, mean=[0.4914, 0.4822, 0.4465]):

        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img):

        if random.uniform(0, 1) > self.probability:
            return img

        for attempt in range(100):
            area = img.size()[1] * img.size()[2]

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                    img[1, x1:x1 + h, y1:y1 + w] = self.mean[1]
                    img[2, x1:x1 + h, y1:y1 + w] = self.mean[2]
                else:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                return img

        return img


class RGB2HSV(object):

    def __init__(self, probability=0.5):
        self.probability = probability

    def __call__(self, img):
        if random.uniform(0, 1) > self.probability:
            return img
        # print(type(img))
        # img = img.numpy()
        # print(type(img))
        # image_np = np.frombuffer(img, np.uint8)
        # image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

        #
        # img.save("/opt/data/private/Cross-modality/test/USL-VI-ReID-main/examples/picture/output.jpg")

        image = np.array(img)
        H, S, V = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))
        # angle = np.random.randint(-180, 180)
        # # angle = 30
        # rotated_h_channel = (H - angle) % 180
        # H = rotated_h_channel
        # #
        hgain = 0.5
        sgain = 0.5
        vgain = 0.5
        r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
        # uint8
        dtype = 'uint8'
        x = np.arange(0, 256, dtype=r.dtype)

        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)
        HSV = cv2.merge((cv2.LUT(H, lut_hue), cv2.LUT(S, lut_sat), cv2.LUT(V, lut_val)))
        # s_mean = np.mean(S)
        # v_mean = np.mean(V)
        #
        # print("S通道平均值:", s_mean)
        # print("V通道平均值:", v_mean)

        img = cv2.cvtColor(HSV, cv2.COLOR_HSV2BGR)
        # output_file = "/opt/data/private/Cross-modality/test/USL-VI-ReID-main/examples/picture/output004_image_{}.jpg".format(int(angle))
        # cv2.imwrite(output_file, img)

        return img


class IR2HSV(object):

    def __init__(self, probability=0.5):
        self.probability = probability

    def __call__(self, img):
        if random.uniform(0, 1) > self.probability:
            # img = Image.fromarray(img)
            return img

        image = np.array(img)
        H, S, V = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))
        # h_value = np.uint8(np.random.randint(0, 256))
        # s_value = np.uint8(np.random.randint(20, 80))
        # H = H + h_value
        # S = S + s_value
        # mean_h = np.random.randint(0, 256)
        # mean_s = np.random.randint(20, 80)
        # mean_h = 128  # 色调的平均值
        # stddev_h = 0  # 色调的标准差
        # mean_s = 50  # 饱和度的平均值
        # stddev_s = 0  # 饱和度的标准差

        # mean_h = 45  # 色调的平均值
        # stddev_h = 55  # 色调的标准差
        # mean_s = 35 # 饱和度的平均值
        # stddev_s = 25  # 饱和度的标准差
        # 52.1
        # mean_h = 71  # 色调的平均值
        # stddev_h = 56.5  # 色调的标准差
        # mean_s = 28  # 饱和度的平均值
        # stddev_s = 22.5  # 饱和度的标准差

        mean_h = 75  # 色调的平均值
        stddev_h = 60  # 色调的标准差
        mean_s = 20  # 饱和度的平均值
        stddev_s = 20  # 饱和度的标准差


        # 生成高斯噪声并应用到 H 和 S 通道
        h_noise = np.random.normal(mean_h, stddev_h, H.shape).astype(np.uint8)
        s_noise = np.random.normal(mean_s, stddev_s, S.shape).astype(np.uint8)

        H = cv2.add(H, h_noise)
        S = cv2.add(S, s_noise)

        HSV = cv2.merge((H, S, V))

        img = cv2.cvtColor(HSV, cv2.COLOR_HSV2BGR)
        img = Image.fromarray(img)
        return img
class IR2HSV2(object):

    def __init__(self, probability=1, block_size=32, area_ratio=0.4, num_areas=4):
        self.probability = probability
        self.block_size = block_size  # The size of each block
        self.area_ratio = area_ratio  # The ratio of the area to be enhanced
        self.num_areas = num_areas  # Number of areas to be enhanced

    def __call__(self, img):
        if random.uniform(0, 1) > self.probability:
            return img

        # Convert PIL image to numpy array
        image = np.array(img)

        # Get image dimensions
        height, width, _ = image.shape

        # Calculate the area to be enhanced
        area_height = int(height * self.area_ratio)
        area_width = int(width * self.area_ratio)

        # Select random areas and apply enhancement
        for _ in range(self.num_areas):
            # Randomly determine the starting point of the area
            start_x = random.randint(0, width - area_width)
            start_y = random.randint(0, height - area_height)

            # Iterate over blocks within the area
            for y in range(start_y, start_y + area_height, self.block_size):
                for x in range(start_x, start_x + area_width, self.block_size):
                    # Adjust block size to prevent exceeding image boundaries
                    block_width = min(self.block_size, width - x)
                    block_height = min(self.block_size, height - y)

                    # Extract the current block
                    block = image[y:y + block_height, x:x + block_width]
                    # Apply HSV enhancement to the current block
                    block = self.apply_hsv_enhancement(block)

                    # Place the processed block back into the image
                    image[y:y + block_height, x:x + block_width] = block

        # Convert the numpy array back to a PIL image
        img = Image.fromarray(image)
        return img
    def apply_hsv_enhancement(self, block):
        # Convert block to HSV

        orig_dtype = block.dtype
        # img = convert_image_dtype(block, torch.float32)
        #
        # img = _rgb2hsv(img)
        # h, s, v = img.unbind(dim=-3)
        # h = (h + hue_factor) % 1.0
        # img = torch.stack((h, s, v), dim=-3)
        # img_hue_adj = _hsv2rgb(img)
        #
        # return convert_image_dtype(img_hue_adj, orig_dtype)

        H, S, V = cv2.split(cv2.cvtColor(block, cv2.COLOR_BGR2HSV))

        # Define means and standard deviations for hue and saturation
        # mean_h = 128
        # stddev_h = 0
        # mean_s = 50
        # stddev_s = 0

        mean_h = 128
        stddev_h = 30
        mean_s = 50
        stddev_s = 20
        # Generate Gaussian noise and apply it to the H and S channels
        h_noise = np.random.normal(mean_h, stddev_h, H.shape).astype(np.uint8)
        s_noise = np.random.normal(mean_s, stddev_s, S.shape).astype(np.uint8)

        # Add noise to H and S channels
        H = cv2.add(H, h_noise)
        S = cv2.add(S, s_noise)

        # Merge modified channels back to HSV
        HSV = cv2.merge((H, S, V))
        # Convert HSV back to BGR
        enhanced_block = cv2.cvtColor(HSV, cv2.COLOR_HSV2BGR)
        return enhanced_block



