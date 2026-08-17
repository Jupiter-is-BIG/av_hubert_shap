# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Visual (video-side) distortion functions, for SHAP robustness analysis of AV-HuBERT
under degraded visual conditions -- the video-side analogue of the audio noise/SNR
perturbations already supported via `noise_wav`/`noise_prob`/`noise_snr`.

Frame-level distortions (CS, CC, BW, GNC, GB, JPEG) operate on a single BGR uint8
frame and are applied before AV-HuBERT's grayscale mouth-crop conversion. VC operates
on the whole clip via ffmpeg re-encoding, since compression artifacts are not a
per-frame operation.
"""

import math
import os
import random

import cv2
import numpy as np


def bgr2ycbcr(img_bgr):
    img_bgr = img_bgr.astype(np.float32)
    img_ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCR_CB)
    img_ycbcr = img_ycrcb[:, :, (0, 2, 1)].astype(np.float32)
    # to [16/255, 235/255]
    img_ycbcr[:, :, 0] = (img_ycbcr[:, :, 0] * (235 - 16) + 16) / 255.0
    # to [16/255, 240/255]
    img_ycbcr[:, :, 1:] = (img_ycbcr[:, :, 1:] * (240 - 16) + 16) / 255.0

    return img_ycbcr


def ycbcr2bgr(img_ycbcr):
    img_ycbcr = img_ycbcr.astype(np.float32)
    # to [0, 1]
    img_ycbcr[:, :, 0] = (img_ycbcr[:, :, 0] * 255.0 - 16) / (235 - 16)
    # to [0, 1]
    img_ycbcr[:, :, 1:] = (img_ycbcr[:, :, 1:] * 255.0 - 16) / (240 - 16)
    img_ycrcb = img_ycbcr[:, :, (0, 2, 1)].astype(np.float32)
    img_bgr = cv2.cvtColor(img_ycrcb, cv2.COLOR_YCR_CB2BGR)

    return img_bgr


def color_saturation(img, param):
    ycbcr = bgr2ycbcr(img)
    ycbcr[:, :, 1] = 0.5 + (ycbcr[:, :, 1] - 0.5) * param
    ycbcr[:, :, 2] = 0.5 + (ycbcr[:, :, 2] - 0.5) * param
    img = ycbcr2bgr(ycbcr).astype(np.uint8)

    return img


def color_contrast(img, param):
    img = img.astype(np.float32) * param
    img = img.astype(np.uint8)

    return img


def block_wise(img, param):
    img = img.copy()
    width = 3
    block = np.ones((width, width, 3)).astype(int) * 128
    param = min(img.shape[0], img.shape[1]) // 96 * param
    for _ in range(param):
        r_w = random.randint(0, img.shape[1] - 1 - width)
        r_h = random.randint(0, img.shape[0] - 1 - width)
        img[r_h:r_h + width, r_w:r_w + width, :] = block

    return img


def gaussian_noise_color(img, param):
    ycbcr = bgr2ycbcr(img) / 255
    size_a = ycbcr.shape
    b = (ycbcr + math.sqrt(param) *
         np.random.randn(size_a[0], size_a[1], size_a[2])) * 255
    b = ycbcr2bgr(b)
    img = np.clip(b, 0, 255).astype(np.uint8)

    return img


def gaussian_blur(img, param):
    img = cv2.GaussianBlur(img, (param, param), param * 1.0 / 6)

    return img


def jpeg_compression(img, param):
    h, w, _ = img.shape
    s_h = h // param
    s_w = w // param
    img = cv2.resize(img, (s_w, s_h))
    img = cv2.resize(img, (w, h))

    return img


def video_compression(vid_in, vid_out, param):
    cmd = f'ffmpeg -y -loglevel error -i "{vid_in}" -crf {param} "{vid_out}"'
    os.system(cmd)

    return


# Severity levels 1 (mildest) -> 5 (worst), matching the standard AVSR robustness
# benchmark used for e.g. LRS3 noisy/distorted evaluation.
DISTORTION_LEVELS = {
    "CS": [0.4, 0.3, 0.2, 0.1, 0.0],        # color saturation, smaller = worse
    "CC": [0.52, 0.42, 0.32, 0.22, 0.12],      # smaller, worse (factor of contrast change)
    "BW": [64, 128, 256, 512, 1024],                  # larger, worse (num of null blocks)
    "GNC": [0.008, 0.016, 0.032, 0.064, 0.128],     # larger, worse (variance of Gaussian noise)
    "GB": [7, 11, 19, 31, 51],                       # larger, worse (kernel size for sd for Gaussian blur)
    "JPEG": [2, 5, 8, 11, 14],                     # larger, worse (image reduce factor for downsample compression)
    "VC": [35, 40, 45, 50, 55],                     # larger CRF, worse
}

FRAME_DISTORTION_FUNCS = {
    "CS": color_saturation,
    "CC": color_contrast,
    "BW": block_wise,
    "GNC": gaussian_noise_color,
    "GB": gaussian_blur,
    "JPEG": jpeg_compression,
}

FRAME_DISTORTION_TYPES = list(FRAME_DISTORTION_FUNCS.keys())
ALL_DISTORTION_TYPES = FRAME_DISTORTION_TYPES + ['VC']


def get_distortion_param(dist_type, level):
    # level starts from 1, list starts from 0
    return DISTORTION_LEVELS[dist_type][level - 1]


def resolve_distortion(dist_type, level):
    """Resolve 'random' type/level selections to concrete values."""
    if dist_type == 'random':
        dist_type = random.choice(ALL_DISTORTION_TYPES)
    if str(level) == 'random':
        level = random.randint(1, 5)
    else:
        level = int(level)
    return dist_type, level


def apply_frame_distortion(frame_bgr, dist_type, level):
    """Apply a per-frame distortion (all types except 'VC') to a single BGR uint8 frame."""
    param = get_distortion_param(dist_type, level)
    return FRAME_DISTORTION_FUNCS[dist_type](frame_bgr, param)
