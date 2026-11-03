"""
Fast CPU replacement for torchvision.transforms.v2.functional.affine.

The pipeline crops and pastes faces by warping the *whole frame* with torchvision's affine,
which on CPU goes through grid_sample and costs ~130 ms per call on a 1080p frame. Without
CUDA, that dominated frame time. cv2.warpAffine computes the same bilinear warp (same
sampling grid, zero padding) in a few ms. GPU tensors and unsupported arguments still go
through torchvision.
"""
import math

import cv2
import numpy as np
import torch
from torchvision.transforms import v2
from torchvision.transforms.v2 import functional as F
from torchvision.transforms.v2.functional._geometry import _get_inverse_affine_matrix

_original_affine = F.affine
# Below this torchvision is already fast enough and handles many-channel tensors better
_MIN_PIXELS = 256 * 256
_CV2_INTERP = {
    v2.InterpolationMode.BILINEAR: cv2.INTER_LINEAR,
    v2.InterpolationMode.NEAREST: cv2.INTER_NEAREST,
}


def _cv2_matrix(width, height, angle, translate, scale, shear, center):
    # Same math as torchvision's affine_image + _affine_grid: the inverse matrix acts on
    # coordinates relative to the image centre, with pixel centres at +0.5
    if center is None:
        center_f = [0.0, 0.0]
    else:
        center_f = [(c - s * 0.5) for c, s in zip(center, [width, height])]
    inv = np.array(_get_inverse_affine_matrix(center_f, angle, [float(t) for t in translate], scale, shear),
                   dtype=np.float64).reshape(2, 3)
    inv3 = np.vstack([inv, [0.0, 0.0, 1.0]])
    cx, cy = width * 0.5, height * 0.5
    to_rel = np.array([[1, 0, 0.5 - cx], [0, 1, 0.5 - cy], [0, 0, 1]], dtype=np.float64)
    to_abs = np.array([[1, 0, cx - 0.5], [0, 1, cy - 0.5], [0, 0, 1]], dtype=np.float64)
    return (to_abs @ inv3 @ to_rel)[:2]


def fast_affine(inpt, angle, translate, scale, shear, interpolation=v2.InterpolationMode.NEAREST,
                fill=None, center=None):
    if isinstance(interpolation, int):
        interpolation = v2.InterpolationMode.NEAREST if interpolation == 0 else interpolation
    if (type(inpt) is not torch.Tensor or inpt.device.type != 'cpu' or inpt.ndim < 2
            or inpt.shape[-1] * inpt.shape[-2] < _MIN_PIXELS
            or inpt.dtype not in (torch.uint8, torch.float32) or interpolation not in _CV2_INTERP
            or not (fill is None or (isinstance(fill, (int, float)) and fill == 0))
            or inpt.requires_grad):
        return _original_affine(inpt, angle=angle, translate=translate, scale=scale, shear=shear,
                                interpolation=interpolation, fill=fill, center=center)

    if isinstance(shear, (int, float)):
        shear = [float(shear), 0.0]
    shear = [float(s) for s in shear] if len(shear) == 2 else [float(shear[0]), 0.0]
    if not isinstance(angle, (int, float)):
        angle = float(angle)
    if isinstance(scale, (np.floating, np.integer)):
        scale = float(scale)
    if math.isclose(scale, 0.0):
        return _original_affine(inpt, angle=angle, translate=translate, scale=scale, shear=shear,
                                interpolation=interpolation, fill=fill, center=center)

    shape = inpt.shape
    height, width = shape[-2], shape[-1]
    matrix = _cv2_matrix(width, height, angle, translate, scale, shear, center)

    # cv2 wants HxWxC with at most 4 channels per call
    planes = inpt.reshape(-1, height, width).permute(1, 2, 0).contiguous().numpy()
    channels = planes.shape[2]
    flags = _CV2_INTERP[interpolation] | cv2.WARP_INVERSE_MAP
    out = np.empty_like(planes)
    for start in range(0, channels, 4):
        chunk = planes[:, :, start:start + 4]
        warped = cv2.warpAffine(chunk, matrix, (width, height), flags=flags,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        out[:, :, start:start + 4] = warped.reshape(height, width, -1)
    return torch.from_numpy(out).permute(2, 0, 1).reshape(shape)


def install():
    """Route every v2.functional.affine call in the app through fast_affine."""
    if F.affine is not fast_affine:
        F.affine = fast_affine
