"""Shared class order and RGB label contract."""
import numpy as np

CLASS_NAMES = ('rock', 'pore', 'heterogeneous_oil', 'throat_oil', 'wall_oil', 'dead_end_oil')
CLASS_NAMES_ZH = ('岩石', '孔道', '非均质剩余油', '孔喉剩余油', '壁面剩余油', '盲端剩余油')
COLORS = np.array([(255,255,255), (128,128,128), (255,0,255), (0,255,0), (0,0,255), (255,255,0)], dtype=np.uint8)
THRESHOLDS = (80, 40, 50, 60, 70, 50)


def rgb_to_labels(rgb):
    """Match historical RGB tolerances, rejecting previously silent unknown colors."""
    rgb = np.asarray(rgb, dtype=np.int32)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError('Expected an H×W×3 RGB mask')
    label = np.full(rgb.shape[:2], -1, dtype=np.int64)
    for index, (color, threshold) in enumerate(zip(COLORS, THRESHOLDS)):
        delta = rgb - color.astype(np.int32)
        label[(delta ** 2).sum(axis=-1) <= threshold ** 2] = index
    if np.any(label < 0):
        raise ValueError(f'Mask has {np.count_nonzero(label < 0)} pixels outside the six-color palette')
    return label


def read_mask(path):
    from PIL import Image
    with Image.open(path) as im:
        if im.mode in ('L', 'I', 'I;16'):
            array = np.asarray(im, dtype=np.int64)
            if array.min() < 0 or array.max() > 5:
                raise ValueError(f'Index mask must contain only 0–5: {path}')
            return array
        return rgb_to_labels(im.convert('RGB'))
