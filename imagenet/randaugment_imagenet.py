import PIL, PIL.ImageOps, PIL.ImageDraw
from PIL import Image
import numpy as np
import random
import torch
from torchvision import transforms
import warnings

import torchvision.transforms.functional as F
import math

FILL_MEAN = False
FILL_COLOR = (0, 0, 0)
INTERPOLATION = Image.BICUBIC
PAD = 0

class ToNumpy:
    def __call__(self, pil_img):
        np_img = np.array(pil_img, dtype=np.uint8)
        if np_img.ndim < 3:
            np_img = np.expand_dims(np_img, axis=-1)
        np_img = np.rollaxis(np_img, 2)  # HWC to CHW
        return np_img


class ToTensor:
    def __init__(self, dtype=torch.float32):
        self.dtype = dtype

    def __call__(self, pil_img):
        np_img = np.array(pil_img, dtype=np.uint8)
        if np_img.ndim < 3:
            np_img = np.expand_dims(np_img, axis=-1)
        np_img = np.rollaxis(np_img, 2)  # HWC to CHW
        return torch.from_numpy(np_img).to(dtype=self.dtype)


_pil_interpolation_to_str = {
    Image.NEAREST: 'PIL.Image.NEAREST',
    Image.BILINEAR: 'PIL.Image.BILINEAR',
    Image.BICUBIC: 'PIL.Image.BICUBIC',
    Image.LANCZOS: 'PIL.Image.LANCZOS',
    Image.HAMMING: 'PIL.Image.HAMMING',
    Image.BOX: 'PIL.Image.BOX',
}


def _pil_interp(method):
    if method == 'bicubic':
        return Image.BICUBIC
    elif method == 'lanczos':
        return Image.LANCZOS
    elif method == 'hamming':
        return Image.HAMMING
    else:
        # default bilinear, do we want to allow nearest?
        return Image.BILINEAR


_RANDOM_INTERPOLATION = (Image.BILINEAR, Image.BICUBIC)


class RandomResizedCropAndInterpolation:
    """Crop the given PIL Image to random size and aspect ratio with random interpolation.

    A crop of random size (default: of 0.08 to 1.0) of the original size and a random
    aspect ratio (default: of 3/4 to 4/3) of the original aspect ratio is made. This crop
    is finally resized to given size.
    This is popularly used to train the Inception networks.

    Args:
        size: expected output size of each edge
        scale: range of size of the origin size cropped
        ratio: range of aspect ratio of the origin aspect ratio cropped
        interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.),
                 interpolation='bilinear'):
        if isinstance(size, tuple):
            self.size = size
        else:
            self.size = (size, size)
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            warnings.warn("range should be of kind (min, max)")

        if interpolation == 'random':
            self.interpolation = _RANDOM_INTERPOLATION
        else:
            self.interpolation = _pil_interp(interpolation)
        self.scale = scale
        self.ratio = ratio

    @staticmethod
    def get_params(img, scale, ratio):
        """Get parameters for ``crop`` for a random sized crop.

        Args:
            img (PIL Image): Image to be cropped.
            scale (tuple): range of size of the origin size cropped
            ratio (tuple): range of aspect ratio of the origin aspect ratio cropped

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
                sized crop.
        """
        area = img.size[0] * img.size[1]

        for attempt in range(10):
            target_area = random.uniform(*scale) * area
            log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
            aspect_ratio = math.exp(random.uniform(*log_ratio))

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if w <= img.size[0] and h <= img.size[1]:
                i = random.randint(0, img.size[1] - h)
                j = random.randint(0, img.size[0] - w)
                return i, j, h, w

        # Fallback to central crop
        in_ratio = img.size[0] / img.size[1]
        if in_ratio < min(ratio):
            w = img.size[0]
            h = int(round(w / min(ratio)))
        elif in_ratio > max(ratio):
            h = img.size[1]
            w = int(round(h * max(ratio)))
        else:  # whole image
            w = img.size[0]
            h = img.size[1]
        i = (img.size[1] - h) // 2
        j = (img.size[0] - w) // 2
        return i, j, h, w

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be cropped and resized.

        Returns:
            PIL Image: Randomly cropped and resized image.
        """
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        if isinstance(self.interpolation, (tuple, list)):
            interpolation = random.choice(self.interpolation)
        else:
            interpolation = self.interpolation
        # print(self.size, np.array(F.resized_crop(img, i, j, h, w, self.size, interpolation)).shape)
        return F.resized_crop(img, i, j, h, w, self.size, interpolation)

    def __repr__(self):
        if isinstance(self.interpolation, (tuple, list)):
            interpolate_str = ' '.join([_pil_interpolation_to_str[x] for x in self.interpolation])
        else:
            interpolate_str = _pil_interpolation_to_str[self.interpolation]
        format_string = self.__class__.__name__ + '(size={0}'.format(self.size)
        format_string += ', scale={0}'.format(tuple(round(s, 4) for s in self.scale))
        format_string += ', ratio={0}'.format(tuple(round(r, 4) for r in self.ratio))
        format_string += ', interpolation={0})'.format(interpolate_str)
        return format_string

    def get_transform_str(self):
        return 'ResizedCropAndInt_scale_%s' % self.scale

def getRandomResizedCropAndInterpolationdef(IMAGE_SIZE, scale):
    def RandomResizedCropAndInterpolation_op(img):
        img = RandomResizedCropAndInterpolation(size=int(IMAGE_SIZE), scale=scale, interpolation='bicubic')(img)
        img = transforms.RandomHorizontalFlip()(img)
        return img

    return RandomResizedCropAndInterpolation_op

def getRandomResizedCropAndInterpolation_op(IMAGE_SIZE):
    def RandomResizedCropAndInterpolation_op(img, v):
        v = np.abs(v)
        img = RandomResizedCropAndInterpolation(size=int(IMAGE_SIZE), scale=(np.abs(v), np.abs(v)+0.01), interpolation='bicubic')(img)
        img = transforms.RandomHorizontalFlip()(img)
        return img

    return RandomResizedCropAndInterpolation_op

def AutoContrast(img, v):
    return PIL.ImageOps.autocontrast(img, v)

def Posterize(img, v):
    v = int(v)
    return PIL.ImageOps.posterize(img, v)

# @ReflectionPadding
def ShearX(img, v):  # [-0.3, 0.3]
    if random.random() > 0.5:
        v = -v
    flipped = False
    if random.random() > 0.5:
        flipped = True
        img = img.transpose(PIL.Image.FLIP_TOP_BOTTOM)
    if FILL_MEAN:
        fillcolor = tuple([int(x) for x in PIL.ImageStat.Stat(img).mean])
    else:
        fillcolor = FILL_COLOR

        
    img = img.transform(img.size, PIL.Image.AFFINE, (1, v, -np.sign(v) * PAD, 0, 1, PAD), resample=INTERPOLATION, fillcolor=(0, 0, 0))
    img = img.transform(img.size, PIL.Image.AFFINE, (1, 0, np.sign(v) * PAD, 0, 1, -PAD), resample=0, fillcolor=(0, 0, 0))
    if flipped:
        img = img.transpose(PIL.Image.FLIP_TOP_BOTTOM)
    return img
#     return img.transform(img.size, PIL.Image.AFFINE, (1, v, 0, 0, 1, 0), resample=INTERPOLATION, fillcolor=fillcolor)

# @ReflectionPadding
def ShearY(img, v):  # [-0.3, 0.3]
    if random.random() > 0.5:
        v = -v
    flipped = False
    if random.random() > 0.5:
        flipped = True
        img = img.transpose(PIL.Image.FLIP_LEFT_RIGHT)
    if FILL_MEAN:
        fillcolor = tuple([int(x) for x in PIL.ImageStat.Stat(img).mean])
    else:
        fillcolor = FILL_COLOR
    img = img.transform(img.size, PIL.Image.AFFINE, (1, 0, PAD, v, 1, -np.sign(v) * PAD), resample=INTERPOLATION, fillcolor=(0, 0, 0))
    img = img.transform(img.size, PIL.Image.AFFINE, (1, 0, -PAD, 0, 1, np.sign(v) * PAD), resample=0, fillcolor=(0, 0, 0))
    if flipped:
        img = img.transpose(PIL.Image.FLIP_LEFT_RIGHT)
    return img
#     return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, v, 1, 0), resample=INTERPOLATION, fillcolor=fillcolor)


# @ReflectionPadding
def getTranslateX(IMAGE_SIZE):
    IMAGE_SIZE = int(IMAGE_SIZE)
    def TranslateX(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
        if random.random() > 0.5:
            v = -v
    #     v = v * img.size[0]
        # WARNING: IMAGE SIZE HARD-CODED!
        v = v * IMAGE_SIZE
        if FILL_MEAN:
            fillcolor = tuple([int(x) for x in PIL.ImageStat.Stat(img).mean])
        else:
            fillcolor = FILL_COLOR
        return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0), fillcolor=fillcolor)
    return TranslateX


# @ReflectionPadding
def getTranslateY(IMAGE_SIZE):
    IMAGE_SIZE = int(IMAGE_SIZE)
    def TranslateY(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
        if random.random() > 0.5:
            v = -v
    #     v = v * img.size[1]
        # WARNING: IMAGE SIZE HARD-CODED!
        v = v * IMAGE_SIZE
        if FILL_MEAN:
            fillcolor = tuple([int(x) for x in PIL.ImageStat.Stat(img).mean])
        else:
            fillcolor = FILL_COLOR
        return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v), fillcolor=fillcolor)
    return TranslateY

# @ReflectionPadding
def Rotate(img, v):  # [-30, 30]
    if random.random() > 0.5:
        v = -v
    if FILL_MEAN:
        fillcolor = tuple([int(x) for x in PIL.ImageStat.Stat(img).mean])
    else:
        fillcolor = FILL_COLOR
    return img.rotate(v, INTERPOLATION, fillcolor=fillcolor)

def Solarize(img, v):  # [0, 256]
    assert 0 <= v <= 256
    return PIL.ImageOps.solarize(img, v)

def SolarizeAdd(img, v):
    return Image.blend(img, Solarize(img, v), 0.5)

def Contrast(img, v):  # [0, 0.9]
    # TODO
    # sign = np.random.choice([-1, 1], 1)[0]
    sign = 1
    return PIL.ImageEnhance.Contrast(img).enhance(1 + sign * v)

def Color(img, v):  # [0, 0.9]
    # TODO
    # sign = np.random.choice([-1, 1], 1)[0]
    sign = 1
    return PIL.ImageEnhance.Color(img).enhance(1 + sign * v)

def Brightness(img, v):  # [0, 0.9]
    # TODO
    # sign = np.random.choice([-1, 1], 1)[0]
    sign = 1
    return PIL.ImageEnhance.Brightness(img).enhance(1 + sign * v)

def Sharpness(img, v):  # [0, 0.9]
    # TODO
    # sign = np.random.choice([-1, 1], 1)[0]
    sign = 1
    return PIL.ImageEnhance.Sharpness(img).enhance(1 + sign * v)

def Identity(img, _):
    return img

def Cutout(img, v, fcolor=None):  # [0, 60] => percentage: [0, 0.2]
    if v <= 0.:
        return img
    v = v * img.size[0]
    return CutoutAbs(img, v, fcolor)

def CutoutAbs(img, v, fcolor=None):  # [0, 60] => percentage: [0, 0.2]
    # assert 0 <= v <= 20
    if v < 0:
        return img
    w, h = img.size
    x0 = np.random.uniform(w)
    y0 = np.random.uniform(h)

    x0 = int(max(0, x0 - v / 2.))
    y0 = int(max(0, y0 - v / 2.))
    x1 = min(w, x0 + v)
    y1 = min(h, y0 + v)

    xy = (x0, y0, x1, y1)
    if FILL_MEAN:
        color = tuple([int(x) for x in PIL.ImageStat.Stat(img).mean])
    else:
        color = FILL_COLOR
    if fcolor is not None:
        color = fcolor
    img = img.copy()
    PIL.ImageDraw.Draw(img).rectangle(xy, color)
    return img


# def Cutout(img, v):  # [0, 60] => percentage: [0, 0.2]
#     if v <= 0.:
#         return img
#     v = int(v * img.size[0])
#     return CutoutAbs(img, v)

# def CutoutAbs(img, v):  # [0, 60] => percentage: [0, 0.2]
#     # assert 0 <= v <= 20
#     if v < 0:
#         return img
#     w, h = img.size
#     x0 = np.random.uniform(w)
#     y0 = np.random.uniform(h)

#     x0 = int(max(0, x0 - v / 2.))
#     y0 = int(max(0, y0 - v / 2.))
#     x1 = min(w, x0 + v)
#     y1 = min(h, y0 + v)
#     v = x1 - x0

#     xy = (x0, y0, x1, y1)
    
#     flipped = False
#     if random.random() > 0.5:
#         flipped = True
#         img = img.transpose(PIL.Image.TRANSPOSE)
#     new_img = np.array(img)
    
#     left_w = int((1. * x0 / w) * v)
#     right_w = int((1. * (w - x1) / w * v))
#     while left_w + right_w < v:
#         if random.random() > 0.5 and x0 > left_w + 1:
#             left_w += 1
#         elif x1 + right_w < w - 1:
#             right_w += 1
#     new_img[x0:x0 + left_w, :][:, y0:y1] = new_img[x0:x0 - left_w:-1, :][:, y0:y1].copy()
#     new_img[x1 - right_w:x1, :][:, y0:y1] = new_img[x1 + right_w:x1:-1, :][:, y0:y1].copy()
    
#     new_img = Image.fromarray(new_img)
#     if flipped:
#         new_img = new_img.transpose(PIL.Image.TRANSPOSE)
#     return new_img


def Invert(img, _):
    return PIL.ImageOps.invert(img)


def Equalize(img, _):
    return PIL.ImageOps.equalize(img)


def identity():
    return [
        (Identity, 0., 1.0),
    ]

def geometric():
    return [
        (ShearX, 0., 0.5),  # 0
        (ShearY, 0., 0.5),  # 1
        (TranslateX, 0., 0.45),  # 2
        (TranslateY, 0., 0.45),  # 3
        (Rotate, 0, 40),  # 4
        (Cutout, 0, 0.5),  # 14
    ]

def color():
    return [
        (AutoContrast, 0, 10),  # 5
        (Contrast, 0, 0.8),  # 10
        (Color, 0, 0.9),  # 11
        (Brightness, 0, 0.8),  # 12
    ]

def quality():
    return [
        (Posterize, 8, 2),
        (Sharpness, 0, 0.9),
    ]

def bad():
    return [
        (Solarize, 256, 128),  # 8
        (SolarizeAdd, 256, 128),
    ]

class GroupAblationRandAugment:
    def __init__(self, n, m, ignore=[], rand_m=False):
        self.augment_list = []
        for l in [identity, geometric, color, quality, bad]:
            if l.__name__ not in ignore:
                self.augment_list += l()
        self.n = n
        self.m = m      # [0, 30]
        self.rand_m = rand_m

    def __call__(self, img):
        inds = np.random.choice(len(self.augment_list), size=self.n, replace=False)
        ops = [self.augment_list[i] for i in inds]
        for op, minval, maxval in ops:
            if self.rand_m:
                m = np.random.uniform(0, self.m)
            else:
                m = float(self.m)
            val = (m / 30) * float(maxval - minval) + minval
            img = op(img, val)

        return img

def augment_list(image_size):  # 16 oeprations and their ranges
    # https://github.com/google-research/uda/blob/master/image/randaugment/policies.py#L57
    l = [
        (Identity, 0., 1.0), # 0
        (ShearX, 0., 0.5),  # 1
        (ShearY, 0., 0.5),  # 2
        (getTranslateX(image_size), 0., 0.45),  # 3
        (getTranslateY(image_size), 0., 0.45),  # 4
        (Rotate, 0, 40),  # 5
        (AutoContrast, 0, 10),  # 6
        (Solarize, 256, 128),  # 7
        (SolarizeAdd, 256, 128),  # 8
        (Posterize, 8, 2),  # 9
        (Contrast, 0, 0.8),  # 10
        (Color, 0, 0.9),  # 11
        (Brightness, 0, 0.8),  # 12
        (Sharpness, 0, 0.9),  # 13
        (Cutout, 0, 0.5),  # 14
        (getRandomResizedCropAndInterpolation_op(image_size), 0.99, 0.08), # 15
        (Invert, 0, 1),  # 16
        (Equalize, 0, 1),  # 17
    ]
    return l



class BetterRandAugment:
    def __init__(self, n, m, rand_m=False, resample=True, verbose=False,
                 transform=None, true_m0=False, randomize_sign=True, image_size=None):
        self.n = n
        self.m = m      # [0, 30]
        self.rand_m = rand_m
        self.augment_list = augment_list(image_size)
        self.verbose = verbose
        self.true_m0 = true_m0
        self.randomize_sign = randomize_sign
        
        self.resample = resample
        if transform is None:
            self.resample_transform()
        elif isinstance(transform, str):
            self.set_transform_str(transform)
        else:
            self.set_transform(transform)

    def resample_transform(self):
        idx_ = list(range(len(self.augment_list)))
        del idx_[15]
        self.op_inds = [15] + random.choices(idx_, k=self.n)
        self.ops = [self.augment_list[i] for i in self.op_inds]
        if self.rand_m:
            self.Ms = np.random.uniform(-self.m, self.m, self.n+1)
        else:
            self.Ms = [self.m for _ in range(self.n+1)]
        if self.verbose:
            print('Resampled transform. Current transform: ')
            print(str(self.get_transform_str()))
    
    def set_transform(self, transform):
        self.op_inds = []
        self.Ms = []
        for ind, m in transform:
            self.op_inds.append(ind)
            self.Ms.append(m)
        self.ops = [self.augment_list[i] for i in self.op_inds]
        if self.verbose:
            print('Manually set transform. Current transform: ')
            print(str(self.get_transform_str()))
    
    def get_transform(self):
        transform = []
        for ind, m in zip(self.op_inds, self.Ms):
            transform.append((ind, m))
        return transform
    
    def get_transform_str(self):
        return ''.join(str(self.get_transform()).split())
    
    def set_transform_str(self, s):
        return self.set_transform(eval(s))

    def __call__(self, img):
        if self.resample:
            if self.verbose:
                print('Updated transform!')
            self.resample_transform()

        for i, (op, minval, maxval) in enumerate(self.ops):
            m = self.Ms[i]
            if np.abs(m) < 0.5 and self.true_m0:
                continue
            if self.randomize_sign:
                if np.random.randn() < 0.5:
                    m *= -1
            if op.__name__ not in ['Contrast', 'Color', 'Brightness', 'Sharpness']:
                m = np.abs(m)
            val = (m / 30) * float(maxval - minval) + minval
            img = op(img, val)
        return img

class BRAAlwaysCutout:
    def __init__(self, n, m, rand_m=False):
        self.n = n
        self.m = m      # [0, 30]
        self.rand_m = rand_m
        self.augment_list = augment_list()
        self.augment_list = [x for x in self.augment_list if 'Cutout' not in x[0].__name__]
        print('Using augmentations:', [x[0].__name__ for x in self.augment_list])

    def __call__(self, img):
        ops = random.choices(self.augment_list, k=self.n)
        for op, minval, maxval in ops:
            if self.rand_m:
                m = np.random.uniform(0, self.m)
            else:
                m = float(self.m)
            val = (m / 30) * float(maxval - minval) + minval
            img = op(img, val)
        img = Cutout(img, 0.5, (128, 128, 128))
        
        return img
