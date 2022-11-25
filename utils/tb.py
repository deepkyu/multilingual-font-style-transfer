import torch
import numpy as np

def magic_image_handler(img):
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()
    if img.ndim == 3:
        img = img.transpose((1, 2, 0))
    elif img.ndim == 2:
        img = np.repeat(img[..., np.newaxis], 3, axis=2)
    elif img.ndim == 4:
        img = img[:4]  # first 4 batch
        img = np.concatenate(img, axis=-1)
        img = img.transpose((1, 2, 0))
    elif img.ndim == 5:
        img = img[:4]  # first 4 batch
        img = np.concatenate(img, axis=-2)
        img = np.concatenate(img, axis=-1)
        img = img.transpose((1, 2, 0))
    else:
        raise ValueError(f'img ndim is {img.ndim}, should be 2~4')
    if img.shape[-1] != 1 or img.shape[-1] != 3:
        img = np.expand_dims(np.concatenate([img[..., i] for i in range(img.shape[-1])], axis=0), -1)
    img = np.clip(img, a_min=0, a_max=255)
    return img