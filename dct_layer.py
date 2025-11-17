import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_dct as dct
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt

class VolumeDCTPatchLayer(nn.Module):
    def __init__(self, patch_size=6, is_concat=False):
        super(VolumeDCTPatchLayer, self).__init__()
        self.patch_size = patch_size
        self.is_concat = is_concat

    def pad_to_multiple(self, x, multiple):
        B, C, H, W, D = x.shape
        pad_h = (multiple - (H % multiple)) % multiple
        pad_w = (multiple - (W % multiple)) % multiple
        padding = (0, 0, 0, pad_w, 0, pad_h)
        x = F.pad(x, padding)
        return x, pad_h, pad_w

    def forward(self, x):
        B, C, H, W, D = x.shape
        x, pad_h, pad_w = self.pad_to_multiple(x, self.patch_size)

        out = []
        for b in range(B):
            channels_out = []
            for c in range(C):
                slices_out = []
                for d_idx in range(D):
                    img = x[b, c, :, :, d_idx]  # (H, W)

                    patches = img.unfold(0, self.patch_size, self.patch_size).unfold(1, self.patch_size, self.patch_size)
                    patches = patches.contiguous().view(-1, self.patch_size, self.patch_size)

                    dct_patches = dct.dct_2d(patches, norm='ortho')

                    n_patches_h = img.shape[0] // self.patch_size
                    n_patches_w = img.shape[1] // self.patch_size
                    reconstructed = dct_patches.view(n_patches_h, n_patches_w, self.patch_size, self.patch_size)
                    reconstructed = reconstructed.permute(0, 2, 1, 3).contiguous().view(img.shape[0], img.shape[1])

                    if pad_h > 0:
                        reconstructed = reconstructed[:-pad_h, :]
                    if pad_w > 0:
                        reconstructed = reconstructed[:, :-pad_w]

                    slices_out.append(reconstructed)
                channels_out.append(torch.stack(slices_out, dim=-1))
            out.append(torch.stack(channels_out))
        out = torch.stack(out)
        
        if self.is_concat:
            out = torch.cat((out, x), 1)
        
        return out
