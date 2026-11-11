"""Step 3a - Shape with a CNN (PyTorch + timm).

* Input: greyscale crop (1 channel) so the network cannot cheat with colour.
* Backbone from timm. `pretrained=True` uses ImageNet weights - ONLY if the
  supervisors allow it (open question 1 on slide 9). With pretrained=False
  everything is trained from scratch on LEGO data.
* The checkpoint stores the class list, so inference always uses the
  same order as training.
"""
from __future__ import annotations

import numpy as np

MEAN, STD = 0.5, 0.25


def build_model(name: str, num_classes: int, pretrained: bool = False, drop_rate: float = 0.2):
    import timm

    # timm adapts the first conv to 1 input channel (sums RGB weights when pretrained)
    return timm.create_model(name, pretrained=pretrained, num_classes=num_classes, in_chans=1, drop_rate=drop_rate)


def save_checkpoint(path, model, classes, model_name, image_size, extra=None):
    import torch

    torch.save(
        {
            "state_dict": model.state_dict(),
            "classes": list(classes),
            "model_name": model_name,
            "image_size": image_size,
            "extra": extra or {},
        },
        path,
    )


class CNNShape:
    def __init__(self, ckpt_path, device: str | None = None):
        import torch

        self.torch = torch
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        self.classes = ck["classes"]
        self.size = ck["image_size"]
        self.model = build_model(ck["model_name"], len(self.classes), pretrained=False)
        self.model.load_state_dict(ck["state_dict"])
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device).eval()

    def _prep(self, greys):
        import cv2

        arr = np.stack([cv2.resize(g, (self.size, self.size), interpolation=cv2.INTER_AREA) for g in greys])
        x = (arr.astype(np.float32) / 255.0 - MEAN) / STD
        return self.torch.from_numpy(x[:, None])

    def predict(self, greys, topk: int = 3, tta: bool = True):
        """Top-k (class, probability) per crop. TTA averages 4 rotations
        (a brick lying on the table has no 'up')."""
        torch = self.torch
        with torch.no_grad():
            x = self._prep(greys).to(self.device)
            views = [x] + ([torch.rot90(x, k, dims=(2, 3)) for k in (1, 2, 3)] if tta else [])
            p = torch.stack([self.model(v).softmax(1) for v in views]).mean(0).cpu().numpy()
        out = []
        for row in p:
            order = np.argsort(row)[::-1][:topk]
            out.append([(self.classes[i], float(row[i])) for i in order])
        return out
