"""NeRF Synthetic / Blender dataset loader."""

import json
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset


class NerfSyntheticDataset(Dataset):
    """Loads a NeRF Synthetic scene (Blender format).

    Each item returns (image, camera) where:
      - image: (H, W, 3) float32 tensor in [0, 1], alpha-composited on white bg
      - camera: dict with 'viewmat' (4x4), 'K' (3x3), 'width', 'height'
    """

    def __init__(self, root: str, split: str = "train", scale: float = 1.0):
        self.root = Path(root)
        self.scale = scale

        with open(self.root / f"transforms_{split}.json") as f:
            meta = json.load(f)

        self.camera_angle_x = meta["camera_angle_x"]
        self.frames = meta["frames"]

        self.images = []
        self.cameras = []
        for frame in self.frames:
            img_path = self.root / f"{frame['file_path']}.png"
            if not img_path.exists():
                img_path = self.root / f"{frame['file_path'].lstrip('./')}.png"

            img = Image.open(img_path)
            if scale != 1.0:
                w, h = img.size
                img = img.resize(
                    (int(w * scale), int(h * scale)), Image.LANCZOS
                )

            img_np = np.array(img, dtype=np.float32) / 255.0

            if img_np.shape[2] == 4:
                alpha = img_np[:, :, 3:4]
                rgb = img_np[:, :, :3] * alpha + (1.0 - alpha)
            else:
                rgb = img_np[:, :, :3]

            self.images.append(torch.from_numpy(rgb))

            c2w = torch.tensor(frame["transform_matrix"], dtype=torch.float32)
            w2c = torch.linalg.inv(c2w)

            h, w_px = rgb.shape[:2]
            focal = 0.5 * w_px / np.tan(0.5 * self.camera_angle_x)

            K = torch.tensor(
                [[focal, 0, w_px / 2.0],
                 [0, focal, h / 2.0],
                 [0, 0, 1]],
                dtype=torch.float32,
            )

            self.cameras.append({
                "viewmat": w2c,
                "K": K,
                "width": w_px,
                "height": h,
            })

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.cameras[idx]
