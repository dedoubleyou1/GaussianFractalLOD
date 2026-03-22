import pytest
import torch
import json
import os
from pathlib import Path
from PIL import Image
from gaussianfractallod.data import NerfSyntheticDataset


@pytest.fixture
def mock_nerf_scene(tmp_path):
    """Create a minimal mock NeRF Synthetic scene."""
    frames = []
    for i in range(3):
        img_path = f"train/r_{i:03d}"
        frames.append({
            "file_path": f"./{img_path}",
            "transform_matrix": [
                [1, 0, 0, float(i)],
                [0, 1, 0, 0],
                [0, 0, 1, 4],
                [0, 0, 0, 1],
            ],
        })

    transforms = {"camera_angle_x": 0.6911, "frames": frames}
    with open(tmp_path / "transforms_train.json", "w") as f:
        json.dump(transforms, f)

    train_dir = tmp_path / "train"
    train_dir.mkdir()
    for i in range(3):
        img = Image.new("RGBA", (32, 32), (128, 64, 200, 255))
        img.save(train_dir / f"r_{i:03d}.png")

    return tmp_path


def test_load_dataset(mock_nerf_scene):
    dataset = NerfSyntheticDataset(str(mock_nerf_scene), split="train")
    assert len(dataset) == 3


def test_dataset_returns_image_and_camera(mock_nerf_scene):
    dataset = NerfSyntheticDataset(str(mock_nerf_scene), split="train")
    image, camera = dataset[0]
    assert image.shape == (32, 32, 3)
    assert image.dtype == torch.float32
    assert "viewmat" in camera
    assert "K" in camera
    assert camera["viewmat"].shape == (4, 4)
    assert camera["K"].shape == (3, 3)
