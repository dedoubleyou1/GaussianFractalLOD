import pytest
import torch

gsplat = pytest.importorskip("gsplat")

from gaussianfractallod.gaussian import Gaussian
from gaussianfractallod.split_tree import SplitTree
from gaussianfractallod.train_splits import train_split_level_step


def _make_frozen_roots(device):
    return Gaussian(
        means=torch.tensor([[0.0, 0.0, 3.0]], device=device),
        scales=torch.tensor([[-1.0, -1.0, -1.0]], device=device),
        opacities=torch.tensor([[2.0]], device=device),
        sh_coeffs=torch.tensor([[0.5, 0.2, 0.1]], device=device),
    )


def test_split_training_step():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        pytest.skip("gsplat requires CUDA")

    roots = _make_frozen_roots(device)
    tree = SplitTree(num_roots=1, sh_dim=3).to(device)
    tree.add_level()
    optimizer = torch.optim.Adam(tree.level_parameters(0), lr=1e-2)

    gt_image = torch.rand(64, 64, 3, device=device)
    camera = {
        "viewmat": torch.eye(4, device=device),
        "K": torch.tensor([[100.0, 0, 32.0], [0, 100.0, 32.0], [0, 0, 1]], device=device),
        "width": 64, "height": 64,
    }

    loss = train_split_level_step(
        roots, tree, target_depth=1, gt_image=gt_image, camera=camera,
        optimizer=optimizer, ssim_weight=0.2,
    )
    assert loss.item() > 0
    assert tree.levels[0].mass_logit.grad is not None
    assert tree.levels[0].position_split.grad is not None
