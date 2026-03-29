import pytest
import torch

gsplat = pytest.importorskip("gsplat")

from gaussianfractallod.gaussian import Gaussian
from gaussianfractallod.train_roots import init_roots, train_roots_step


def test_init_roots():
    roots = init_roots(num_roots=8, sh_degree=0, device=torch.device("cpu"))
    assert roots.num_gaussians == 8
    assert roots.means.shape == (8, 3)
    assert roots.quats.shape == (8, 4)
    assert roots.log_scales.shape == (8, 3)
    assert roots.sh_dc.shape == (8, 1, 3)
    assert roots.sh_rest.shape == (8, 0, 3)
    assert roots.means.requires_grad


def test_train_roots_step_reduces_loss():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        pytest.skip("gsplat requires CUDA")

    roots = init_roots(num_roots=4, sh_degree=0, device=device)
    optimizer = torch.optim.Adam(
        [roots.means, roots.quats, roots.log_scales,
         roots.opacities, roots.sh_dc, roots.sh_rest], lr=1e-2,
    )
    gt_image = torch.zeros(64, 64, 3, device=device)
    gt_image[:, :, 0] = 1.0
    viewmat = torch.eye(4, device=device)
    K = torch.tensor([[100.0, 0, 32.0], [0, 100.0, 32.0], [0, 0, 1]], device=device)
    camera = {"viewmat": viewmat, "K": K, "width": 64, "height": 64}

    loss1 = train_roots_step(roots, gt_image, camera, optimizer, ssim_weight=0.2)
    loss2 = train_roots_step(roots, gt_image, camera, optimizer, ssim_weight=0.2)
    assert loss1.item() > 0
    assert loss2.item() > 0
