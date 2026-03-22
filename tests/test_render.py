import pytest
import torch

gsplat = pytest.importorskip("gsplat")

from gaussianfractallod.gaussian import Gaussian
from gaussianfractallod.render import render_gaussians


def test_render_produces_image():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        pytest.skip("gsplat requires CUDA")

    gaussians = Gaussian(
        means=torch.tensor([[0.0, 0.0, 3.0]], device=device),
        scales=torch.tensor([[-1.0, -1.0, -1.0]], device=device),
        opacities=torch.tensor([[5.0]], device=device),
        sh_coeffs=torch.tensor([[1.0, 0.0, 0.0]], device=device),
    )

    viewmat = torch.eye(4, device=device)
    K = torch.tensor(
        [[200.0, 0, 64.0], [0, 200.0, 64.0], [0, 0, 1]], device=device
    )

    image = render_gaussians(
        gaussians, viewmat=viewmat, K=K,
        width=128, height=128,
        background=torch.ones(3, device=device),
    )
    assert image.shape == (128, 128, 3)
    assert image.dtype == torch.float32
    assert not torch.allclose(image, torch.ones_like(image), atol=0.1)
