import torch
from gaussianfractallod.gaussian import Gaussian
from gaussianfractallod.subdivide import subdivide_to_8, _binary_cut_along_axis


def _make_parent():
    return Gaussian(
        means=torch.tensor([[0.0, 0.0, 0.0]]),
        quats=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),   # identity quaternion
        log_scales=torch.tensor([[0.0, 0.0, 0.0]]),     # unit scale
        opacities=torch.tensor([[2.0]]),  # sigmoid(2) ~ 0.88
        sh_coeffs=torch.tensor([[0.5, 0.3, 0.1]]),
    )


def test_binary_cut_doubles_count():
    parent = _make_parent()
    children = _binary_cut_along_axis(parent, axis=0)
    assert children.num_gaussians == 2


def test_subdivide_produces_8x():
    parent = _make_parent()
    children = subdivide_to_8(parent)
    assert children.num_gaussians == 8


def test_subdivide_batch():
    N = 5
    parents = Gaussian(
        means=torch.randn(N, 3),
        quats=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).expand(N, 4).contiguous(),
        log_scales=torch.zeros(N, 3),
        opacities=torch.ones(N, 1) * 2.0,
        sh_coeffs=torch.randn(N, 3),
    )
    children = subdivide_to_8(parents)
    assert children.num_gaussians == 40  # 5 x 8


def test_subdivide_children_near_parent():
    parent = _make_parent()
    children = subdivide_to_8(parent)
    # All children should be within parent's extent (roughly)
    dists = (children.means - parent.means).norm(dim=-1)
    assert dists.max() < 3.0  # within 3 sigma


def test_subdivide_children_covariance_positive_definite():
    parent = Gaussian(
        means=torch.tensor([[1.0, 2.0, 3.0]]),
        quats=torch.randn(1, 4),
        log_scales=torch.randn(1, 3),
        opacities=torch.tensor([[2.0]]),
        sh_coeffs=torch.randn(1, 3),
    )
    children = subdivide_to_8(parent)
    cov = children.covariance()
    eigvals = torch.linalg.eigvalsh(cov)
    assert (eigvals > 0).all()


def test_subdivide_opacity_reasonable():
    parent = _make_parent()
    children = subdivide_to_8(parent)
    # Children should have lower opacity than parent but not zero
    parent_prob = torch.sigmoid(parent.opacities).item()
    for i in range(children.num_gaussians):
        child_prob = torch.sigmoid(children.opacities[i:i+1]).item()
        assert 0.01 < child_prob < parent_prob, f"child opacity {child_prob} out of range"
