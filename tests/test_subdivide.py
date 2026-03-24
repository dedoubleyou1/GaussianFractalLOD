import torch
from gaussianfractallod.gaussian import Gaussian
from gaussianfractallod.subdivide import subdivide_to_8, _binary_cut_along_axis


def _make_parent():
    return Gaussian(
        means=torch.tensor([[0.0, 0.0, 0.0]]),
        L_flat=torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]),  # isotropic, scale=1
        opacities=torch.tensor([[2.0]]),  # sigmoid(2) ≈ 0.88
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
    parents = Gaussian(
        means=torch.randn(5, 3),
        L_flat=torch.zeros(5, 6),
        opacities=torch.ones(5, 1) * 2.0,
        sh_coeffs=torch.randn(5, 3),
    )
    children = subdivide_to_8(parents)
    assert children.num_gaussians == 40  # 5 × 8


def test_subdivide_children_near_parent():
    parent = _make_parent()
    children = subdivide_to_8(parent)
    # All children should be within parent's extent (roughly)
    dists = (children.means - parent.means).norm(dim=-1)
    assert dists.max() < 3.0  # within 3 sigma


def test_subdivide_children_covariance_positive_definite():
    parent = Gaussian(
        means=torch.tensor([[1.0, 2.0, 3.0]]),
        L_flat=torch.randn(1, 6),  # random covariance
        opacities=torch.tensor([[2.0]]),
        sh_coeffs=torch.randn(1, 3),
    )
    children = subdivide_to_8(parent)
    cov = children.covariance()
    eigvals = torch.linalg.eigvalsh(cov)
    assert (eigvals > 0).all()


def test_subdivide_opacity_halved_per_cut():
    parent = _make_parent()
    children = subdivide_to_8(parent)
    # 3 binary cuts: each halves opacity in probability space
    parent_prob = torch.sigmoid(parent.opacities).item()
    expected_prob = parent_prob / 8.0  # halved 3 times
    for i in range(children.num_gaussians):
        child_prob = torch.sigmoid(children.opacities[i:i+1]).item()
        assert abs(child_prob - expected_prob) < 0.01
