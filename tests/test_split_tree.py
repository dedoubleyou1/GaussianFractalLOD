import torch
from gaussianfractallod.gaussian import Gaussian
from gaussianfractallod.split_tree import GaussianTree


def _make_roots(n=1):
    return Gaussian(
        means=torch.randn(n, 3),
        L_flat=torch.zeros(n, 6),
        opacities=torch.ones(n, 1) * 2.0,
        sh_coeffs=torch.randn(n, 3),
    )


def test_create_tree():
    tree = GaussianTree()
    assert tree.depth == 0


def test_set_root_level():
    tree = GaussianTree()
    roots = _make_roots(1)
    tree.set_root_level(roots)
    assert tree.depth == 1
    assert tree.levels[0].num_gaussians == 1


def test_add_level():
    tree = GaussianTree()
    tree.set_root_level(_make_roots(1))
    tree.add_level()
    assert tree.depth == 2
    assert tree.levels[1].num_gaussians == 8  # 1 × 8


def test_add_two_levels():
    tree = GaussianTree()
    tree.set_root_level(_make_roots(1))
    tree.add_level()
    tree.add_level()
    assert tree.depth == 3
    assert tree.levels[2].num_gaussians == 64  # 8 × 8


def test_root_level_frozen():
    tree = GaussianTree()
    tree.set_root_level(_make_roots(1))
    for p in tree.levels[0].parameters():
        assert not p.requires_grad


def test_child_level_trainable():
    tree = GaussianTree()
    tree.set_root_level(_make_roots(1))
    tree.add_level()
    for p in tree.levels[1].parameters():
        assert p.requires_grad


def test_get_gaussians_at_depth():
    tree = GaussianTree()
    tree.set_root_level(_make_roots(2))
    tree.add_level()
    g0 = tree.get_gaussians_at_depth(0)
    g1 = tree.get_gaussians_at_depth(1)
    assert g0.num_gaussians == 2
    assert g1.num_gaussians == 16  # 2 × 8
