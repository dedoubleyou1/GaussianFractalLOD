import torch
from gaussianfractallod.gaussian import Gaussian
from gaussianfractallod.split_tree import SplitTree
from gaussianfractallod.reconstruct import reconstruct


def _make_roots(n=2, sh_dim=3):
    return Gaussian(
        means=torch.randn(n, 3),
        L_flat=torch.zeros(n, 6),
        opacities=torch.ones(n, 1) * 0.8,
        sh_coeffs=torch.randn(n, sh_dim),
    )


def test_reconstruct_depth_zero_returns_roots():
    roots = _make_roots(2)
    tree = SplitTree(num_roots=2, sh_dim=3)
    result = reconstruct(roots, tree, target_depth=0)
    assert result.num_gaussians == 2
    torch.testing.assert_close(result.means, roots.means)


def test_reconstruct_depth_one_doubles_count():
    roots = _make_roots(2, sh_dim=3)
    tree = SplitTree(num_roots=2, sh_dim=3)
    tree.add_level()
    result = reconstruct(roots, tree, target_depth=1)
    assert result.num_gaussians == 4


def test_reconstruct_depth_two():
    roots = _make_roots(1, sh_dim=3)
    tree = SplitTree(num_roots=1, sh_dim=3)
    tree.add_level()
    tree.add_level()
    result = reconstruct(roots, tree, target_depth=2)
    assert result.num_gaussians == 4


def test_reconstruct_respects_occupancy():
    roots = _make_roots(1, sh_dim=3)
    tree = SplitTree(num_roots=1, sh_dim=3)
    tree.add_level()
    tree.set_occupancy(level=0, node_idx=0, child_b=False)
    result = reconstruct(roots, tree, target_depth=1)
    assert result.num_gaussians == 1


def test_reconstruct_deeper_than_tree_returns_leaves():
    roots = _make_roots(1, sh_dim=3)
    tree = SplitTree(num_roots=1, sh_dim=3)
    tree.add_level()
    result = reconstruct(roots, tree, target_depth=5)
    assert result.num_gaussians == 2
