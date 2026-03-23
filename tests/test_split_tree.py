import torch
from gaussianfractallod.split_tree import SplitTree


def test_create_empty_tree():
    tree = SplitTree(num_roots=4, sh_dim=3)
    assert tree.num_roots == 4
    assert tree.depth == 0
    assert tree.num_splits == 0


def test_add_level():
    tree = SplitTree(num_roots=2, sh_dim=3)
    tree.add_level()
    assert tree.depth == 1
    assert tree.num_splits == 2


def test_add_two_levels():
    tree = SplitTree(num_roots=1, sh_dim=3)
    tree.add_level()
    tree.add_level()
    assert tree.depth == 2
    assert tree.num_splits == 3


def test_split_vars_are_parameters():
    tree = SplitTree(num_roots=1, sh_dim=3)
    tree.add_level()
    params = list(tree.level_parameters(0))
    assert len(params) == 3  # cut_direction, cut_offset, color_split
    for p in params:
        assert p.requires_grad


def test_split_vars_initialized_correctly():
    tree = SplitTree(num_roots=2, sh_dim=3)
    tree.add_level()
    sv = tree.get_level_split_vars(0)
    # Cut offset should be 0 (symmetric split)
    torch.testing.assert_close(sv.cut_offset, torch.zeros(2))
    # Color split should be 0
    torch.testing.assert_close(sv.color_split, torch.zeros(2, 3))


def test_occupancy_mask():
    tree = SplitTree(num_roots=2, sh_dim=3)
    tree.add_level()
    assert tree.get_occupancy(0).all()
    tree.set_occupancy(level=0, node_idx=0, child_b=False)
    occ = tree.get_occupancy(0)
    assert occ[0, 0] == True
    assert occ[0, 1] == False
    assert occ[1, 0] == True
    assert occ[1, 1] == True
