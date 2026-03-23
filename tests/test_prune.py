import torch
from gaussianfractallod.split_tree import SplitTree
from gaussianfractallod.prune import prune_level


def test_prune_low_mass_children():
    tree = SplitTree(num_roots=2, sh_dim=3)
    tree.add_level()
    with torch.no_grad():
        # sigmoid(-5) ≈ 0.007 < 0.05 threshold → child A of root 0 pruned
        tree.levels[0].mass_logit[0] = -5.0
        tree.levels[0].mass_logit[1] = 0.0
    stats = prune_level(tree, level_idx=0, mass_threshold=0.05)
    occ = tree.get_occupancy(0)
    assert occ[0, 0] == False
    assert occ[0, 1] == True
    assert occ[1, 0] == True
    assert occ[1, 1] == True
    assert stats["mass"] == 1


def test_prune_renormalizes_mass_to_survivor():
    tree = SplitTree(num_roots=1, sh_dim=3)
    tree.add_level()
    with torch.no_grad():
        tree.levels[0].mass_logit[0] = -5.0
        tree.levels[0].color_split[0] = torch.tensor([0.5, 0.3, 0.1])
    prune_level(tree, level_idx=0, mass_threshold=0.05)
    occ = tree.get_occupancy(0)
    assert occ[0, 0] == False
    assert occ[0, 1] == True
    torch.testing.assert_close(tree.levels[0].color_split[0], torch.zeros(3))
    assert torch.sigmoid(tree.levels[0].mass_logit[0]).item() < 0.001


def test_prune_converged_splits():
    """Splits with near-zero magnitude should be pruned entirely."""
    tree = SplitTree(num_roots=2, sh_dim=3)
    tree.add_level()
    with torch.no_grad():
        # Node 0: split vars near zero → converged, should prune both children
        tree.levels[0].position_split[0] = torch.tensor([0.001, 0.0, 0.0])
        tree.levels[0].cov_split[0] = torch.zeros(6)
        tree.levels[0].color_split[0] = torch.zeros(3)
        # Node 1: meaningful split → should keep
        tree.levels[0].position_split[1] = torch.tensor([0.5, 0.3, 0.0])
    stats = prune_level(tree, level_idx=0, split_magnitude_threshold=0.01)
    occ = tree.get_occupancy(0)
    # Node 0: both children pruned (converged)
    assert occ[0, 0] == False
    assert occ[0, 1] == False
    assert stats["convergence"] == 1
    # Node 1: both children kept
    assert occ[1, 0] == True
    assert occ[1, 1] == True


def test_prune_low_opacity():
    """Children with negligible effective opacity should be pruned."""
    tree = SplitTree(num_roots=1, sh_dim=3)
    tree.add_level()
    with torch.no_grad():
        tree.levels[0].position_split[0] = torch.tensor([0.5, 0.0, 0.0])
        # Very asymmetric: pi_A ≈ 0.007
        tree.levels[0].mass_logit[0] = -5.0

    # Parent opacity is very low (sigmoid(-5) ≈ 0.007)
    parent_opacities = torch.tensor([-5.0])
    stats = prune_level(
        tree, level_idx=0,
        mass_threshold=0.001,  # don't trigger mass pruning
        opacity_threshold=0.005,
        parent_opacities=parent_opacities,
    )
    # Both children should have very low effective opacity
    assert stats["opacity"] >= 1
