import torch
from gaussianfractallod.split_tree import SplitTree
from gaussianfractallod.prune import prune_level


def test_prune_low_mass_children():
    tree = SplitTree(num_roots=2, sh_dim=3)
    tree.add_level()
    with torch.no_grad():
        # Large positive offset → π_left ≈ 1, π_right ≈ 0 → right child pruned
        tree.levels[0].cut_offset[0] = 5.0
        # Node 1: meaningful offset so it doesn't get convergence-pruned
        tree.levels[0].cut_offset[1] = 0.5
    stats = prune_level(tree, level_idx=0, mass_threshold=0.05)
    occ = tree.get_occupancy(0)
    assert occ[0, 0] == False   # right child of node 0 pruned (mass)
    assert occ[0, 1] == True    # left child survives
    assert occ[1, 0] == True    # node 1 both survive
    assert occ[1, 1] == True
    assert stats["mass"] == 1


def test_prune_renormalizes_mass_to_survivor():
    tree = SplitTree(num_roots=1, sh_dim=3)
    tree.add_level()
    with torch.no_grad():
        tree.levels[0].cut_offset[0] = 5.0  # right child ≈ 0 mass
        tree.levels[0].color_split[0] = torch.tensor([0.5, 0.3, 0.1])
    prune_level(tree, level_idx=0, mass_threshold=0.05)
    occ = tree.get_occupancy(0)
    assert occ[0, 0] == False  # right pruned
    assert occ[0, 1] == True   # left survives
    # Dead split vars should be zeroed
    torch.testing.assert_close(tree.levels[0].color_split[0], torch.zeros(3))
    # Offset should push all mass to left (surviving) child
    assert tree.levels[0].cut_offset[0].item() == 10.0


def test_prune_converged_splits():
    tree = SplitTree(num_roots=2, sh_dim=3)
    tree.add_level()
    with torch.no_grad():
        # Node 0: near-zero offset + color → converged
        tree.levels[0].cut_offset[0] = 0.001
        tree.levels[0].color_split[0] = torch.zeros(3)
        # Node 1: meaningful split
        tree.levels[0].cut_offset[1] = 1.0
    stats = prune_level(tree, level_idx=0, split_magnitude_threshold=0.01)
    occ = tree.get_occupancy(0)
    assert occ[0, 0] == False  # both pruned (converged)
    assert occ[0, 1] == False
    assert stats["convergence"] == 1
    assert occ[1, 0] == True   # kept
    assert occ[1, 1] == True
