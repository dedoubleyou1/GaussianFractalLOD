import torch
from gaussianfractallod.split_tree import SplitTree
from gaussianfractallod.prune import prune_level


def test_prune_low_mass_children():
    tree = SplitTree(num_roots=2, sh_dim=3)
    tree.add_level()
    with torch.no_grad():
        tree.levels[0].mass_logit[0] = -5.0
        tree.levels[0].mass_logit[1] = 0.0
    pruned_count = prune_level(tree, level_idx=0, threshold=0.01)
    occ = tree.get_occupancy(0)
    assert occ[0, 0] == False
    assert occ[0, 1] == True
    assert occ[1, 0] == True
    assert occ[1, 1] == True
    assert pruned_count == 1


def test_prune_renormalizes_mass_to_survivor():
    tree = SplitTree(num_roots=1, sh_dim=3)
    tree.add_level()
    with torch.no_grad():
        tree.levels[0].mass_logit[0] = -5.0
        tree.levels[0].color_split[0] = torch.tensor([0.5, 0.3, 0.1])
    prune_level(tree, level_idx=0, threshold=0.01)
    occ = tree.get_occupancy(0)
    assert occ[0, 0] == False
    assert occ[0, 1] == True
    torch.testing.assert_close(tree.levels[0].color_split[0], torch.zeros(3))
    assert torch.sigmoid(tree.levels[0].mass_logit[0]).item() < 0.001
