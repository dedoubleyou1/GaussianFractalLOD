import torch
from pathlib import Path
from gaussianfractallod.gaussian import Gaussian
from gaussianfractallod.split_tree import SplitTree
from gaussianfractallod.checkpoint import save_checkpoint, load_checkpoint


def test_checkpoint_round_trip(tmp_path):
    roots = Gaussian(
        means=torch.randn(4, 3),
        scales=torch.randn(4, 3),
        opacities=torch.randn(4, 1),
        sh_coeffs=torch.randn(4, 3),
    )
    tree = SplitTree(num_roots=4, sh_dim=3)
    tree.add_level()

    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, roots, tree, phase=1, level=0)

    loaded_roots, loaded_tree, meta = load_checkpoint(path)
    torch.testing.assert_close(loaded_roots.means, roots.means)
    torch.testing.assert_close(loaded_roots.scales, roots.scales)
    assert loaded_tree.depth == tree.depth
    assert loaded_tree.num_roots == tree.num_roots
    assert meta["phase"] == 1
    assert meta["level"] == 0


def test_checkpoint_preserves_split_vars(tmp_path):
    roots = Gaussian(
        means=torch.randn(2, 3),
        scales=torch.randn(2, 3),
        opacities=torch.randn(2, 1),
        sh_coeffs=torch.randn(2, 3),
    )
    tree = SplitTree(num_roots=2, sh_dim=3)
    tree.add_level()
    with torch.no_grad():
        tree.levels[0].mass_logit.fill_(0.7)
        tree.levels[0].color_split.fill_(0.1)

    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, roots, tree, phase=2, level=0)

    _, loaded_tree, _ = load_checkpoint(path)
    torch.testing.assert_close(
        loaded_tree.levels[0].mass_logit,
        torch.tensor([0.7, 0.7]),
    )
