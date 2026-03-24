import torch
from pathlib import Path
from gaussianfractallod.gaussian import Gaussian
from gaussianfractallod.split_tree import GaussianTree
from gaussianfractallod.checkpoint import save_checkpoint, load_checkpoint


def test_checkpoint_round_trip(tmp_path):
    roots = Gaussian(
        means=torch.randn(2, 3),
        L_flat=torch.randn(2, 6),
        opacities=torch.randn(2, 1),
        sh_coeffs=torch.randn(2, 3),
    )
    tree = GaussianTree()
    tree.set_root_level(roots)
    tree.add_level()

    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, roots, tree, phase=2, level=1)

    loaded_roots, loaded_tree, meta = load_checkpoint(path)
    torch.testing.assert_close(loaded_roots.means, roots.means)
    torch.testing.assert_close(loaded_roots.L_flat, roots.L_flat)
    assert loaded_tree.depth == tree.depth
    assert meta["phase"] == 2
    assert meta["level"] == 1
