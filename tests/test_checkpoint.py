import torch
from pathlib import Path
from gaussianfractallod.gaussian import Gaussian
from gaussianfractallod.split_tree import GaussianTree
from gaussianfractallod.checkpoint import save_checkpoint, load_checkpoint


def test_checkpoint_round_trip(tmp_path):
    roots = Gaussian(
        means=torch.randn(2, 3),
        quats=torch.randn(2, 4),
        log_scales=torch.randn(2, 3),
        opacities=torch.randn(2, 1),
        sh_dc=torch.randn(2, 1, 3),
        sh_rest=torch.zeros(2, 0, 3),
    )
    tree = GaussianTree()
    tree.set_root_level(roots)
    tree.add_level()

    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, roots, tree, phase=2, level=1)

    loaded_roots, loaded_tree, meta = load_checkpoint(path)
    torch.testing.assert_close(loaded_roots.means, roots.means)
    torch.testing.assert_close(loaded_roots.quats, roots.quats)
    torch.testing.assert_close(loaded_roots.log_scales, roots.log_scales)
    torch.testing.assert_close(loaded_roots.sh_dc, roots.sh_dc)
    assert loaded_tree.depth == tree.depth
    assert meta["phase"] == 2
    assert meta["level"] == 1
