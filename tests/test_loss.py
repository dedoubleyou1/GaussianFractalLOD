import torch
from gaussianfractallod.loss import rendering_loss


def test_loss_zero_for_identical():
    img = torch.rand(64, 64, 3)
    loss = rendering_loss(img, img, ssim_weight=0.2)
    assert loss.item() < 1e-5


def test_loss_positive_for_different():
    pred = torch.rand(64, 64, 3)
    gt = torch.rand(64, 64, 3)
    loss = rendering_loss(pred, gt, ssim_weight=0.2)
    assert loss.item() > 0


def test_loss_is_differentiable():
    pred = torch.rand(64, 64, 3, requires_grad=True)
    gt = torch.rand(64, 64, 3)
    loss = rendering_loss(pred, gt, ssim_weight=0.2)
    loss.backward()
    assert pred.grad is not None
