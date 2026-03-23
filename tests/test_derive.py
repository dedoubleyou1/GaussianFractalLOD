import torch
from gaussianfractallod.gaussian import Gaussian
from gaussianfractallod.derive import derive_children, SplitVariables


def test_gaussian_construction():
    g = Gaussian(
        means=torch.zeros(1, 3),
        L_flat=torch.zeros(1, 6),
        opacities=torch.ones(1, 1),
        sh_coeffs=torch.zeros(1, 3),
    )
    assert g.means.shape == (1, 3)
    assert g.L_flat.shape == (1, 6)
    assert g.opacities.shape == (1, 1)
    assert g.sh_coeffs.shape == (1, 3)


def test_gaussian_num_gaussians():
    g = Gaussian(
        means=torch.zeros(5, 3),
        L_flat=torch.zeros(5, 6),
        opacities=torch.ones(5, 1),
        sh_coeffs=torch.zeros(5, 3),
    )
    assert g.num_gaussians == 5


def test_L_matrix_shape():
    g = Gaussian(
        means=torch.zeros(3, 3),
        L_flat=torch.zeros(3, 6),
        opacities=torch.ones(3, 1),
        sh_coeffs=torch.zeros(3, 3),
    )
    L = g.L_matrix()
    assert L.shape == (3, 3, 3)
    # Diagonal should be exp(0) = 1
    assert torch.allclose(L[:, 0, 0], torch.ones(3))


def test_covariance_positive_definite():
    g = Gaussian(
        means=torch.zeros(2, 3),
        L_flat=torch.randn(2, 6),
        opacities=torch.ones(2, 1),
        sh_coeffs=torch.zeros(2, 3),
    )
    cov = g.covariance()
    torch.testing.assert_close(cov, cov.transpose(-1, -2))
    eigvals = torch.linalg.eigvalsh(cov)
    assert (eigvals > 0).all()


def _make_parent(sh_dim=3):
    L_flat = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    return Gaussian(
        means=torch.tensor([[1.0, 2.0, 3.0]]),
        L_flat=L_flat,
        opacities=torch.tensor([[0.8]]),
        sh_coeffs=torch.randn(1, sh_dim),
    )


def _make_split_vars(sh_dim=3):
    return SplitVariables(
        mass_logit=torch.tensor([0.0]),
        position_split=torch.tensor([[0.1, 0.0, 0.0]]),
        variance_split=torch.zeros(1, 3),
        color_split=torch.zeros(1, sh_dim),
    )


def test_opacity_conservation():
    parent = _make_parent()
    sv = _make_split_vars()
    child_a, child_b = derive_children(parent, sv)
    total_opacity = child_a.opacities + child_b.opacities
    torch.testing.assert_close(total_opacity, parent.opacities, atol=1e-6, rtol=1e-6)


def test_center_of_mass_conservation():
    parent = _make_parent()
    sv = _make_split_vars()
    child_a, child_b = derive_children(parent, sv)
    pi_a = torch.sigmoid(sv.mass_logit)
    pi_b = 1.0 - pi_a
    weighted_mean = pi_a * child_a.means + pi_b * child_b.means
    torch.testing.assert_close(weighted_mean, parent.means, atol=1e-5, rtol=1e-5)


def test_color_conservation():
    parent = _make_parent()
    sv = SplitVariables(
        mass_logit=torch.tensor([0.5]),
        position_split=torch.tensor([[0.1, 0.2, -0.1]]),
        variance_split=torch.zeros(1, 3),
        color_split=torch.randn(1, 3) * 0.1,
    )
    child_a, child_b = derive_children(parent, sv)
    pi_a = torch.sigmoid(sv.mass_logit)
    pi_b = 1.0 - pi_a
    weighted_color = pi_a * child_a.sh_coeffs + pi_b * child_b.sh_coeffs
    torch.testing.assert_close(weighted_color, parent.sh_coeffs, atol=1e-5, rtol=1e-5)


def test_derivation_is_differentiable():
    parent = _make_parent()
    sv = SplitVariables(
        mass_logit=torch.tensor([0.0], requires_grad=True),
        position_split=torch.tensor([[0.1, 0.0, 0.0]], requires_grad=True),
        variance_split=torch.zeros(1, 3, requires_grad=True),
        color_split=torch.zeros(1, 3, requires_grad=True),
    )
    child_a, child_b = derive_children(parent, sv)
    loss = child_a.means.sum() + child_b.means.sum()
    loss.backward()
    assert sv.mass_logit.grad is not None
    assert sv.position_split.grad is not None


def test_children_covariance_positive_definite():
    parent = _make_parent()
    sv = SplitVariables(
        mass_logit=torch.tensor([0.3]),
        position_split=torch.tensor([[0.2, 0.1, -0.1]]),
        variance_split=torch.randn(1, 3) * 0.1,
        color_split=torch.zeros(1, 3),
    )
    child_a, child_b = derive_children(parent, sv)
    for child in [child_a, child_b]:
        cov = child.covariance()
        eigvals = torch.linalg.eigvalsh(cov)
        assert (eigvals > 0).all(), f"Non-positive eigenvalues: {eigvals}"
