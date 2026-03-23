import torch
from gaussianfractallod.gaussian import Gaussian
from gaussianfractallod.derive import derive_children, SplitVariables, _Phi


def test_gaussian_construction():
    g = Gaussian(
        means=torch.zeros(1, 3),
        L_flat=torch.zeros(1, 6),
        opacities=torch.ones(1, 1),
        sh_coeffs=torch.zeros(1, 3),
    )
    assert g.means.shape == (1, 3)
    assert g.L_flat.shape == (1, 6)


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
        cut_direction=torch.tensor([[0.0, 0.0, 1.0]]),  # cut along z
        cut_offset=torch.tensor([0.0]),  # symmetric
        color_split=torch.zeros(1, sh_dim),
    )


def test_opacity_conservation():
    parent = _make_parent()
    sv = _make_split_vars()
    child_a, child_b = derive_children(parent, sv)
    total_opacity = child_a.opacities + child_b.opacities
    torch.testing.assert_close(total_opacity, parent.opacities, atol=1e-5, rtol=1e-5)


def test_center_of_mass_conservation():
    parent = _make_parent()
    sv = _make_split_vars()
    child_a, child_b = derive_children(parent, sv)
    pi_right = 1.0 - _Phi(sv.cut_offset)
    pi_left = _Phi(sv.cut_offset)
    weighted_mean = pi_right * child_a.means + pi_left * child_b.means
    torch.testing.assert_close(weighted_mean, parent.means, atol=1e-4, rtol=1e-4)


def test_color_conservation():
    parent = _make_parent()
    sv = SplitVariables(
        cut_direction=torch.tensor([[1.0, 1.0, 0.0]]),
        cut_offset=torch.tensor([0.3]),  # asymmetric
        color_split=torch.randn(1, 3) * 0.1,
    )
    child_a, child_b = derive_children(parent, sv)
    pi_right = 1.0 - _Phi(sv.cut_offset)
    pi_left = _Phi(sv.cut_offset)
    weighted_color = pi_right * child_a.sh_coeffs + pi_left * child_b.sh_coeffs
    torch.testing.assert_close(weighted_color, parent.sh_coeffs, atol=1e-4, rtol=1e-4)


def test_covariance_conservation():
    """Total covariance (law of total variance) should be conserved."""
    parent = _make_parent()
    sv = SplitVariables(
        cut_direction=torch.tensor([[1.0, 0.5, -0.3]]),
        cut_offset=torch.tensor([0.2]),
        color_split=torch.zeros(1, 3),
    )
    child_a, child_b = derive_children(parent, sv)
    pi_right = (1.0 - _Phi(sv.cut_offset)).unsqueeze(-1).unsqueeze(-1)
    pi_left = _Phi(sv.cut_offset).unsqueeze(-1).unsqueeze(-1)

    cov_a = child_a.covariance()
    cov_b = child_b.covariance()
    cov_p = parent.covariance()

    delta_a = (child_a.means - parent.means).unsqueeze(-1)
    delta_b = (child_b.means - parent.means).unsqueeze(-1)
    scatter_a = delta_a @ delta_a.transpose(-1, -2)
    scatter_b = delta_b @ delta_b.transpose(-1, -2)

    reconstructed = pi_right * (cov_a + scatter_a) + pi_left * (cov_b + scatter_b)
    torch.testing.assert_close(reconstructed, cov_p, atol=1e-3, rtol=1e-3)


def test_derivation_is_differentiable():
    parent = _make_parent()
    sv = SplitVariables(
        cut_direction=torch.tensor([[0.0, 0.0, 1.0]], requires_grad=True),
        cut_offset=torch.tensor([0.0], requires_grad=True),
        color_split=torch.zeros(1, 3, requires_grad=True),
    )
    child_a, child_b = derive_children(parent, sv)
    loss = child_a.means.sum() + child_b.means.sum()
    loss.backward()
    assert sv.cut_direction.grad is not None
    assert sv.cut_offset.grad is not None


def test_children_covariance_positive_definite():
    parent = _make_parent()
    sv = SplitVariables(
        cut_direction=torch.tensor([[0.7, -0.3, 0.5]]),
        cut_offset=torch.tensor([0.5]),
        color_split=torch.zeros(1, 3),
    )
    child_a, child_b = derive_children(parent, sv)
    for child in [child_a, child_b]:
        cov = child.covariance()
        eigvals = torch.linalg.eigvalsh(cov)
        assert (eigvals > 0).all(), f"Non-positive eigenvalues: {eigvals}"


def test_symmetric_split_gives_equal_children():
    """Cut at d=0 should give symmetric children."""
    parent = _make_parent()
    sv = SplitVariables(
        cut_direction=torch.tensor([[0.0, 0.0, 1.0]]),
        cut_offset=torch.tensor([0.0]),
        color_split=torch.zeros(1, 3),
    )
    child_a, child_b = derive_children(parent, sv)
    # Symmetric split: both children should have same opacity
    torch.testing.assert_close(child_a.opacities, child_b.opacities, atol=1e-5, rtol=1e-5)
