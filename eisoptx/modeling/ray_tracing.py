"""
Classes and functions that make use of ray tracing.

In general, the tensors are arranged as follows:
    Dim 0: Number of field angles
    Dim 1: Number of pupil intersections
    Dim 2: Number of wavelengths
    Dim 3: Number of samples (optical systems)
"""

import torch


def get_z_mask(r: torch.Tensor):
    """Return mask whose values are [0, 0, 1] for dimensions [x, y, z].

    The shape of the mask is [3, *] with as many dimensions as "r".

    Args:
        r: Ray position vectors (shape: [3, *]).
    """
    z_mask = torch.cat((r.new_zeros(2), r.new_ones(1)), dim=0).view(
        -1, *(1,) * (r.dim() - 1)
    )
    return z_mask


def update_ray_coordinates(r: torch.Tensor, d: torch.Tensor, distance: torch.Tensor):
    """Return updated ray position vectors from the direction vectors and ray-marching distance.

    Args:
        r: Ray position vectors (shape: [3, *]).
        d: Ray direction vectors in normalized form (shape: [3, *]).
        distance: Ray-marching distance (shape: [*]).
    """
    delta_r = d * distance
    delta_z = delta_r[2]
    r = r + delta_r
    return r, delta_z


def find_marching_distance_spherical(
    r: torch.Tensor, d: torch.Tensor, c: torch.Tensor, eps: float = 1e-6
):
    """Return ray-marching distance required to reach the spherical surface, and intermediate values.

    Args:
        r: Ray position vectors (shape: [3, *, n_lens]).
        d: Ray direction vectors in normalized form (shape: [3, *, n_lens]).
        c: Surface curvatures (shape: [n_lens]).
        eps: Small number to prevent division by zero.
    """
    z, cz = r[2], d[2]
    e = -(r * d).sum(dim=0)
    mz = z + e * cz
    m2 = (r**2).sum(dim=0) - e**2
    temp = c * m2 - 2 * mz
    cos2_theta = cz**2 - c * temp

    # Check for missed rays
    # Allow cos(theta)^2 to be above 1 due to numerical errors, but not below "eps"
    failures = cos2_theta - eps < 0

    cos_theta = (cos2_theta.where(~failures, 1.0)).sqrt()
    dist = e + temp / (cz + cos_theta)

    return dist, failures, cos_theta, cos2_theta


@torch.no_grad()
def find_marching_distance_boundaries(
    r: torch.Tensor,
    d: torch.Tensor,
    c: torch.Tensor,
    a: torch.Tensor | None,
    eps: float = 1e-6,
):
    """Return minimum and maximum ray-marching distance to prevent a surface profile from being undefined.

    Args:
        r: Ray position vectors (shape: [3, *, n_lens]).
        d: Ray direction vectors in normalized form (shape: [3, *, n_lens]).
        c: Surface curvatures (shape: [n_lens]).
        a: Additional aspherical coefficients including conic constant (shape: [n_lens, n_coefficients]).
        eps: Small number to prevent division by zero.
    """
    # Find rho_max
    k = 0
    if a is not None:
        k = a[..., 0]
    alpha = (1 + k) * c**2
    rho_max = 1 / alpha.clip(min=eps)
    # Define the quadratic equation
    a = 1 - d[2] ** 2
    b = 2 * (r * d)[:2].sum(dim=0)
    c = (r[:2] ** 2).sum(dim=0) - rho_max
    # Find the roots
    sqrt_term = (b**2 - 4 * a * c).clip(min=eps).sqrt()
    denom = (2 * a).clip(min=eps)
    t_min = (-b - sqrt_term) / denom
    t_max = (-b + sqrt_term) / denom
    return t_min, t_max


def find_marching_distance_aspherical(
    r: torch.Tensor,
    d: torch.Tensor,
    c: torch.Tensor,
    a: torch.Tensor | None,
    max_iter: int = 16,
    tol: float = 1e-8,
    failure_tol: float = 1e-6,
    eps: float = 1e-6,
):
    """Return distance from the rays to the aspherical surface.

    Args:
        r: Ray position vectors (shape: [3, *, n_lens]).
        d: Ray direction vectors in normalized form (shape: [3, *, n_lens]).
        c: Surface curvatures (shape: [n_lens]).
        a: Additional aspherical coefficients including conic constant (shape: [n_lens, n_coefficients]).
        max_iter: Maximum number of iterations.
        tol: Tolerance criterion between two iterations.
        failure_tol: Tolerance criterion for ray failures (should be larger than tol).
        eps: Small number to prevent division by zero.
    """
    # Find region of validity for the ray-marching distance (where the surface is defined)
    t_min, t_max = find_marching_distance_boundaries(
        *(tensor.detach() for tensor in (r, d, c, a)), eps=eps
    )
    t_min = t_min + eps
    t_max = t_max - eps

    # Initial guess for the ray-marching distance
    dist = approximate_marching_distance(
        *(tensor.detach() for tensor in (t_min, t_max, r, d, c, a)), eps=eps
    )

    # Refine the initial guess
    dist = refine_marching_distance(
        *(tensor.detach() for tensor in (dist, t_min, t_max, r, d, c, a)),
        max_iter=max_iter,
        tol=tol,
    )

    # Final iteration with autograd
    dist, is_inside, delta_z, _ = update_marching_distance(dist, r, d, c, a)
    dist = dist.clip(min=t_min, max=t_max)

    # Failures if the ray is outside the validity region or the error in z is larger than tol
    failures = ~is_inside | (delta_z.abs() > failure_tol)

    # For failures, output the ray-marching distance to a transversal plane at the vertex instead
    dist = dist.where(~failures, -r[2] / d[2].clip(min=eps))

    return dist, failures


@torch.no_grad()
def approximate_marching_distance(
    t_min: torch.Tensor,
    t_max: torch.Tensor,
    r: torch.Tensor,
    d: torch.Tensor,
    c: torch.Tensor,
    a: torch.Tensor | None,
    n_points: int = 5,
    rel_range: float = 0.5,
    eps: float = 1e-6,
):
    """Return initial guess for the ray-marching distance for aspherical surfaces.

    t_min: Minimum ray-marching distance (shape: [*, n_lens]).
    t_max: Maximum ray-marching distance (shape: [*, n_lens]).
    r: Ray position vectors (shape: [3, *, n_lens]).
    d: Ray direction vectors in normalized form (shape: [3, *, n_lens]).
    c: Surface curvatures (shape: [n_lens]).
    a: Additional aspherical coefficients including conic constant (shape: [n_lens, n_coefficients]).
    n_points: Number of points to sample around the initial guess.
    rel_range: Range of distances around the initial guess proportional to the ray height.
    eps: Small number to prevent division by zero.
    """
    # Set the initial guess for the ray-marching distance at the horizontal plane at the vertex
    dist = -r[2] / d[2].clip(min=eps)

    # Extend the initial guess to a range of distances around the initial guess proportional to the ray height
    # As rays get farther from the optical axis, larger displacements from the vertex in z are expected
    h = ((r + dist * d)[:2] ** 2).sum(dim=0).sqrt()
    dist_range = (
        rel_range
        * h
        * torch.linspace(-1, 1, n_points).to(h).view((-1,) + (1,) * h.ndim)
    )
    dist = dist + dist_range
    dist = dist.clip(min=t_min, max=t_max)

    # Apply one ray-marching refinement step to all initial guesses
    updated_dist, _, delta_z, *_ = update_marching_distance(
        dist, r[:, None], d[:, None], c, a
    )
    dist = updated_dist.clip(min=t_min, max=t_max)

    # Favor initial guess if the distance is closer to zero (in case there are multiple intersections)
    merit = 1 / dist.abs().clip(min=eps)

    # Remove initial guess from consideration if there is no improvement
    r1 = r[:, None, ...] + dist * d[:, None, ...]
    rho = (r1[:2] ** 2).sum(dim=0)
    z_surface, is_inside = evaluate_aspherical_profile(rho, c, a)
    updated_delta_z = z_surface - r1[2]
    delta_z_improvement = delta_z.abs() - updated_delta_z.abs()
    merit = merit * delta_z_improvement.sign()

    # Select best initial guess
    min_idx = merit.argmax(dim=0)
    dist = dist.gather(0, min_idx[None, ...]).squeeze(dim=0)
    return dist


@torch.no_grad()
def refine_marching_distance(
    dist: torch.Tensor,
    t_min: torch.Tensor,
    t_max: torch.Tensor,
    r: torch.Tensor,
    d: torch.Tensor,
    c: torch.Tensor,
    a: torch.Tensor | None,
    max_iter: int = 16,
    tol: float = 1e-10,
):
    """Return iteratively refined initial guess for ray-marching distance.

    dist: Initial guess for ray-marching distance (shape: [*, n_lens]).
    t_min: Minimum ray-marching distance (shape: [*, n_lens]).
    t_max: Maximum ray-marching distance (shape: [*, n_lens]).
    r: Ray position vectors (shape: [3, *, n_lens]).
    d: Ray direction vectors in normalized form (shape: [3, *, n_lens]).
    c: Surface curvatures (shape: [n_lens]).
    a: Additional aspherical coefficients including conic constant (shape: [n_lens, n_coefficients]).
    max_iter: Maximum number of iterations.
    tol: Tolerance criterion between two iterations.
    """
    # Refine the selected initial guess
    for i in range(max_iter):
        updated_dist, _, delta_z, *_ = update_marching_distance(dist, r, d, c, a)
        updated_dist = updated_dist.clip(min=t_min, max=t_max)
        max_difference = (updated_dist - dist).abs().max().item()
        if max_difference < tol:
            break
        dist = updated_dist
    return dist


def update_marching_distance(
    dist: torch.Tensor,
    r0: torch.Tensor,
    d: torch.Tensor,
    c: torch.Tensor,
    a: torch.Tensor | None,
    eps: float = 1e-6,
):
    """Return updated ray-marching distance, intermediate values, and gradients.

    Args:
        dist: Initial guess for ray-marching distance (shape: [*, n_lens]).
        r0: Initial ray position vectors (shape: [3, *, n_lens]).
        d: Ray direction vectors in normalized form (shape: [3, *, n_lens]).
        c: Curvature of aspherical surfaces (shape: [n_lens]).
        a: Additional aspherical coefficients including conic constant (shape: [n_lens, n_coefficients]).
        eps: Small number to avoid division by zero.
    """
    r1 = r0 + dist * d
    rho = (r1[:2] ** 2).sum(dim=0)
    z_surface, is_inside = evaluate_aspherical_profile(rho, c, a)

    derivative, *_ = evaluate_aspherical_profile(rho, c, a, compute_derivative=True)
    g = torch.cat((2 * r1[:2] * derivative, -torch.ones_like(r1[0:1])), dim=0)

    delta_z = z_surface - r1[2]
    dist = dist - delta_z / (g * d).sum(dim=0).clip(max=-eps)
    return dist, is_inside, delta_z, g


def evaluate_aspherical_profile(
    rho: torch.Tensor,
    c: torch.Tensor,
    a: torch.Tensor | None,
    compute_derivative: bool = False,
    eps: float = 1e-6,
):
    """Return profile or derivative of an aspherical surface at a given lateral position.

    The aspherical profile is given by the equation:
        z = c * r^2 / (1 + sqrt(1 - alpha * r^2)) + sum(p_i * r^(2(i + 2)))

    The derivative of the aspherical profile is given by:
        dz/dr = c / (2 * sqrt(1 - alpha * r^2)) + sum((i + 2) * p_i * r^(2(i + 1)))

    Args:
        rho: Squared transverse ray coordinates (shape: [*, n_lens]).
        c: Curvature of aspherical surfaces (shape: [n_lens]).
        a: Additional aspherical coefficients including conic constant (shape: [n_lens, n_coefficients]).
        compute_derivative: If True, compute the derivative; otherwise, compute the profile.
        eps: Small number to prevent numerical errors.
    """
    k = 0
    p = None
    if a is not None:
        k = a[..., 0]
        if a.shape[-1] > 1:
            p = a[..., 1:]

    alpha = (1 + k) * c**2
    rho_max = 1 / alpha.clip(min=eps)
    is_inside = rho < rho_max - eps

    sqrt_term = ((1 - alpha * rho).clip(min=eps)).sqrt()

    if compute_derivative:
        output = c / (2 * sqrt_term)
    else:
        output = rho * c / (1 + sqrt_term)

    if p is not None:
        exp = torch.arange(p.shape[-1]).to(rho)  # [0, 1, 2, ...]
        expanded_rho = rho[..., None].expand(*rho.shape, len(exp))
        # Sum over the aspherical coefficients using dot product for efficiency
        # (Equivalent to (exp * p * rho_raised).sum(dim=-1))
        if compute_derivative:
            aspherical_term = torch.einsum(
                "...p,...p->...", (exp + 2) * p, expanded_rho ** (exp + 1)
            )
        else:
            aspherical_term = torch.einsum(
                "...p,...p->...", p, expanded_rho ** (exp + 2)
            )

        output = output + aspherical_term

    # For ray failures, pretend that the interface is flat
    output = output.where(is_inside, 0.0)
    return output, is_inside


def apply_snell_spherical(
    r: torch.Tensor,
    d: torch.Tensor,
    c: torch.Tensor,
    mu: torch.Tensor,
    cos_theta: torch.Tensor | None = None,
    eps: float = 1e-6,
):
    """Return updated direction vectors after refraction at spherical interface, and intermediate results.

    Args:
        r: Ray position vectors (shape: [3, *, n_wavelengths, n_lens]).
        d: Ray direction vectors in normalized form (shape: [3, *, n_wavelengths, n_lens]).
        c: Surface curvatures (shape: [n_lens]).
        mu: Refractive index ratios (n1/n2) (shape: [n_wavelengths, n_lens]).
        cos_theta: From angles between rays and surface normals (precomputed) (shape: [*, n_wavelengths, n_lens]).
        eps: Small number to prevent division by zero.
    """
    if cos_theta is None:
        cos_theta = find_marching_distance_spherical(r, d, c)[2]

    cos2_prime = 1 - mu**2 * (1 - cos_theta**2)

    # Check for total internal reflexion
    # Allow cos(theta')^2 to be above 1 due to numerical errors, but not below "eps"
    tir = cos2_prime - eps < 0

    cos_prime = (cos2_prime.where(~tir, 1.0)).sqrt()
    g = cos_prime - mu * cos_theta
    d = mu * d - g * (c * r - get_z_mask(d))

    # Check for ray directions going backward
    backward = d[2] - eps < 0

    return d, tir, backward, cos2_prime


def apply_snell_aspherical(
    r: torch.Tensor,
    d: torch.Tensor,
    c: torch.Tensor,
    a: torch.Tensor | None,
    mu: torch.Tensor,
    eps: float = 1e-6,
):
    """Return updated direction vectors after refraction at aspherical interface, and intermediate results.

    Args:
        r: Ray position vectors (shape: [3, *, n_wavelengths, n_lens]).
        d: Ray direction vectors in normalized form (shape: [3, *, n_wavelengths, n_lens]).
        c: Surface curvatures (shape: [n_lens]).
        a: Additional aspherical coefficients including conic constant (shape: [n_lens, n_coefficients]).
        mu: Refractive index ratios (n1/n2) (shape: [n_wavelengths, n_lens]).
        eps: Small number to prevent division by zero.
    """
    derivative, *_ = evaluate_aspherical_profile(
        (r[:2] ** 2).sum(dim=0), c, a, compute_derivative=True
    )
    n = -torch.cat((2 * r[:2] * derivative, -torch.ones_like(r[0:1])), dim=0)
    n = n / n.norm(dim=0)
    cos_theta = (d * n).sum(dim=0)
    cos2_theta = cos_theta**2

    cos2_prime = 1 - mu**2 * (1 - cos2_theta)

    # Total internal reflection
    tir = cos2_prime < eps
    cos_prime = cos2_prime.where(~tir, 1.0).sqrt()

    d = (cos_prime - mu * cos_theta) * n + mu * d

    # Check for ray directions going backward
    backward = d[2] - eps < 0

    return d, tir, backward, cos2_prime, cos2_theta, n[-1]


def reset_bad_rays(
    r: torch.Tensor, d: torch.Tensor, ray_valid: torch.Tensor, normalize: bool = False
):
    """Reset and return position and direction vectors of rays that didn't trace successfully.

    The goal is to avoid NaNs in the forward/backward pass.

    Args:
        r: Ray position vectors (shape: [3, *]).
        d: Ray direction vectors (shape: [3, *]).
        ray_valid: Boolean mask that indicates rays that traced successfully (shape: [*]).
        normalize: Whether to re-normalize the direction vectors to prevent the propagation of numerical errors.
    """
    r = r.where(ray_valid, 0.0)
    d = d.where(ray_valid, get_z_mask(d))
    if normalize:
        with torch.no_grad():
            norm = d.norm(dim=0)
        d = d / norm
    return r, d


def shift_rays(r: torch.Tensor, s: torch.Tensor):
    """Return shifted ray coordinates so that coordinates in "z" are w.r.t. next optical surface.

    Args:
        r: Ray position vectors (shape: [3, *, n_lens]).
        s: Spacings (shape: [n_lens]).
    """
    r = r - get_z_mask(r) * s
    return r


def apply_phase_shift(
    r: torch.Tensor,
    d: torch.Tensor,
    p: torch.Tensor,
    wavelength_ratios: torch.Tensor,
    eps: float = 1e-6,
):
    """Return updated direction vectors based on polynomial coefficients "p" of a diffractive surface.

    The phase shift is given by:
        phi(r) = 2pi/lambda_0 * (p_0 r^2 + p_1 r^4 + p_2 r^6 + ...)

    The generalized Snell's law is given by:
        n' sin(i') = n sin(i) + (lambda/2pi) (d phi/d r)

        l' = l + (lambda/2pi) (d phi/d r) x/r
        m' = m + (lambda/2pi) (d phi/d r) y/r

    We assume air on each side.

    Args:
        r: Ray position vectors (shape: [3, *, n_wavelengths, n_lens]).
        d: Ray direction vectors in normalized form (shape: [3, *, n_wavelengths, n_lens]).
        p: Polynomial coefficients of the phase profile (shape: [n_lens, n_coefficients]).
        wavelength_ratios: Wavelength ratios (shape: [n_wavelengths]).
        eps: Small number to prevent division by zero.
    """
    # Compute d phi / d r
    delta_d = 0
    rho_squared = (r[:2] ** 2).sum(dim=0)
    for i, pp in enumerate(p.unbind(dim=-1)):
        delta_d = delta_d + (i + 1) * pp * rho_squared**i
    delta_d = delta_d * 2 * wavelength_ratios.view(-1, 1) * r[:2]
    lm = d[:2] + delta_d

    # Check for failures (rays going backward)
    n_square = 1 - (lm**2).sum(dim=0)
    failures = n_square - eps < 0

    n = n_square.clip(min=eps).sqrt()

    d = torch.cat((lm, n[None, ...]), dim=0)
    return d, failures, n_square
