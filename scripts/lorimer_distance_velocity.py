import numpy as np
import csv

import astropy.units as u
from astropy.coordinates import (
    SkyCoord,
    Galactocentric,
    CartesianDifferential,
    galactocentric_frame_defaults,
)


# ============================================================
# 1. Galactocentric / LSR setup
# ============================================================

galactocentric_frame_defaults.set("v4.0")

ASTROPY_GC_FRAME = Galactocentric()

R0 = ASTROPY_GC_FRAME.galcen_distance
Z_SUN = ASTROPY_GC_FRAME.z_sun
V_SUN_GALCEN = ASTROPY_GC_FRAME.galcen_v_sun

U_SUN = V_SUN_GALCEN.d_x
W_SUN = V_SUN_GALCEN.d_z

# Adopt V_sun in the direction of Galactic rotation.
# This is used only to decompose Astropy's total v_y into V_c + V_sun.
V_SUN_PEC_Y = 12.24 * u.km / u.s

# Astropy-implied circular speed under this decomposition.
VC_SUN = V_SUN_GALCEN.d_y - V_SUN_PEC_Y


def make_galactocentric_frame():
    """
    Build the Galactocentric frame using Astropy's latest default
    Galactocentric parameters.

    The total solar Galactocentric velocity is kept equal to Astropy's
    default value. The decomposition into V_c and V_sun is only used
    when subtracting the local circular velocity.
    """
    v_sun_galcen = CartesianDifferential(
        U_SUN,
        VC_SUN + V_SUN_PEC_Y,
        W_SUN,
    )

    return Galactocentric(
        galcen_distance=R0,
        z_sun=Z_SUN,
        galcen_v_sun=v_sun_galcen,
    )


GALCEN_FRAME = make_galactocentric_frame()


def sun_position_galactocentric():
    """
    Sun position in the adopted Galactocentric frame.
    """
    x_sun = -R0
    y_sun = 0.0 * u.kpc
    z_sun = Z_SUN.to(u.kpc)

    return x_sun, y_sun, z_sun


def circular_velocity_at_position(x, y, vc=VC_SUN):
    """
    Local circular velocity vector at Galactocentric Cartesian position.

    In the Astropy convention used here:
        Sun: x = -R0, y = 0
        local rotation at the Sun: +y

    Therefore:
        e_phi = (y/R, -x/R, 0)
    """
    R_cyl = np.sqrt(x**2 + y**2)

    ephi_x = y / R_cyl
    ephi_y = -x / R_cyl

    vx_c = vc * ephi_x
    vy_c = vc * ephi_y
    vz_c = np.zeros_like(vx_c.value) * vc.unit

    return vx_c, vy_c, vz_c


def transverse_projection(vx, vy, vz, nx, ny, nz):
    """
    Project a 3D velocity vector onto the sky plane perpendicular to
    the line-of-sight unit vector n.
    """
    v_dot_n = vx * nx + vy * ny + vz * nz

    vtx = vx - v_dot_n * nx
    vty = vy - v_dot_n * ny
    vtz = vz - v_dot_n * nz

    return vtx, vty, vtz


# ============================================================
# 2. Lorimer et al. 2006 Model C distance prior
# ============================================================

# ------------------------------------------------------------
# Lorimer Model C parameters.
#
# The radial distribution is:
#   rho_R(R) proportional to (R/R0)^B exp[-C (R - R0)/R0]
#
# Commonly used Model C values:
#   B = 1.9
#   C = 5.0
#
# The vertical distribution is taken as:
#   rho_z(z) proportional to exp(-|z|/z0)
#
# For normal pulsars, z0 = 0.33 kpc is a common choice.
# ------------------------------------------------------------

LORIMER_B = 1.9
LORIMER_C = 5.0
LORIMER_Z0_KPC = 0.33

# Use the same R0 and z_sun as the Astropy Galactocentric frame.
# This gives internal consistency between the distance prior and
# the LSR velocity correction.
LORIMER_PRIOR_R0_KPC = R0.to_value(u.kpc)
LORIMER_PRIOR_Z_SUN_KPC = Z_SUN.to_value(u.kpc)

# If you want to use the original Lorimer-style solar radius explicitly,
# replace the above line by:
# LORIMER_PRIOR_R0_KPC = 8.5
# LORIMER_PRIOR_Z_SUN_KPC = 0.0


def galactocentric_R_z_from_lbd(
    l_deg,
    b_deg,
    d_kpc,
    R0_kpc=LORIMER_PRIOR_R0_KPC,
    z_sun_kpc=LORIMER_PRIOR_Z_SUN_KPC,
):
    """
    Convert heliocentric Galactic coordinates and distance to
    Galactocentric cylindrical radius R and height z.

    Parameters
    ----------
    l_deg, b_deg : float
        Galactic longitude and latitude in degrees.

    d_kpc : array_like
        Heliocentric distance in kpc.

    R0_kpc : float
        Solar Galactocentric radius in kpc.

    z_sun_kpc : float
        Solar height above the Galactic plane in kpc.

    Returns
    -------
    R_kpc, z_kpc : ndarray
        Galactocentric cylindrical radius and vertical height.
    """
    l_rad = np.deg2rad(l_deg)
    b_rad = np.deg2rad(b_deg)

    d = np.asarray(d_kpc, dtype=float)

    d_plane = d * np.cos(b_rad)

    R_kpc = np.sqrt(
        R0_kpc**2
        + d_plane**2
        - 2.0 * R0_kpc * d_plane * np.cos(l_rad)
    )

    z_kpc = z_sun_kpc + d * np.sin(b_rad)

    return R_kpc, z_kpc


def log_lorimer_model_c_distance_prior(
    d_kpc,
    l_deg,
    b_deg,
    R0_kpc=LORIMER_PRIOR_R0_KPC,
    z_sun_kpc=LORIMER_PRIOR_Z_SUN_KPC,
    B=LORIMER_B,
    C=LORIMER_C,
    z0_kpc=LORIMER_Z0_KPC,
):
    """
    Logarithm of the Lorimer et al. 2006 Model C line-of-sight
    distance prior.

    The prior along a fixed line of sight is:

        p(d | l, b) proportional to
        (R/R0)^B exp[-C (R - R0)/R0] exp(-|z|/z0) d^2

    The d^2 factor is the line-of-sight volume element.

    Parameters
    ----------
    d_kpc : array_like
        Distance grid in kpc.

    l_deg, b_deg : float
        Galactic longitude and latitude in degrees.

    Returns
    -------
    log_prior : ndarray
        Log prior values, up to an arbitrary additive constant.
    """
    d = np.asarray(d_kpc, dtype=float)

    logp = np.full_like(d, -np.inf, dtype=float)
    valid = d > 0.0

    if not np.any(valid):
        return logp

    R_kpc, z_kpc = galactocentric_R_z_from_lbd(
        l_deg=l_deg,
        b_deg=b_deg,
        d_kpc=d[valid],
        R0_kpc=R0_kpc,
        z_sun_kpc=z_sun_kpc,
    )

    # Avoid log(0) exactly at R = 0.
    R_safe = np.maximum(R_kpc, 1.0e-8)

    log_radial = B * np.log(R_safe / R0_kpc) - C * (R_safe - R0_kpc) / R0_kpc
    log_vertical = -np.abs(z_kpc) / z0_kpc

    # Line-of-sight volume element.
    log_volume = 2.0 * np.log(d[valid])

    logp[valid] = log_radial + log_vertical + log_volume

    return logp


def log_parallax_likelihood_for_distance(
    d_kpc,
    parallax_mas,
    parallax_err_minus_mas,
    parallax_err_plus_mas,
):
    """
    Log likelihood p(parallax_obs | d).

    Units:
        d_kpc in kpc
        parallax_mas in mas

    Since 1 mas corresponds to 1 kpc:
        parallax_model_mas = 1 / d_kpc

    Asymmetric parallax errors are treated with a two-piece Gaussian:
        if parallax_model < parallax_obs, use parallax_err_minus_mas
        if parallax_model >= parallax_obs, use parallax_err_plus_mas

    For symmetric parallax errors, simply set:
        parallax_err_minus_mas = parallax_err_plus_mas = sigma_parallax
    """
    d = np.asarray(d_kpc, dtype=float)

    if parallax_err_minus_mas <= 0 or parallax_err_plus_mas <= 0:
        raise ValueError(
            "Parallax errors must be positive. "
            "Check parallax_err_minus_mas and parallax_err_plus_mas."
        )

    parallax_model_mas = 1.0 / d

    sigma = np.where(
        parallax_model_mas < parallax_mas,
        parallax_err_minus_mas,
        parallax_err_plus_mas,
    )

    loglike = -0.5 * ((parallax_mas - parallax_model_mas) / sigma)**2
    loglike += -np.log(sigma)

    return loglike


def sample_distance_lorimer_model_c(
    parallax_mas,
    parallax_err_minus_mas,
    parallax_err_plus_mas,
    l_deg,
    b_deg,
    n_mc=100,
    seed=12345,
    d_min_kpc=0.01,
    d_max_kpc=50.0,
    n_grid=20000,
):
    """
    Draw distance samples from:

        p(d | parallax, l, b)
        proportional to p(parallax | d) p_LorimerC(d | l, b)

    using a 1D grid and inverse-CDF sampling.

    Returns
    -------
    distance_samples_kpc : ndarray, shape (n_mc,)
        Distance samples in kpc.
    """
    rng = np.random.default_rng(seed)

    if parallax_err_minus_mas <= 0 or parallax_err_plus_mas <= 0:
        raise ValueError(
            "Parallax errors must be positive. "
            "You cannot use Lorimer posterior sampling with zero parallax error."
        )

    # Use a logarithmic grid to resolve both nearby and distant regions.
    d_grid = np.geomspace(d_min_kpc, d_max_kpc, n_grid)

    loglike = log_parallax_likelihood_for_distance(
        d_kpc=d_grid,
        parallax_mas=parallax_mas,
        parallax_err_minus_mas=parallax_err_minus_mas,
        parallax_err_plus_mas=parallax_err_plus_mas,
    )

    logprior = log_lorimer_model_c_distance_prior(
        d_kpc=d_grid,
        l_deg=l_deg,
        b_deg=b_deg,
    )

    logpost_density = loglike + logprior

    # Convert posterior density on a non-uniform grid into probability
    # mass per grid cell. This is important because d_grid is logarithmic.
    dd = np.gradient(d_grid)
    logmass = logpost_density + np.log(dd)

    finite = np.isfinite(logmass)
    if not np.any(finite):
        raise RuntimeError(
            "All posterior grid values are non-finite. "
            "Check parallax, errors, and distance grid."
        )

    max_logmass = np.max(logmass[finite])
    weights = np.zeros_like(logmass)
    weights[finite] = np.exp(logmass[finite] - max_logmass)

    weight_sum = np.sum(weights)
    if weight_sum <= 0 or not np.isfinite(weight_sum):
        raise RuntimeError(
            "Posterior weights are invalid. "
            "Try increasing d_max_kpc or checking the input data."
        )

    weights /= weight_sum

    cdf = np.cumsum(weights)
    cdf /= cdf[-1]

    # Build inverse CDF. Prepend zero to make interpolation stable.
    cdf_for_interp = np.concatenate(([0.0], cdf))
    d_for_interp = np.concatenate(([d_grid[0]], d_grid))

    # Remove repeated CDF values.
    keep = np.concatenate(([True], np.diff(cdf_for_interp) > 0.0))
    cdf_for_interp = cdf_for_interp[keep]
    d_for_interp = d_for_interp[keep]

    u_random = rng.uniform(0.0, 1.0, n_mc)
    distance_samples_kpc = np.interp(u_random, cdf_for_interp, d_for_interp)

    # Optional warning if the posterior is close to the upper grid boundary.
    if np.percentile(distance_samples_kpc, 99) > 0.9 * d_max_kpc:
        print(
            "Warning: distance posterior may be close to d_max_kpc. "
            f"l={l_deg:.3f}, b={b_deg:.3f}, "
            f"parallax={parallax_mas:.5f} mas. "
            "Consider increasing d_max_kpc."
        )

    return distance_samples_kpc


# ============================================================
# 3. One-source LSR-corrected transverse peculiar velocity
#    using pmra and pmdec, with distance as direct input
# ============================================================

def vpec_trans_single_from_pmra_pmdec_distance(
    l_deg,
    b_deg,
    pmra_masyr,
    pmdec_masyr,
    distance_kpc,
):
    """
    Compute LSR-corrected transverse peculiar velocity for one source.

    Inputs
    ------
    l_deg, b_deg : float
        Galactic longitude and latitude in degrees.

    pmra_masyr, pmdec_masyr : float
        ICRS proper motions:
            pmra_masyr  = mu_alpha * cos(delta), in mas/yr
            pmdec_masyr = mu_delta, in mas/yr

    distance_kpc : float
        Distance in kpc.

    Returns
    -------
    vpec_trans : float
        LSR-corrected transverse peculiar speed in km/s.

    Notes
    -----
    radial_velocity is set to zero. The final transverse projection
    removes the line-of-sight component, so the unknown radial velocity
    does not enter the projected transverse peculiar speed.
    """
    if distance_kpc <= 0 or not np.isfinite(distance_kpc):
        return np.nan

    distance = distance_kpc * u.kpc

    # Use l,b to determine the sky position, then convert to ICRS.
    c_pos_gal = SkyCoord(
        l=l_deg * u.deg,
        b=b_deg * u.deg,
        frame="galactic",
    )
    c_pos_icrs = c_pos_gal.icrs

    # Build ICRS coordinate with pmra and pmdec.
    c_icrs = SkyCoord(
        ra=c_pos_icrs.ra,
        dec=c_pos_icrs.dec,
        distance=distance,
        pm_ra_cosdec=pmra_masyr * u.mas / u.yr,
        pm_dec=pmdec_masyr * u.mas / u.yr,
        radial_velocity=0.0 * u.km / u.s,
        frame="icrs",
    )

    c_galcen = c_icrs.transform_to(GALCEN_FRAME)

    # Galactocentric position
    x = c_galcen.cartesian.x.to(u.kpc)
    y = c_galcen.cartesian.y.to(u.kpc)
    z = c_galcen.cartesian.z.to(u.kpc)

    # Galactocentric velocity
    vx = c_galcen.velocity.d_x.to(u.km / u.s)
    vy = c_galcen.velocity.d_y.to(u.km / u.s)
    vz = c_galcen.velocity.d_z.to(u.km / u.s)

    # Subtract local circular velocity at the pulsar position.
    vcx, vcy, vcz = circular_velocity_at_position(x, y)

    vpec_x = vx - vcx
    vpec_y = vy - vcy
    vpec_z = vz - vcz

    # Line-of-sight unit vector from the Sun to the pulsar.
    x_sun, y_sun, z_sun = sun_position_galactocentric()

    rx = x - x_sun
    ry = y - y_sun
    rz = z - z_sun

    rnorm = np.sqrt(rx**2 + ry**2 + rz**2)

    nx = (rx / rnorm).decompose()
    ny = (ry / rnorm).decompose()
    nz = (rz / rnorm).decompose()

    # Project the peculiar velocity onto the sky plane.
    vtx, vty, vtz = transverse_projection(
        vpec_x,
        vpec_y,
        vpec_z,
        nx,
        ny,
        nz,
    )

    vpec_trans = np.sqrt(vtx**2 + vty**2 + vtz**2)

    return vpec_trans.to_value(u.km / u.s)


def heliocentric_vtrans_single_from_pm_distance(
    pmra_masyr,
    pmdec_masyr,
    distance_kpc,
):
    """
    Compute ordinary heliocentric transverse speed:

        V_trans = 4.74047 * mu_tot * d

    where mu_tot is in mas/yr and d is in kpc.
    """
    if distance_kpc <= 0 or not np.isfinite(distance_kpc):
        return np.nan

    mu_tot = np.sqrt(pmra_masyr**2 + pmdec_masyr**2)
    vtrans = 4.74047 * mu_tot * distance_kpc

    return vtrans


# ============================================================
# 4. One-source Monte Carlo wrapper: Lorimer Model C version
# ============================================================

def simulate_one_source_lsr_velocities_from_pmra_pmdec_lorimer(
    source_name,
    l_deg,
    b_deg,
    pmra_masyr,
    pmdec_masyr,
    parallax_mas,
    parallax_err_minus_mas,
    parallax_err_plus_mas,
    n_mc=100,
    seed=12345,
):
    """
    Generate Monte Carlo samples of distance and transverse velocities
    for one source using the Lorimer et al. 2006 Model C distance prior.

    Only parallax uncertainty is sampled here.
    Proper-motion uncertainties are not sampled.

    Outputs include:
        distance_samples_kpc
        vtrans_helio_samples
        vpec_lsr_samples
    """
    distance_samples_kpc = sample_distance_lorimer_model_c(
        parallax_mas=parallax_mas,
        parallax_err_minus_mas=parallax_err_minus_mas,
        parallax_err_plus_mas=parallax_err_plus_mas,
        l_deg=l_deg,
        b_deg=b_deg,
        n_mc=n_mc,
        seed=seed,
    )

    vtrans_helio_samples = np.array([
        heliocentric_vtrans_single_from_pm_distance(
            pmra_masyr=pmra_masyr,
            pmdec_masyr=pmdec_masyr,
            distance_kpc=d_k,
        )
        for d_k in distance_samples_kpc
    ])

    vpec_lsr_samples = np.array([
        vpec_trans_single_from_pmra_pmdec_distance(
            l_deg=l_deg,
            b_deg=b_deg,
            pmra_masyr=pmra_masyr,
            pmdec_masyr=pmdec_masyr,
            distance_kpc=d_k,
        )
        for d_k in distance_samples_kpc
    ])

    finite = np.isfinite(vpec_lsr_samples)

    if not np.any(finite):
        return {
            "source_name": source_name,
            "distance_samples_kpc": distance_samples_kpc,
            "vtrans_helio_samples": vtrans_helio_samples,
            "vpec_lsr_samples": np.full(n_mc, np.nan),
            "d16": np.nan,
            "d50": np.nan,
            "d84": np.nan,
            "vtrans16": np.nan,
            "vtrans50": np.nan,
            "vtrans84": np.nan,
            "v16": np.nan,
            "v50": np.nan,
            "v84": np.nan,
            "err_minus": np.nan,
            "err_plus": np.nan,
        }

    # If any velocity samples failed, refill with the median to keep fixed shape.
    if np.sum(finite) < n_mc:
        median_fill = np.nanmedian(vpec_lsr_samples)
        vpec_lsr_samples = np.where(
            np.isfinite(vpec_lsr_samples),
            vpec_lsr_samples,
            median_fill,
        )

    d16, d50, d84 = np.percentile(distance_samples_kpc, [16, 50, 84])
    vt16, vt50, vt84 = np.percentile(vtrans_helio_samples, [16, 50, 84])
    v16, v50, v84 = np.percentile(vpec_lsr_samples, [16, 50, 84])

    return {
        "source_name": source_name,
        "distance_samples_kpc": distance_samples_kpc,
        "vtrans_helio_samples": vtrans_helio_samples,
        "vpec_lsr_samples": vpec_lsr_samples,
        "d16": d16,
        "d50": d50,
        "d84": d84,
        "vtrans16": vt16,
        "vtrans50": vt50,
        "vtrans84": vt84,
        "v16": v16,
        "v50": v50,
        "v84": v84,
        "err_minus": v50 - v16,
        "err_plus": v84 - v50,
    }


# ============================================================
# 5. Batch calculation for 77 sources
# ============================================================

def generate_lsr_velocity_matrix_for_sample_lorimer(
    source_names,
    l_deg_all,
    b_deg_all,
    pmra_all,
    pmdec_all,
    parallax_all,
    parallax_err_minus_all,
    parallax_err_plus_all,
    n_mc=100,
    seed0=20260610,
):
    """
    Generate distance and velocity sample matrices for the full pulsar sample.

    Outputs
    -------
    distance_samples_matrix : ndarray, shape (n_source, n_mc)
        Row i contains n_mc Lorimer-prior distance samples for source i.

    vtrans_helio_samples_matrix : ndarray, shape (n_source, n_mc)
        Row i contains n_mc ordinary heliocentric transverse speed samples.

    vpec_lsr_samples_matrix : ndarray, shape (n_source, n_mc)
        Row i contains n_mc LSR-corrected transverse peculiar velocity samples.

    summary_rows : list of dict
        Each row stores source name, distance median, velocity median,
        lower error and upper error.
    """
    source_names = np.asarray(source_names)
    l_deg_all = np.asarray(l_deg_all, dtype=float)
    b_deg_all = np.asarray(b_deg_all, dtype=float)
    pmra_all = np.asarray(pmra_all, dtype=float)
    pmdec_all = np.asarray(pmdec_all, dtype=float)
    parallax_all = np.asarray(parallax_all, dtype=float)
    parallax_err_minus_all = np.asarray(parallax_err_minus_all, dtype=float)
    parallax_err_plus_all = np.asarray(parallax_err_plus_all, dtype=float)

    n_source = len(source_names)

    distance_samples_matrix = np.zeros((n_source, n_mc))
    vtrans_helio_samples_matrix = np.zeros((n_source, n_mc))
    vpec_lsr_samples_matrix = np.zeros((n_source, n_mc))

    summary_rows = []

    print()
    print("=" * 80)
    print("Generating Lorimer Model C distance and velocity samples")
    print("=" * 80)
    print(f"Number of sources               = {n_source}")
    print(f"Samples per source              = {n_mc}")
    print(f"Astropy R0                      = {R0}")
    print(f"Astropy z_sun                   = {Z_SUN}")
    print(f"Astropy galcen_v_sun            = {V_SUN_GALCEN}")
    print(f"VC_SUN inferred                 = {VC_SUN}")
    print("-" * 80)
    print("Lorimer Model C prior parameters")
    print(f"B                               = {LORIMER_B}")
    print(f"C                               = {LORIMER_C}")
    print(f"z0                              = {LORIMER_Z0_KPC} kpc")
    print(f"R0 used in prior                = {LORIMER_PRIOR_R0_KPC} kpc")
    print(f"z_sun used in prior             = {LORIMER_PRIOR_Z_SUN_KPC} kpc")
    print("=" * 80)

    for i in range(n_source):
        result_i = simulate_one_source_lsr_velocities_from_pmra_pmdec_lorimer(
            source_name=source_names[i],
            l_deg=l_deg_all[i],
            b_deg=b_deg_all[i],
            pmra_masyr=pmra_all[i],
            pmdec_masyr=pmdec_all[i],
            parallax_mas=parallax_all[i],
            parallax_err_minus_mas=parallax_err_minus_all[i],
            parallax_err_plus_mas=parallax_err_plus_all[i],
            n_mc=n_mc,
            seed=seed0 + i,
        )

        distance_samples_matrix[i, :] = result_i["distance_samples_kpc"]
        vtrans_helio_samples_matrix[i, :] = result_i["vtrans_helio_samples"]
        vpec_lsr_samples_matrix[i, :] = result_i["vpec_lsr_samples"]

        summary_rows.append({
            "index": i,
            "source_name": str(source_names[i]),

            "d16": result_i["d16"],
            "d50": result_i["d50"],
            "d84": result_i["d84"],
            "d_err_minus": result_i["d50"] - result_i["d16"],
            "d_err_plus": result_i["d84"] - result_i["d50"],

            "vtrans16": result_i["vtrans16"],
            "vtrans50": result_i["vtrans50"],
            "vtrans84": result_i["vtrans84"],
            "vtrans_err_minus": result_i["vtrans50"] - result_i["vtrans16"],
            "vtrans_err_plus": result_i["vtrans84"] - result_i["vtrans50"],

            "vpec16": result_i["v16"],
            "vpec50": result_i["v50"],
            "vpec84": result_i["v84"],
            "vpec_err_minus": result_i["err_minus"],
            "vpec_err_plus": result_i["err_plus"],
        })

        print(
            f"{i + 1:03d}/{n_source:03d} "
            f"{str(source_names[i]):>14s}: "
            f"d = {result_i['d50']:.3f} "
            f"+{result_i['d84'] - result_i['d50']:.3f} "
            f"-{result_i['d50'] - result_i['d16']:.3f} kpc; "
            f"Vpec_LSR = {result_i['v50']:.3f} "
            f"+{result_i['err_plus']:.3f} "
            f"-{result_i['err_minus']:.3f} km/s"
        )

    return (
        distance_samples_matrix,
        vtrans_helio_samples_matrix,
        vpec_lsr_samples_matrix,
        summary_rows,
    )


# ============================================================
