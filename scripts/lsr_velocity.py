"""Final parallax-based LSR method extracted from peculiar_velocity.ipynb cell 13.
Frame explicitly frozen to v4.0, identical to latest in Astropy 6.1.7.
"""
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
    default value:
        (12.9, 245.6, 7.78) km/s

    The decomposition into V_c and V_sun is only used when subtracting
    the local circular velocity.
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
    R = np.sqrt(x**2 + y**2)

    ephi_x = y / R
    ephi_y = -x / R

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
# 2. One-source LSR-corrected transverse peculiar velocity
#    using pmra and pmdec
# ============================================================

def vpec_trans_single_from_pmra_pmdec(
    l_deg,
    b_deg,
    pmra_masyr,
    pmdec_masyr,
    parallax_mas,
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

    parallax_mas : float
        Parallax in mas.

    Returns
    -------
    vpec_trans : float
        LSR-corrected transverse peculiar speed in km/s.

    Notes
    -----
    This function is appropriate for your current data arrays, because
    your proper motions are pmra and pmdec, not pm_l_cosb and pm_b.
    """
    if parallax_mas <= 0:
        return np.nan

    distance = (1.0 / parallax_mas) * u.kpc

    # Use l,b to determine the sky position, then convert to ICRS.
    c_pos_gal = SkyCoord(
        l=l_deg * u.deg,
        b=b_deg * u.deg,
        frame="galactic",
    )
    c_pos_icrs = c_pos_gal.icrs

    # Build ICRS coordinate with pmra and pmdec.
    # radial_velocity is set to zero; the final transverse projection
    # removes the line-of-sight component.
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


# ============================================================
# 3. Asymmetric parallax sampler
# ============================================================

def draw_asymmetric_parallax(
    parallax_mas,
    parallax_err_minus_mas,
    parallax_err_plus_mas,
    n_mc=100,
    seed=12345,
):
    """
    Draw Monte Carlo parallax samples with asymmetric Gaussian errors.

    The left and right error bars are treated as half-Gaussian widths.
    Non-positive parallaxes are rejected.
    """
    rng = np.random.default_rng(seed)
    samples = []

    while len(samples) < n_mc:
        side = rng.random(n_mc) < 0.5

        offsets_left = np.abs(
            rng.normal(
                0.0,
                parallax_err_minus_mas,
                size=n_mc,
            )
        )

        offsets_right = np.abs(
            rng.normal(
                0.0,
                parallax_err_plus_mas,
                size=n_mc,
            )
        )

        draws = np.where(
            side,
            parallax_mas - offsets_left,
            parallax_mas + offsets_right,
        )

        draws = draws[draws > 0]

        samples.extend(draws.tolist())

    return np.array(samples[:n_mc])


# ============================================================
# 4. One-source Monte Carlo wrapper
# ============================================================

def simulate_one_source_lsr_velocities_from_pmra_pmdec(
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
    Generate Monte Carlo samples of LSR-corrected transverse peculiar
    velocity for one source.

    Only parallax uncertainty is sampled here.
    Proper-motion uncertainties are not sampled.
    """
    parallax_samples = draw_asymmetric_parallax(
        parallax_mas=parallax_mas,
        parallax_err_minus_mas=parallax_err_minus_mas,
        parallax_err_plus_mas=parallax_err_plus_mas,
        n_mc=n_mc,
        seed=seed,
    )

    vpec_samples = np.array([
        vpec_trans_single_from_pmra_pmdec(
            l_deg=l_deg,
            b_deg=b_deg,
            pmra_masyr=pmra_masyr,
            pmdec_masyr=pmdec_masyr,
            parallax_mas=par_k,
        )
        for par_k in parallax_samples
    ])

    vpec_samples = vpec_samples[np.isfinite(vpec_samples)]

    if len(vpec_samples) == 0:
        return {
            "source_name": source_name,
            "parallax_samples": parallax_samples,
            "vpec_samples": np.full(n_mc, np.nan),
            "v16": np.nan,
            "v50": np.nan,
            "v84": np.nan,
            "err_minus": np.nan,
            "err_plus": np.nan,
        }

    # If any samples failed, refill with the median to keep a fixed matrix shape.
    if len(vpec_samples) < n_mc:
        median_fill = np.nanmedian(vpec_samples)
        vpec_samples = np.pad(
            vpec_samples,
            (0, n_mc - len(vpec_samples)),
            mode="constant",
            constant_values=median_fill,
        )

    v16, v50, v84 = np.percentile(vpec_samples, [16, 50, 84])

    return {
        "source_name": source_name,
        "parallax_samples": parallax_samples,
        "vpec_samples": vpec_samples,
        "v16": v16,
        "v50": v50,
        "v84": v84,
        "err_minus": v50 - v16,
        "err_plus": v84 - v50,
    }


# ============================================================
# 5. Batch calculation for 77 sources
# ============================================================

def generate_lsr_velocity_matrix_for_sample(
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
    Generate a velocity-sample matrix for the full pulsar sample.

    Output
    ------
    vpec_lsr_samples_matrix : ndarray, shape (n_source, n_mc)
        Row i contains n_mc LSR-corrected transverse velocity samples
        for source i.

    summary_rows : list of dict
        Each row stores source name, median, lower error and upper error.
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

    vpec_lsr_samples_matrix = np.zeros((n_source, n_mc))
    summary_rows = []

    print()
    print("=" * 80)
    print("Generating LSR-corrected transverse peculiar velocity samples")
    print("=" * 80)
    print(f"Number of sources  = {n_source}")
    print(f"Samples per source = {n_mc}")
    print(f"R0                 = {R0}")
    print(f"z_sun              = {Z_SUN}")
    print(f"galcen_v_sun       = {V_SUN_GALCEN}")
    print(f"VC_SUN inferred    = {VC_SUN}")
    print("=" * 80)

    for i in range(n_source):
        result_i = simulate_one_source_lsr_velocities_from_pmra_pmdec(
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

        vpec_lsr_samples_matrix[i, :] = result_i["vpec_samples"]

        summary_rows.append({
            "index": i,
            "source_name": str(source_names[i]),
            "v16": result_i["v16"],
            "v50": result_i["v50"],
            "v84": result_i["v84"],
            "err_minus": result_i["err_minus"],
            "err_plus": result_i["err_plus"],
        })

        print(
            f"{i + 1:03d}/{n_source:03d} "
            f"{str(source_names[i]):>14s}: "
            f"{result_i['v50']:.3f} "
            f"+{result_i['err_plus']:.3f} "
            f"-{result_i['err_minus']:.3f} km/s"
        )

    return vpec_lsr_samples_matrix, summary_rows


# ============================================================
