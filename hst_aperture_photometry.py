#!/usr/bin/env python3
"""
hst_aperture_photometry.py
==========================

Curve-of-growth aperture photometry on HST ACS/WFC drizzled images,
with comparison to HAP catalog values and ground-based references.

Given only an RA/Dec (and a proposal ID to narrow the MAST query),
this script will:
  1. Query MAST for HAP single-visit mosaic (SVM) observations
  2. Download the drizzled DRC images (science + weight)
  3. Read flux calibration from FITS headers (PHOTFLAM, PHOTPLAM)
  4. Perform circular aperture photometry at a range of radii
  5. Estimate local background from a sigma-clipped annulus
  6. Download and cross-match the HAP point and segment catalogs
  7. Produce a curve-of-growth plot and a summary table

Usage
-----
    python hst_aperture_photometry.py 150.0 2.2 --proposal-id 12345

Dependencies
------------
    astropy, astroquery, photutils, matplotlib, numpy, scipy

Notes
-----
- Images are ACS/WFC DRC files in units of ELECTRONS/S (rate images).
  Flux calibration: f_lambda = PHOTFLAM * (count rate in e/s).
  The AB zeropoint is: ZPT = -2.5*log10(PHOTFLAM) - 5*log10(PHOTPLAM) - 2.408
  so that mag_AB = -2.5*log10(count_rate) + ZPT.

- The DRC pixel scale is 0.05"/pixel for standard ACS/WFC drizzle products.

- "Curve of growth" means measuring the enclosed flux in apertures of
  increasing radius.  For a well-isolated source the curve should flatten
  once the aperture captures all the light.  Comparing where the HAP
  catalog value falls on this curve tells you how much flux was missed.

- Background is estimated from a sigma-clipped annulus around the source.
  The default annulus (3"-4.5") is chosen to be well outside the source
  wings but inside the ~10" scale where the ICL / cluster light gradient
  becomes significant.  Adjust if your source is in a crowded field.
"""

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np

# Suppress noisy warnings from astropy/astroquery during normal use
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
C_ANGSTROM_PER_S = 2.99792458e18   # speed of light in Angstrom / s


# ===================================================================
# 1. MAST query and download
# ===================================================================

def query_mast_hap(ra, dec, proposal_id=None, search_radius_arcmin=5.0,
                   max_sep_arcsec=10.0):
    """
    Query MAST for HAP single-visit mosaic (HAP-SVM) ACS/WFC observations
    covering a given sky position.

    The search strategy is:
      1. Cone search at (ra, dec) for all HAP-SVM ACS/WFC images.
         If ``proposal_id`` is given, restrict to that program.
      2. Group results by (proposal_id, target_name) - each group
         represents one visit/pointing with one or more filters.
      3. For each group, verify the source actually falls on the
         detector by checking that it lands on a pixel with nonzero
         weight (done later in the pipeline).  Here we just pick the
         group whose pointing center is closest to the source.
      4. If multiple groups are within ``max_sep_arcsec`` of each other
         (i.e. overlapping pointings), prefer the one with the most
         filters, then the deepest total exposure time.

    Parameters
    ----------
    ra, dec : float
        Target coordinates in decimal degrees (ICRS).
    proposal_id : str or None
        Optional HST proposal ID to restrict the search.  If None,
        all programs covering this position are considered.
    search_radius_arcmin : float
        Cone-search radius in arcminutes.  Default 5' covers the full
        ACS/WFC diagonal (~3.4') with margin.
    max_sep_arcsec : float
        When auto-selecting among multiple groups, groups whose pointing
        centers are within this distance of the closest group are treated
        as "tied" and ranked by depth.  Default: 10".

    Returns
    -------
    obs : astropy.table.Table
        Filtered observation table for the selected group.
    """
    from astroquery.mast import Observations
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    from collections import defaultdict

    target = SkyCoord(ra=ra, dec=dec, unit="deg")

    # --- Step 1: Query MAST ---
    if proposal_id:
        # Criteria search filtered by proposal
        obs = Observations.query_criteria(
            proposal_id=proposal_id,
            instrument_name="ACS/WFC",
            dataproduct_type="image",
            provenance_name="HAP-SVM",
        )
    else:
        # Cone search by position
        all_obs = Observations.query_region(
            target, radius=search_radius_arcmin * u.arcmin,
        )
        obs = all_obs[
            (all_obs["instrument_name"] == "ACS/WFC") &
            (all_obs["provenance_name"] == "HAP-SVM") &
            (all_obs["dataproduct_type"] == "image")
        ]

    # Drop 'detection' combined-filter products
    obs = obs[[f != "detection" for f in obs["filters"]]]

    if len(obs) == 0:
        loc = f"proposal {proposal_id}" if proposal_id else f"({ra:.5f}, {dec:.5f})"
        raise RuntimeError(f"No HAP-SVM ACS/WFC observations found for {loc}.")

    # If we searched by proposal, still filter by position (proposals can
    # have multiple widely separated targets)
    if proposal_id:
        obs_coords = SkyCoord(ra=obs["s_ra"], dec=obs["s_dec"], unit="deg")
        seps = target.separation(obs_coords)
        obs = obs[seps < search_radius_arcmin * u.arcmin]
        if len(obs) == 0:
            raise RuntimeError(
                f"No observations within {search_radius_arcmin}' of "
                f"({ra:.5f}, {dec:.5f}) in proposal {proposal_id}."
            )

    # --- Step 2: Group by (proposal_id, target_name) ---
    groups = defaultdict(list)
    for row in obs:
        key = (str(row["proposal_id"]), str(row["target_name"]))
        groups[key].append(row)

    # --- Step 3: Rank groups ---
    #   Primary:  closest pointing center to target
    #   Tiebreak: most filters, then deepest total exposure
    group_scores = []
    for (pid, tgt), rows in groups.items():
        # Mean pointing center for this group
        mean_ra = np.mean([float(r["s_ra"]) for r in rows])
        mean_dec = np.mean([float(r["s_dec"]) for r in rows])
        center = SkyCoord(ra=mean_ra, dec=mean_dec, unit="deg")
        sep = target.separation(center).arcsec

        n_filters = len(set(r["filters"] for r in rows))
        total_exp = sum(float(r["t_exptime"]) for r in rows)

        group_scores.append((sep, -n_filters, -total_exp, pid, tgt, rows))

    group_scores.sort()  # sorts by (separation, -n_filters, -total_exp)

    # Report what was found
    print(f"Found {len(group_scores)} observation group(s):")
    for sep, neg_nf, neg_texp, pid, tgt, rows in group_scores:
        filters = sorted(set(r["filters"] for r in rows))
        print(f"  Proposal {pid}, target '{tgt}': {filters}, "
              f"pointing {sep:.1f}\" from source, "
              f"total exp {-neg_texp:.0f}s")

    # Select the best group
    best = group_scores[0]
    sep, _, _, pid, tgt, rows = best

    # Check if there are near-ties (multiple groups at similar distance)
    ties = [g for g in group_scores
            if g[0] < best[0] + max_sep_arcsec and g is not best]
    if ties:
        print(f"\n  Note: {len(ties)} other group(s) at similar distance. "
              f"Use --proposal-id to select a specific one.")

    print(f"\n  -> Selected: proposal {pid}, target '{tgt}'")

    # Reconstruct the observation table for the selected group
    selected_obsids = set(r["obs_id"] for r in rows)
    obs = obs[[str(r["obs_id"]) in selected_obsids for r in obs]]

    print(f"  {len(obs)} observations:")
    for row in obs:
        print(f"    {row['obs_id']}  filter={row['filters']}  "
              f"t_exp={row['t_exptime']:.0f}s")

    return obs


def download_drizzled_images(obs, download_dir):
    """
    Download the combined single-visit DRC mosaics for each filter.

    The HAP pipeline produces both individual-exposure DRC files and a
    combined mosaic per filter.  The combined mosaics have obs_id like
    ``hst_17114_04_acs_wfc_f475w_jf0004`` (no trailing exposure suffix).
    We identify them by checking that the obs_id ends with the association
    root (no extra characters after the filter + root).

    Returns
    -------
    paths : dict
        {filter_name: local_fits_path}
    """
    from astroquery.mast import Observations

    products = Observations.get_product_list(obs)
    drcs = products[products["productSubGroupDescription"] == "DRC"]

    # The combined mosaics have shorter obs_ids (no individual exposure suffix).
    # Strategy: for each filter, pick the DRC file whose obs_id is shortest
    # (that's the combined one, e.g. "..._f475w_jf0004" vs "..._f475w_jf0004yi").
    filter_products = {}
    for row in drcs:
        fname = row["productFilename"]
        oid = row["obs_id"]
        # Identify the filter from the filename
        for filt in ("f275w", "f336w", "f435w", "f475w", "f555w",
                     "f606w", "f625w", "f775w", "f814w", "f850lp"):
            if filt in fname.lower():
                filt_upper = filt.upper()
                if filt_upper not in filter_products:
                    filter_products[filt_upper] = (oid, row)
                else:
                    # Keep the one with the shorter obs_id (= combined mosaic)
                    existing_oid = filter_products[filt_upper][0]
                    if len(oid) < len(existing_oid):
                        filter_products[filt_upper] = (oid, row)
                break

    # Download
    from astropy.table import vstack, Table

    to_download = Table(
        rows=[v[1] for v in filter_products.values()],
        names=drcs.colnames,
    )
    manifest = Observations.download_products(to_download, download_dir=download_dir)

    paths = {}
    for row in manifest:
        local = row["Local Path"]
        for filt in filter_products:
            if filt.lower() in local.lower() and local.endswith("_drc.fits"):
                paths[filt] = local
                break

    print(f"\nDownloaded {len(paths)} DRC files:")
    for filt, p in sorted(paths.items()):
        print(f"  {filt}: {os.path.basename(p)}")

    return paths


def download_hap_catalogs(obs, download_dir):
    """
    Download HAP point-source and segment catalogs (ECSV files).

    Returns
    -------
    catalog_paths : dict
        {(filter, catalog_type): local_path}
        e.g. {('F475W', 'point'): '...point-cat.ecsv', ('F475W', 'segment'): ...}
    """
    from astroquery.mast import Observations

    products = Observations.get_product_list(obs)

    cat_mask = [
        "cat" in str(f).lower() and "total" not in str(oid).lower()
        for f, oid in zip(products["productFilename"], products["obs_id"])
    ]
    cats = products[cat_mask]

    if len(cats) == 0:
        print("Warning: no HAP catalogs found.")
        return {}

    manifest = Observations.download_products(cats, download_dir=download_dir)

    catalog_paths = {}
    for row in manifest:
        local = row["Local Path"]
        basename = os.path.basename(local).lower()

        # Determine filter
        filt = None
        for f in ("f275w", "f336w", "f435w", "f475w", "f555w",
                  "f606w", "f625w", "f775w", "f814w", "f850lp"):
            if f in basename:
                filt = f.upper()
                break
        if filt is None:
            continue

        # Determine catalog type
        if "point-cat" in basename:
            catalog_paths[(filt, "point")] = local
        elif "segment-cat" in basename:
            catalog_paths[(filt, "segment")] = local

    print(f"\nDownloaded {len(catalog_paths)} HAP catalogs:")
    for key, path in sorted(catalog_paths.items()):
        print(f"  {key[0]} {key[1]}: {os.path.basename(path)}")

    return catalog_paths


# ===================================================================
# 2. FITS inspection and calibration
# ===================================================================

def read_image_and_calibration(fits_path, ra, dec):
    """
    Open a DRC FITS file and extract everything needed for photometry.

    Parameters
    ----------
    fits_path : str
        Path to the DRC FITS file.
    ra, dec : float
        Target coordinates (degrees).

    Returns
    -------
    info : dict with keys
        'sci'       : 2D science array (electrons/s)
        'wht'       : 2D weight array
        'wcs'       : WCS object
        'photflam'  : inverse sensitivity (erg/s/cm^2/Angstrom per e/s)
        'photplam'  : pivot wavelength (Angstrom)
        'zpt_ab'    : AB zeropoint such that mag = -2.5*log10(count_rate) + zpt
        'pixscale'  : pixel scale (arcsec/pixel)
        'exptime'   : total exposure time (seconds)
        'src_x', 'src_y' : source pixel position (0-indexed, float)
        'bunit'     : data units string
    """
    from astropy.io import fits as pyfits
    from astropy.wcs import WCS
    from astropy.coordinates import SkyCoord

    hdul = pyfits.open(fits_path)
    sci_hdr = hdul[1].header
    pri_hdr = hdul[0].header

    photflam = sci_hdr["PHOTFLAM"]
    photplam = sci_hdr["PHOTPLAM"]
    bunit = sci_hdr.get("BUNIT", "UNKNOWN")

    # AB zeropoint:
    #   mag_AB = -2.5*log10(f_lambda) - 5*log10(lambda_pivot) - 2.408
    # Since f_lambda = PHOTFLAM * count_rate:
    #   mag_AB = -2.5*log10(count_rate) + [-2.5*log10(PHOTFLAM) - 5*log10(PHOTPLAM) - 2.408]
    #                                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    #                                                    = zpt_ab
    zpt_ab = -2.5 * np.log10(photflam) - 5.0 * np.log10(photplam) - 2.408

    # Pixel scale from CD matrix
    cd11 = sci_hdr["CD1_1"]
    cd12 = sci_hdr.get("CD1_2", 0.0)
    cd21 = sci_hdr.get("CD2_1", 0.0)
    cd22 = sci_hdr["CD2_2"]
    pixscale = np.sqrt(abs(cd11 * cd22 - cd12 * cd21)) * 3600.0  # arcsec/pix

    wcs = WCS(sci_hdr)
    target = SkyCoord(ra=ra, dec=dec, unit="deg")
    px, py = wcs.world_to_pixel(target)
    src_x, src_y = float(px), float(py)

    info = dict(
        sci=hdul[1].data,
        wht=hdul[2].data,
        wcs=wcs,
        photflam=photflam,
        photplam=photplam,
        zpt_ab=zpt_ab,
        pixscale=pixscale,
        exptime=pri_hdr.get("EXPTIME", None),
        src_x=src_x,
        src_y=src_y,
        bunit=bunit,
    )

    # Sanity checks
    ny, nx = info["sci"].shape
    if not (0 <= src_x < nx and 0 <= src_y < ny):
        raise ValueError(
            f"Target pixel ({src_x:.1f}, {src_y:.1f}) is outside the "
            f"image ({nx}x{ny}).  Check your coordinates."
        )
    if bunit.upper() != "ELECTRONS/S":
        print(f"  WARNING: BUNIT = '{bunit}' - expected ELECTRONS/S.  "
              f"Calibration may be wrong.")

    hdul.close()
    return info


# ===================================================================
# 3. Aperture photometry
# ===================================================================

def aperture_photometry_cog(
    sci,
    wht,
    src_x,
    src_y,
    pixscale,
    photflam,
    photplam,
    zpt_ab,
    radii_arcsec=None,
    bkg_inner_arcsec=3.0,
    bkg_outer_arcsec=4.5,
):
    """
    Perform circular aperture photometry at a series of radii to build
    a curve of growth, with formal error propagation via the IVM weight map.

    Error budget
    ------------
    The DRC weight extension is an inverse-variance map (IVM), so the
    variance of each pixel is 1/wht (in (e/s)^2).  For the raw aperture
    sum the variance is simply the sum of per-pixel variances.

    Background subtraction adds a second independent term: the uncertainty
    in our estimate of the background level, propagated over the N_aper
    pixels being subtracted.

    Total variance of the net count rate:
        sigma_net^2 = sum(1/wht_i)  +  N_aper^2 * bkg_std^2 / N_bkg

    This is then propagated linearly through the PHOTFLAM calibration:
        sigma_f_lambda = PHOTFLAM * sigma_net
        sigma_f_nu     = sigma_f_lambda * lambda_pivot^2 / c
        sigma_mag      = 1.0857 * sigma_net / |net|

    NOTE: Drizzling introduces pixel-to-pixel correlations, so the IVM-
    based errors can underestimate the true noise by a factor of ~1.5-2x.
    Compare with the empirical background scatter as a sanity check.

    Parameters
    ----------
    sci : 2D ndarray
        Science image in electrons/s.
    wht : 2D ndarray
        Weight (inverse-variance) map from the DRC extension 2.
    src_x, src_y : float
        Source pixel coordinates.
    pixscale : float
        Pixel scale in arcsec/pixel.
    photflam, photplam, zpt_ab : float
        Flux calibration values (see read_image_and_calibration).
    radii_arcsec : list of float or None
        Aperture radii to measure.  Defaults to a reasonable grid.
    bkg_inner_arcsec, bkg_outer_arcsec : float
        Inner/outer radii of the background annulus.

    Returns
    -------
    measurements : list of dict
        Each dict has keys: radius_arcsec, radius_pix, raw_counts,
        bkg_per_pix, net_counts, sigma_net, f_lambda, f_lambda_err,
        f_nu_uJy, f_nu_uJy_err, mag_ab, mag_err, snr.
    bkg_stats : dict
        Background median, mean, std (all in e/s/pix), and n_pix.
    """
    from photutils.aperture import CircularAperture, CircularAnnulus
    from photutils.aperture import aperture_photometry as ap_phot
    from astropy.stats import sigma_clipped_stats

    if radii_arcsec is None:
        radii_arcsec = [
            0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50,
            0.75, 1.00, 1.25, 1.50, 2.00, 2.50,
        ]

    pos = (src_x, src_y)

    # --- Background estimation ---
    bkg_ann = CircularAnnulus(
        pos,
        r_in=bkg_inner_arcsec / pixscale,
        r_out=bkg_outer_arcsec / pixscale,
    )
    bkg_mask = bkg_ann.to_mask(method="center")
    bkg_data = bkg_mask.multiply(sci)
    bkg_values = bkg_data[bkg_mask.data > 0]
    bkg_mean, bkg_median, bkg_std = sigma_clipped_stats(bkg_values, sigma=3.0)

    # Number of background pixels (for background subtraction uncertainty)
    n_bkg_pix = len(bkg_values)

    # --- Aperture photometry at each radius ---
    measurements = []
    for r_arcsec in radii_arcsec:
        r_pix = r_arcsec / pixscale
        aperture = CircularAperture(pos, r=r_pix)
        phot = ap_phot(sci, aperture)
        raw = float(phot["aperture_sum"][0])

        bkg_total = bkg_median * aperture.area
        net = raw - bkg_total
        n_aper = aperture.area

        # --- Error from IVM weight map ---
        # Variance of raw aperture sum: sum of 1/wht for each pixel
        aper_mask_obj = aperture.to_mask(method="center")
        wht_cutout = aper_mask_obj.multiply(wht)
        aper_pixel_mask = aper_mask_obj.data > 0
        wht_vals = wht_cutout[aper_pixel_mask]

        good_wht = wht_vals > 0
        if np.any(good_wht):
            var_raw = np.sum(1.0 / wht_vals[good_wht])
        else:
            var_raw = np.inf

        # Variance from background subtraction uncertainty:
        # bkg_std/sqrt(N_bkg) is the std error of the background estimate;
        # multiplied by N_aper pixels being subtracted
        var_bkg_sub = n_aper**2 * (bkg_std**2 / n_bkg_pix)

        # Total variance of net count rate
        var_net = var_raw + var_bkg_sub
        sigma_net = np.sqrt(var_net)

        # S/N from the formal error budget
        snr = net / sigma_net if sigma_net > 0 else 0.0

        # --- Propagate to physical flux errors ---
        f_lambda = photflam * net                              # erg/s/cm^2/Angstrom
        f_lambda_err = photflam * sigma_net

        f_nu = f_lambda * (photplam ** 2) / C_ANGSTROM_PER_S  # erg/s/cm^2/Hz
        f_nu_uJy = f_nu * 1e29                                # uJy
        f_nu_uJy_err = f_lambda_err * (photplam ** 2) / C_ANGSTROM_PER_S * 1e29

        mag_ab = (-2.5 * np.log10(net) + zpt_ab) if net > 0 else 99.0
        mag_err = (1.0857 * sigma_net / abs(net)) if net > 0 else 99.0

        measurements.append(dict(
            radius_arcsec=r_arcsec,
            radius_pix=r_pix,
            raw_counts=raw,
            bkg_per_pix=float(bkg_median),
            net_counts=float(net),
            sigma_net=float(sigma_net),
            f_lambda=float(f_lambda),
            f_lambda_err=float(f_lambda_err),
            f_nu_uJy=float(f_nu_uJy),
            f_nu_uJy_err=float(f_nu_uJy_err),
            mag_ab=float(mag_ab),
            mag_err=float(mag_err),
            snr=float(snr),
        ))

    bkg_stats = dict(
        mean=float(bkg_mean),
        median=float(bkg_median),
        std=float(bkg_std),
        n_pix=int(n_bkg_pix),
    )

    return measurements, bkg_stats


# ===================================================================
# 4. HAP catalog cross-match
# ===================================================================

def crossmatch_hap_catalog(catalog_path, ra, dec, match_radius_arcsec=2.0):
    """
    Read a HAP ECSV catalog and return the closest match within
    ``match_radius_arcsec`` of (ra, dec).

    Returns
    -------
    match : dict or None
        All columns for the best match, plus 'separation_arcsec'.
    """
    from astropy.table import Table
    from astropy.coordinates import SkyCoord
    import astropy.units as u

    cat = Table.read(catalog_path, format="ascii.ecsv")
    cat_coords = SkyCoord(ra=cat["RA"], dec=cat["DEC"], unit="deg")
    target = SkyCoord(ra=ra, dec=dec, unit="deg")
    seps = target.separation(cat_coords)

    within = seps < match_radius_arcsec * u.arcsec
    if not np.any(within):
        return None

    best_idx = np.argmin(seps)
    if seps[best_idx] > match_radius_arcsec * u.arcsec:
        return None

    row = cat[best_idx]
    result = {col: _serialize(row[col]) for col in cat.colnames}
    result["separation_arcsec"] = float(seps[best_idx].arcsec)

    return result


def _serialize(val):
    """Convert numpy/astropy types to plain Python for JSON."""
    if hasattr(val, "item"):
        return val.item()
    return val


# ===================================================================
# 5. Plotting
# ===================================================================

def plot_curve_of_growth(results, output_path, reference_fluxes=None):
    """
    Plot the curve of growth for all filters on a single figure.

    Parameters
    ----------
    results : dict
        {filter_name: {'measurements': [...], 'hap_segment': {...}, ...}}
    output_path : str
        Where to save the PNG.
    reference_fluxes : dict or None
        {label: flux_uJy} to draw as horizontal reference lines.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # --- Academic-style rcParams ---
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.linewidth": 1.0,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 5,
        "ytick.major.size": 5,
        "xtick.minor.size": 3,
        "ytick.minor.size": 3,
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        "xtick.top": True,
        "ytick.right": True,
    })

    n_filters = len(results)
    fig, axes = plt.subplots(1, n_filters, figsize=(5.5 * n_filters, 4.5),
                             squeeze=False)

    colors = {"F475W": "tab:blue", "F606W": "tab:red", "F814W": "tab:green",
              "F435W": "tab:purple", "F555W": "tab:orange", "F775W": "tab:gray"}

    for i, (filt, data) in enumerate(sorted(results.items())):
        ax = axes[0, i]
        meas = data["measurements"]
        radii = np.array([m["radius_arcsec"] for m in meas])
        fluxes = np.array([m["f_nu_uJy"] for m in meas])
        flux_errs = np.array([m.get("f_nu_uJy_err", 0.0) for m in meas])
        color = colors.get(filt, "black")

        # Data with error bars
        ax.errorbar(radii, fluxes, yerr=flux_errs, fmt="o-", color=color,
                    lw=1.5, ms=5, capsize=2, capthick=1,
                    label=f"{filt} aperture", zorder=5)

        # HAP segment reference
        hap_seg = data.get("hap_segment")
        if hap_seg and "MagSegment" in hap_seg:
            hap_flux = 3631e6 * 10 ** (-hap_seg["MagSegment"] / 2.5)  # uJy
            ax.axhline(hap_flux, color="0.5", ls="--", lw=1.0,
                       label=f"HAP segment ({hap_flux:.2f} $\\mu$Jy)")

        # External references
        if reference_fluxes:
            ref_colors = ["tab:green", "tab:orange", "tab:purple", "tab:brown"]
            for j, (label, flux) in enumerate(reference_fluxes.items()):
                ax.axhline(flux, color=ref_colors[j % len(ref_colors)],
                           ls=":", lw=1.0, alpha=0.8,
                           label=f"{label} ({flux:.2f} $\\mu$Jy)")

        ax.set_xlabel(r"Aperture radius (arcsec)")
        ax.set_ylabel(r"$f_\nu$ ($\mu$Jy)")
        ax.set_title(filt, fontsize=12)
        ax.legend(fontsize=8, loc="lower right", framealpha=0.9,
                  edgecolor="0.7")
        ax.set_xlim(0, max(radii) * 1.1)
        ax.set_ylim(0, max(fluxes + flux_errs) * 1.15)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"\nCurve-of-growth plot saved to {output_path}")


def plot_cutouts(image_infos, output_path, cutout_radius_arcsec=4.0):
    """
    Plot asinh-stretched cutouts centered on the source with aperture overlays.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from astropy.visualization import AsinhStretch, ImageNormalize

    n = len(image_infos)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5.5), squeeze=False)

    for i, (filt, info) in enumerate(sorted(image_infos.items())):
        ax = axes[0, i]
        sci = info["sci"]
        px, py = info["src_x"], info["src_y"]
        pscale = info["pixscale"]
        r_pix = int(cutout_radius_arcsec / pscale)

        ix, iy = int(round(px)), int(round(py))
        cutout = sci[iy - r_pix:iy + r_pix + 1, ix - r_pix:ix + r_pix + 1]

        norm = ImageNormalize(
            cutout, stretch=AsinhStretch(a=0.01),
            vmin=np.nanpercentile(cutout, 5),
            vmax=np.nanpercentile(cutout, 99.5),
        )
        ax.imshow(cutout, origin="lower", cmap="gray_r", norm=norm)

        cx = r_pix + (px - ix)
        cy = r_pix + (py - iy)
        ax.plot(cx, cy, "+", color="red", ms=15, mew=1.5)

        # Aperture circles
        for rad, col, ls in [(0.25, "cyan", "-"), (1.0, "lime", "-"),
                              (2.0, "yellow", "--")]:
            circ = plt.Circle((cx, cy), rad / pscale, fill=False,
                              color=col, lw=1.5, ls=ls, label=f'{rad}"')
            ax.add_patch(circ)

        size = 2 * cutout_radius_arcsec
        ax.set_title(f'{filt} ({size:.0f}" x {size:.0f}" cutout)', fontsize=13)
        ax.legend(fontsize=9, loc="upper left")

        # Scale bar (1")
        bar_pix = 1.0 / pscale
        ax.plot([10, 10 + bar_pix], [10, 10], "w-", lw=2)
        ax.text(10 + bar_pix / 2, 15, '1"', color="white", ha="center",
                fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Cutout plot saved to {output_path}")


# ===================================================================
# 6. Summary report
# ===================================================================

def write_summary(results, ra, dec, proposal_id, output_path):
    """Write a markdown summary table."""
    lines = [
        "# HST Aperture Photometry - Curve of Growth Analysis\n",
        f"**Target:** RA={ra:.6f}, Dec={dec:.6f}",
        f"**HST Proposal:** {proposal_id}",
        f"**Filters:** {', '.join(sorted(results.keys()))}\n",
    ]

    for filt in sorted(results.keys()):
        data = results[filt]
        lines.append(f"## {filt}\n")
        lines.append(f"PHOTFLAM = {data['photflam']:.6e}, "
                      f"PHOTPLAM = {data['photplam']:.2f}, "
                      f"ZPT_AB = {data['zpt_ab']:.4f}")
        lines.append(f"Background: median = {data['bkg_stats']['median']:.6f}, "
                      f"std = {data['bkg_stats']['std']:.6f} e/s/pix\n")

        lines.append("| Radius (\") | Flux (uJy) | sigma_flux (uJy) | mag_AB | sigma_mag | S/N |")
        lines.append("|------------|-----------|-------------|--------|-------|-----|")
        for m in data["measurements"]:
            lines.append(
                f"| {m['radius_arcsec']:.2f} | {m['f_nu_uJy']:.3f} | "
                f"{m.get('f_nu_uJy_err', 0):.3f} | "
                f"{m['mag_ab']:.2f} | {m.get('mag_err', 0):.3f} | "
                f"{m['snr']:.1f} |"
            )

        # HAP comparison
        hap_seg = data.get("hap_segment")
        hap_pt = data.get("hap_point")
        if hap_seg:
            lines.append(f"\nHAP segment: MagSegment = {hap_seg.get('MagSegment', 'N/A')}, "
                          f"Area = {hap_seg.get('Area', 'N/A')} pix, "
                          f"Elongation = {hap_seg.get('Elongation', 'N/A')}")
        if hap_pt:
            lines.append(f"HAP point: MagAp1 = {hap_pt.get('MagAp1', 'N/A')}, "
                          f"MagAp2 = {hap_pt.get('MagAp2', 'N/A')}, "
                          f"CI = {hap_pt.get('CI', 'N/A')}")
        lines.append("")

    # Plateau comparison
    lines.append("## Plateau Summary\n")
    lines.append("| Filter | Plateau (uJy) | HAP segment (uJy) | Ratio |")
    lines.append("|--------|--------------|-------------------|-------|")
    for filt in sorted(results.keys()):
        meas = results[filt]["measurements"]
        plateau = np.mean([m["f_nu_uJy"] for m in meas[-3:]])
        hap_seg = results[filt].get("hap_segment")
        if hap_seg and "MagSegment" in hap_seg:
            hap_flux = 3631e6 * 10 ** (-hap_seg["MagSegment"] / 2.5)
            ratio = plateau / hap_flux
            lines.append(f"| {filt} | {plateau:.3f} | {hap_flux:.3f} | {ratio:.2f}x |")
        else:
            lines.append(f"| {filt} | {plateau:.3f} | - | - |")

    text = "\n".join(lines)
    with open(output_path, "w") as f:
        f.write(text)
    print(f"\nSummary written to {output_path}")


# ===================================================================
# Main pipeline
# ===================================================================

def run_pipeline(
    ra,
    dec,
    proposal_id=None,
    output_dir="./hst_photometry_output",
    radii_arcsec=None,
    bkg_inner=3.0,
    bkg_outer=4.5,
    reference_fluxes=None,
):
    """
    End-to-end pipeline: query -> download -> measure -> compare -> plot.

    Parameters
    ----------
    ra, dec : float
        Target ICRS coordinates in decimal degrees.
    proposal_id : str
        HST proposal ID.
    output_dir : str
        Directory for all outputs (created if needed).
    radii_arcsec : list of float or None
        Custom aperture radii.
    bkg_inner, bkg_outer : float
        Background annulus radii in arcsec.
    reference_fluxes : dict or None
        {label: flux_in_uJy} for optional horizontal reference lines
        on the curve-of-growth plot.

    Returns
    -------
    results : dict
        Full results dictionary (also saved as JSON).
    """
    os.makedirs(output_dir, exist_ok=True)
    download_dir = os.path.join(output_dir, "downloads")

    # --- Step 1: Query MAST ---
    print("=" * 60)
    print("Step 1: Querying MAST for HAP-SVM observations")
    print("=" * 60)
    obs = query_mast_hap(ra, dec, proposal_id)

    # --- Step 2: Download images ---
    print("\n" + "=" * 60)
    print("Step 2: Downloading drizzled DRC images")
    print("=" * 60)
    image_paths = download_drizzled_images(obs, download_dir)

    # --- Step 3: Download catalogs ---
    print("\n" + "=" * 60)
    print("Step 3: Downloading HAP catalogs")
    print("=" * 60)
    catalog_paths = download_hap_catalogs(obs, download_dir)

    # --- Step 4: Photometry ---
    print("\n" + "=" * 60)
    print("Step 4: Aperture photometry")
    print("=" * 60)

    results = {}
    image_infos = {}

    for filt, fpath in sorted(image_paths.items()):
        print(f"\n--- {filt} ---")
        info = read_image_and_calibration(fpath, ra, dec)
        image_infos[filt] = info

        print(f"  Pixel scale: {info['pixscale']:.4f} arcsec/pix")
        print(f"  Source at pixel ({info['src_x']:.2f}, {info['src_y']:.2f})")
        print(f"  ZPT_AB = {info['zpt_ab']:.4f}")

        measurements, bkg_stats = aperture_photometry_cog(
            info["sci"],
            info["wht"],
            info["src_x"],
            info["src_y"],
            info["pixscale"],
            info["photflam"],
            info["photplam"],
            info["zpt_ab"],
            radii_arcsec=radii_arcsec,
            bkg_inner_arcsec=bkg_inner,
            bkg_outer_arcsec=bkg_outer,
        )

        print(f"  Background: {bkg_stats['median']:.6f} +/- {bkg_stats['std']:.6f} e/s/pix")
        plateau = np.mean([m["f_nu_uJy"] for m in measurements[-3:]])
        print(f"  Plateau flux: {plateau:.3f} uJy (mean of last 3 radii)")

        results[filt] = dict(
            photflam=info["photflam"],
            photplam=info["photplam"],
            zpt_ab=info["zpt_ab"],
            pixscale=info["pixscale"],
            bkg_stats=bkg_stats,
            measurements=measurements,
        )

        # Cross-match HAP catalogs
        for cat_type in ("point", "segment"):
            key = (filt, cat_type)
            if key in catalog_paths:
                match = crossmatch_hap_catalog(catalog_paths[key], ra, dec)
                results[filt][f"hap_{cat_type}"] = match
                if match:
                    sep = match["separation_arcsec"]
                    print(f"  HAP {cat_type} match at {sep:.3f}\"")
                else:
                    print(f"  No HAP {cat_type} match within 2\"")

    # --- Step 5: Save and plot ---
    print("\n" + "=" * 60)
    print("Step 5: Generating outputs")
    print("=" * 60)

    json_path = os.path.join(output_dir, "photometry_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results JSON saved to {json_path}")

    plot_curve_of_growth(
        results,
        os.path.join(output_dir, "curve_of_growth.png"),
        reference_fluxes=reference_fluxes,
    )

    plot_cutouts(
        image_infos,
        os.path.join(output_dir, "source_cutouts.png"),
    )

    write_summary(
        results, ra, dec, proposal_id,
        os.path.join(output_dir, "photometry_summary.md"),
    )

    print("\n" + "=" * 60)
    print("Done!  All outputs in:", output_dir)
    print("=" * 60)

    return results


# ===================================================================
# CLI entry point
# ===================================================================

def main():
    parser = argparse.ArgumentParser(
        description="HST ACS/WFC curve-of-growth aperture photometry",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("ra", type=float, help="RA in decimal degrees (ICRS)")
    parser.add_argument("dec", type=float, help="Dec in decimal degrees (ICRS)")
    parser.add_argument(
        "--proposal-id", "-p", default=None,
        help="HST proposal ID to restrict search (optional; auto-selects if omitted)",
    )
    parser.add_argument(
        "--output-dir", "-o", default="./hst_photometry_output",
        help="Output directory (default: ./hst_photometry_output)",
    )
    parser.add_argument(
        "--bkg-inner", type=float, default=3.0,
        help="Background annulus inner radius in arcsec (default: 3.0)",
    )
    parser.add_argument(
        "--bkg-outer", type=float, default=4.5,
        help="Background annulus outer radius in arcsec (default: 4.5)",
    )
    parser.add_argument(
        "--radii", type=float, nargs="+", default=None,
        help="Custom aperture radii in arcsec (default: 0.1 to 2.5)",
    )
    parser.add_argument(
        "--ref", nargs=2, action="append", metavar=("LABEL", "FLUX_UJY"),
        help="Add a reference flux line: --ref 'Legacy g' 4.212  (repeatable)",
    )

    args = parser.parse_args()

    # Parse reference fluxes
    reference_fluxes = None
    if args.ref:
        reference_fluxes = {label: float(flux) for label, flux in args.ref}

    run_pipeline(
        ra=args.ra,
        dec=args.dec,
        proposal_id=args.proposal_id,
        output_dir=args.output_dir,
        radii_arcsec=args.radii,
        bkg_inner=args.bkg_inner,
        bkg_outer=args.bkg_outer,
        reference_fluxes=reference_fluxes,
    )


if __name__ == "__main__":
    main()
