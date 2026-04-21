"""
edi_with_alpha.py — Enhanced Dust Index with Linear Spectral Unmixing (EDI-α)
==============================================================================

Reference:
    Wang, W., Samat, A., Abuduwaili, J., Ge, Y., De Maeyer, P., & Van de Voorde, T.
    (2022). A novel hybrid sand and dust storm detection method using MODIS data
    on GEE platform. European Journal of Remote Sensing, 55(1), 420–428.
    https://doi.org/10.1080/22797254.2022.2093278

    EDI originally proposed by:
    Han, L., Tsunekawa, A., Tsubo, M., & Zhou, W. (2013). An enhanced dust index
    for Asian dust detection with MODIS images. International Journal of Remote
    Sensing, 34(19), 6484–6495.

How this differs from edi.py
-----------------------------
edi.py uses a simple two-index difference (NDDi − NDVI) as a proxy for dust
detection. That approach is fast but treats all pixels equally, regardless of
how "dusty" a pixel actually is — it can still trigger over bright bare soil
that mimics the dust spectral shape.

This module implements the *proper* EDI formulation from Han et al. (2013),
as used in the Wang et al. (2022) pipeline. The key addition is α (alpha):
the dust endmember fraction derived from Linear Spectral Unmixing (LSU).

α explicitly encodes "what fraction of this pixel is pure dust?", ranging
from 0 (no dust) to 1 (entirely dust). Multiplying the SWIR term by α and
the Blue term by (1−α) creates an asymmetric weighting that:
  - Amplifies dust signals in pixels that are mostly dust (α → 1)
  - Suppresses false positives over bright bare soil (low α despite bright SWIR)
  - Naturally handles mixed pixels at dust plume edges

EDI formula (Han et al., 2013 / Wang et al., 2022):

    EDI = [ α · ρ_2.13  −  (1−α) · ρ_0.469 ]
          ─────────────────────────────────────
          [ α · ρ_2.13  +  (1−α) · ρ_0.469 ]

Where:
    ρ_0.469  = surface reflectance at 0.469 μm = MODIS Band 3 (Blue)   → sur_refl_b03
    ρ_2.13   = surface reflectance at 2.13  μm = MODIS Band 7 (SWIR-3) → sur_refl_b07
    α        = dust endmember fraction [0, 1] from Linear Spectral Unmixing

Input image convention (matches sds_analysis.ipynb):
    Expects images with band names 'sur_refl_b01' … 'sur_refl_b07',
    already scaled to reflectance (multiplied by 0.0001, range ≈ 0–1).

Usage in notebook:
    from edi_with_alpha import compute_dust_fraction, compute_edi_alpha, classify_edi_alpha
    img_alpha = compute_dust_fraction(img)        # adds 'alpha_dust' band
    img_edi   = compute_edi_alpha(img_alpha)      # adds 'EDI_alpha' band
    img_class = classify_edi_alpha(img_edi.select('EDI_alpha'))  # 'EDI_alpha_class'
"""

import ee


# ==============================================================================
# ENDMEMBER SPECTRA FOR LINEAR SPECTRAL UNMIXING
# ==============================================================================
#
# These are the reflectance "fingerprints" of three pure surface types in the
# Dubai region, across all 7 MODIS bands (B1–B7, scaled 0–1).
#
# The LSU solver decomposes each pixel into a weighted mix of these three types.
# We extract only the dust weight (α) for the EDI calculation.
#
# Important calibration note:
#   These values are approximated from Han et al. (2013) and the MODIS spectral
#   library for arid Gulf environments. For best results, extract actual
#   endmember spectra from MOD09GA imagery over Dubai during:
#     - Dust:      a confirmed SDS event (e.g., April 2022 Gulf storm)
#     - Bare soil: a clear day over the Al Qudra desert area (south Dubai)
#     - Cloud:     optically thick cloud pixels over the Arabian Sea
#
# Endmember order here: [B1, B2, B3, B4, B5, B6, B7]
# (matches the band selection order used in compute_dust_fraction)

DUST_ENDMEMBER = [
    0.35,   # B1 Red    — moderately bright; dust aerosol scatters red light
    0.38,   # B2 NIR    — slightly elevated NIR under thick dust pall
    0.30,   # B3 Blue   — elevated blue (atmospheric scattering from dust)
    0.36,   # B4 Green
    0.42,   # B5 SWIR-1 — SWIR elevated as dust absorbs/scatters mid-infrared
    0.40,   # B6 SWIR-2
    0.38,   # B7 SWIR-3 — SWIR-3 higher than Blue; this contrast defines the index
]

BARE_SOIL_ENDMEMBER = [
    0.28,   # B1 Red    — sandy desert surface is bright but less than dust
    0.32,   # B2 NIR    — bare soil: NIR slightly > Red
    0.18,   # B3 Blue   — soil: low blue, key distinction from dust
    0.26,   # B4 Green
    0.38,   # B5 SWIR-1 — desert sand: strong SWIR signature
    0.42,   # B6 SWIR-2
    0.44,   # B7 SWIR-3 — bare soil SWIR-3 is high but Blue stays low
]

CLOUD_ENDMEMBER = [
    0.80,   # B1 Red    — thick cloud: uniformly high reflectance across visible
    0.82,   # B2 NIR
    0.82,   # B3 Blue
    0.81,   # B4 Green
    0.70,   # B5 SWIR-1 — clouds begin to drop off in SWIR bands
    0.55,   # B6 SWIR-2
    0.45,   # B7 SWIR-3 — cloud SWIR-3 is notably lower than visible; separates
             #             cloud from dust (dust has SWIR > Blue, cloud has both high)
]

# Water endmember — added as 4th endmember so alpha_water is available for the
# WB (Water Bodies) threshold in apply_multi_index_thresholds.py.
# Water strongly absorbs NIR and SWIR, giving near-zero values in B2–B7.
# B4 (Green) is slightly elevated due to backscatter from optically deep water.
WATER_ENDMEMBER = [
    0.04,   # B1 Red    — water: very low red reflectance
    0.02,   # B2 NIR    — water: strong NIR absorption
    0.06,   # B3 Blue   — water: slightly higher blue (backscatter)
    0.05,   # B4 Green  — water: slight green peak for clear water
    0.01,   # B5 SWIR-1 — water: near-zero SWIR (absorbed)
    0.01,   # B6 SWIR-2
    0.01,   # B7 SWIR-3 — water: essentially zero beyond 1μm
]

# Endmember matrix — order determines output band indices:
#   band 0 = alpha_dust, band 1 = alpha_soil,
#   band 2 = alpha_cloud, band 3 = alpha_water
ENDMEMBER_MATRIX = [
    DUST_ENDMEMBER,
    BARE_SOIL_ENDMEMBER,
    CLOUD_ENDMEMBER,
    WATER_ENDMEMBER,
]


# ==============================================================================
# FUNCTION 1: LINEAR SPECTRAL UNMIXING → DUST FRACTION α
# ==============================================================================

def compute_dust_fraction(image):
    """
    Estimate the dust endmember fraction (α) for each pixel via Linear
    Spectral Unmixing (LSU) using GEE's ee.Image.unmix().

    Theory — LSU models each pixel as:
        ρ_pixel(λ) = α_dust · ρ_dust(λ)
                   + α_soil · ρ_soil(λ)
                   + α_cloud · ρ_cloud(λ)
                   + ε

    subject to:
        α_dust + α_soil + α_cloud = 1   (fractions sum to one)
        α_i ≥ 0                         (no negative abundances)

    GEE solves this as constrained least-squares; sumToOne and nonNegative
    flags enforce the physical constraints.

    The output 'alpha_dust' band is the per-pixel dust abundance (0 to 1):
        α ≈ 0  → pixel is predominantly bare soil or cloud (no dust)
        α ≈ 1  → pixel is predominantly pure dust

    Parameters
    ----------
    image : ee.Image
        MOD09GA image with scaled bands 'sur_refl_b01'–'sur_refl_b07'.
        Bands must be in reflectance units (0.0001 scale already applied).

    Returns
    -------
    ee.Image
        Input image with an additional 'alpha_dust' band (float, 0–1).
    """

    # Select the 7 reflectance bands in the order expected by ENDMEMBER_MATRIX.
    # The unmix function expects bands in the same spectral order as endmembers.
    bands = image.select([
        'sur_refl_b01',   # B1 Red
        'sur_refl_b02',   # B2 NIR
        'sur_refl_b03',   # B3 Blue
        'sur_refl_b04',   # B4 Green
        'sur_refl_b05',   # B5 SWIR-1
        'sur_refl_b06',   # B6 SWIR-2
        'sur_refl_b07',   # B7 SWIR-3
    ])

    # Run constrained least-squares spectral unmixing.
    # sumToOne=True  → sum of all fractions = 1 (conservation of matter)
    # nonNegative=True → no fraction can be negative (physical constraint)
    # Output: 3-band image, one band per endmember, in ENDMEMBER_MATRIX order
    unmixed = bands.unmix(
        endmembers=ENDMEMBER_MATRIX,
        sumToOne=True,
        nonNegative=True
    )

    # Rename output bands to match their endmember identity.
    # band_0 = dust, band_1 = soil, band_2 = cloud, band_3 = water
    alpha_bands = unmixed.rename(
        ['alpha_dust', 'alpha_soil', 'alpha_cloud', 'alpha_water']
    )

    # Attach dust, cloud, and water fractions to the image.
    # alpha_cloud and alpha_water are required by apply_multi_index_thresholds
    # for the TnC and WB class criteria respectively.
    # alpha_soil is an intermediate product and is not added to keep bands lean.
    return image.addBands(
        alpha_bands.select(['alpha_dust', 'alpha_cloud', 'alpha_water'])
    )


# ==============================================================================
# FUNCTION 2: COMPUTE α-WEIGHTED EDI
# ==============================================================================

def compute_edi_alpha(image):
    """
    Compute the Enhanced Dust Index (EDI) using the dust endmember fraction α.

    Formula (Han et al., 2013):

        EDI = [ α · ρ_2.13  −  (1−α) · ρ_0.469 ]
              ─────────────────────────────────────
              [ α · ρ_2.13  +  (1−α) · ρ_0.469 ]

    Step-by-step derivation:
        1.  α · ρ_2.13          →  scale SWIR-3 by the dust fraction
                                    (how much SWIR comes from pure dust)
        2.  (1−α) · ρ_0.469    →  scale Blue by the background fraction
                                    (how much Blue comes from non-dust background)
        3.  numerator   = (1) − (2)
        4.  denominator = (1) + (2)
        5.  EDI = numerator / denominator  (normalized to range −1 … +1)

    Why this works physically:
        - Pure dust pixel (α → 1):
            SWIR term is large (α·B7 ≈ B7), Blue term → 0 ((1−α)·B3 ≈ 0)
            EDI → B7/B7 = +1  (maximum, confirmed dust)

        - Pure non-dust background (α → 0):
            SWIR term → 0, Blue term is large ((1−α)·B3 ≈ B3)
            EDI → −B3/B3 = −1  (minimum, no dust)

        - Mixed pixel at plume edge (α ≈ 0.5):
            Both terms contribute; EDI ≈ (0.5·B7 − 0.5·B3) / (0.5·B7 + 0.5·B3)
            which reduces to the simple NDDi for a 50/50 mix.

    Critical advantage over simple NDDi:
        Bright bare desert soils can have moderately high B7/B3 ratios even
        without dust. Over desert, α_dust (from LSU) will be low because the
        spectral shape matches the bare_soil endmember, not the dust endmember.
        This low α suppresses the SWIR term, preventing false positives.

    Parameters
    ----------
    image : ee.Image
        Image with scaled reflectance bands 'sur_refl_b03', 'sur_refl_b07'
        and an 'alpha_dust' band (output of compute_dust_fraction).

    Returns
    -------
    ee.Image
        Input image with an additional 'EDI_alpha' band (float, −1 to +1).
    """

    B3    = image.select('sur_refl_b03')   # Blue, ρ_0.469 μm
    B7    = image.select('sur_refl_b07')   # SWIR-3, ρ_2.13 μm
    alpha = image.select('alpha_dust')     # dust endmember fraction [0, 1]

    # Background (non-dust) fraction: everything that is not pure dust
    one_minus_alpha = ee.Image(1).subtract(alpha)

    # Weighted SWIR term: contribution of the dust fraction to SWIR reflectance
    alpha_swir = alpha.multiply(B7)

    # Weighted Blue term: contribution of the background fraction to Blue reflectance
    bg_blue = one_minus_alpha.multiply(B3)

    # Normalized difference between the two weighted terms
    numerator   = alpha_swir.subtract(bg_blue)
    denominator = alpha_swir.add(bg_blue)

    # Guard against division by zero at masked/noisy edges.
    # epsilon is negligibly small relative to typical reflectance values.
    epsilon = ee.Image(1e-6)
    edi = numerator.divide(denominator.add(epsilon))

    return image.addBands(edi.rename('EDI_alpha'))


# ==============================================================================
# FUNCTION 3: CLASSIFY EDI INTO DUST LIKELIHOOD CLASSES
# ==============================================================================

def classify_edi_alpha(edi_band):
    """
    Convert the continuous EDI_alpha band into a 3-class dust likelihood map.

    Thresholds are adapted from Wang et al. (2022) and Han et al. (2013).
    These should be calibrated against local visibility or AOD observations.

    Classes
    -------
    0  →  Non-dust          EDI < 0.05   (background, vegetation, water)
    1  →  Moderate dust     0.05 ≤ EDI < 0.20
    2  →  High dust         EDI ≥ 0.20   (active SDS event)

    Parameters
    ----------
    edi_band : ee.Image
        Single-band image named 'EDI_alpha', output of compute_edi_alpha.

    Returns
    -------
    ee.Image
        Single-band 'EDI_alpha_class' image with integer values 0, 1, or 2.
    """

    # Build class layers then take the maximum to get the highest applicable class
    high     = edi_band.gte(0.20).multiply(2)          # 2 where EDI ≥ 0.20
    moderate = edi_band.gte(0.05).And(                 # 1 where 0.05 ≤ EDI < 0.20
               edi_band.lt(0.20)).multiply(1)

    edi_class = high.max(moderate).rename('EDI_alpha_class')
    return edi_class
