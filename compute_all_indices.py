"""
compute_all_indices.py — Spectral Index Computation for SDS Analysis
=====================================================================

Reference:
    Wang, W., Samat, A., Abuduwaili, J., Ge, Y., De Maeyer, P., & Van de Voorde, T.
    (2022). A novel hybrid sand and dust storm detection method using MODIS data
    on GEE platform. European Journal of Remote Sensing, 55(1), 420–428.
    https://doi.org/10.1080/22797254.2022.2093278

    Index formulas from Table 2 of Wang et al. (2022).

Purpose:
    Compute the six supporting spectral indices (Table 2) needed for the
    multi-index threshold labelling pipeline (Step 2 of AL-SVM).
    EDI-α is computed separately in edi_with_alpha.py.

Pipeline position:
    scale_and_mask_MOD09GA()   → (this script) compute_all_indices()
    → compute_dust_fraction()  → compute_edi_alpha()
    → attach_auxiliary_data()  → apply_multi_index_thresholds()

Input image convention (matches sds_analysis.ipynb):
    Expects MOD09GA images with band names 'sur_refl_b01' … 'sur_refl_b07',
    already scaled to reflectance (multiplied by 0.0001, range ≈ 0–1).

Indices computed
----------------
    NDDI   (B7 − B3) / (B7 + B3)             Qu et al. 2006
    NDVI   (B2 − B1) / (B2 + B1)             Tucker 1979
    NDWI   (B4 − B2) / (B4 + B2)             McFeeters 2007
    NDSI   (B4 − B6) / (B4 + B6)             Salomonson & Appel 2004
    MBWI   2·B4 − B1 − B2 − B6 − B7          X.X. Wang et al. 2018
    MBSCI  (B4 + B1) / (B6 + B7)             Wang et al. 2022

MODIS band mapping (MOD09GA v061, 500m, scaled 0–1):
    sur_refl_b01  B1  Red      0.620–0.670 μm
    sur_refl_b02  B2  NIR      0.841–0.876 μm
    sur_refl_b03  B3  Blue     0.459–0.479 μm
    sur_refl_b04  B4  Green    0.545–0.565 μm
    sur_refl_b05  B5  SWIR-1   1.230–1.250 μm  (not used in any index below)
    sur_refl_b06  B6  SWIR-2   1.628–1.652 μm
    sur_refl_b07  B7  SWIR-3   2.105–2.155 μm
"""

import ee


def compute_all_indices(image):
    """
    Compute the six SDS-relevant spectral indices (Table 2, Wang et al. 2022)
    and attach them as new bands to the image.

    EDI-α is NOT computed here — it requires Linear Spectral Unmixing (LSU)
    which is handled separately in edi_with_alpha.compute_edi_alpha().

    Parameters
    ----------
    image : ee.Image
        MOD09GA image with scaled reflectance bands 'sur_refl_b01'…'sur_refl_b07'.

    Returns
    -------
    ee.Image
        Input image with six additional bands: NDDI, NDVI, NDWI, NDSI, MBWI, MBSCI.
    """

    B1 = image.select('sur_refl_b01')   # Red
    B2 = image.select('sur_refl_b02')   # NIR
    B3 = image.select('sur_refl_b03')   # Blue
    B4 = image.select('sur_refl_b04')   # Green
    B6 = image.select('sur_refl_b06')   # SWIR-2
    B7 = image.select('sur_refl_b07')   # SWIR-3

    # Small epsilon prevents division by zero at masked/noisy edges
    eps = ee.Image(1e-9)

    # ------------------------------------------------------------------
    # NDDI — Normalized Difference Dust Index   [Qu et al. 2006]
    #   (B7 − B3) / (B7 + B3)
    #
    # Uses SWIR-3 and Blue. Dust has elevated SWIR-3 relative to Blue
    # due to aerosol scattering, giving positive NDDI over active dust.
    # Cloud has both B7 and B3 high, suppressing the ratio → negative NDDI.
    # Range: −1 to +1. Key primary discriminator in Table 3 for all 11 classes.
    # ------------------------------------------------------------------
    nddi = (
        B7.subtract(B3)
          .divide(B7.add(B3).add(eps))
          .rename('NDDI')
    )

    # ------------------------------------------------------------------
    # NDVI — Normalized Difference Vegetation Index   [Tucker 1979]
    #   (B2 − B1) / (B2 + B1)
    #
    # Standard vegetation greenness index. High for dense canopy (> 0.5),
    # near zero for bare soil and dust, negative for water.
    # Used in Table 3 to exclude vegetated pixels from dust/cloud classes.
    # ------------------------------------------------------------------
    ndvi = (
        B2.subtract(B1)
          .divide(B2.add(B1).add(eps))
          .rename('NDVI')
    )

    # ------------------------------------------------------------------
    # NDWI — Normalized Difference Water Index   [McFeeters 2007]
    #   (B4 − B2) / (B4 + B2)   uses Green (B4) and NIR (B2)
    #
    # NOTE: Uses NIR (B2), NOT SWIR. Do not confuse with NDSI (which uses B6).
    # Open water has high Green but absorbs in NIR → positive NDWI.
    # Dust and soil have higher NIR than Green → negative NDWI.
    # Used in Table 3 for WB, WD, and SI classes.
    # ------------------------------------------------------------------
    ndwi = (
        B4.subtract(B2)
          .divide(B4.add(B2).add(eps))
          .rename('NDWI')
    )

    # ------------------------------------------------------------------
    # NDSI — Normalized Difference Snow Index   [Salomonson & Appel 2004]
    #   (B4 − B6) / (B4 + B6)   uses Green (B4) and SWIR-2 (B6)
    #
    # NOTE: Uses SWIR-2 (B6), NOT NIR. Different denominator from NDWI.
    # Snow is bright in Green but strongly absorbs in SWIR-2 → high NDSI.
    # Desert sand has similar SWIR-2 and Green → NDSI near zero or negative.
    # Used in Table 3 for SI, TcC, TnC, and BL classes.
    # ------------------------------------------------------------------
    ndsi = (
        B4.subtract(B6)
          .divide(B4.add(B6).add(eps))
          .rename('NDSI')
    )

    # ------------------------------------------------------------------
    # MBWI — Multi-Band Water Index   [X.X. Wang et al. 2018]
    #   2·B4 − B1 − B2 − B6 − B7
    #
    # Linear combination designed for water body extraction.
    # Water: high Green (B4), very low NIR/SWIR → large positive MBWI.
    # Bare land, dust, vegetation: lower Green + moderate/high SWIR → negative.
    # Used in Table 3 for 8 out of 11 classes as a key non-water separator.
    # Range is unbounded; typical values −2 to +1.
    # ------------------------------------------------------------------
    mbwi = (
        B4.multiply(2)
          .subtract(B1)
          .subtract(B2)
          .subtract(B6)
          .subtract(B7)
          .rename('MBWI')
    )

    # ------------------------------------------------------------------
    # MBSCI — Modified Bare Soil / Cloud Index   [Wang et al. 2022]
    #   (B4 + B1) / (B6 + B7)   uses (Green + Red) / (SWIR-2 + SWIR-3)
    #
    # IMPORTANT: denominator is B6+B7 (not 2·B3·B6 — earlier formulation error).
    #
    # Clouds and snow: very bright in visible (B1, B4), weak in SWIR → high MBSCI.
    # Dust: moderate visible + moderate SWIR → lower MBSCI than cloud.
    # Bare desert: moderate visible + strong SWIR → MBSCI < 1.
    #
    # Table 3 cloud thresholds:
    #   MBSCI > 10           → Snow or Thick Cloud
    #   1 < MBSCI ≤ 10       → Cirrostratus / thin cloud range
    #   MBSCI ≤ 1            → Non-cloud surface (dust, soil, veg, water)
    # ------------------------------------------------------------------
    mbsci = (
        B4.add(B1)
          .divide(B6.add(B7).add(eps))
          .rename('MBSCI')
    )

    return image.addBands([nddi, ndvi, ndwi, ndsi, mbwi, mbsci])
