"""
apply_multi_index_thresholds.py — Table 3 Threshold Labelling (Wang et al. 2022)
=================================================================================

Reference:
    Wang, W., Samat, A., Abuduwaili, J., Ge, Y., De Maeyer, P., & Van de Voorde, T.
    (2022). A novel hybrid sand and dust storm detection method using MODIS data
    on GEE platform. European Journal of Remote Sensing, 55(1), 420–428.
    https://doi.org/10.1080/22797254.2022.2093278

Purpose:
    Implement Table 3 (Wang et al. 2022) — empirical multi-index threshold
    criteria to automatically label pixels into 11 land surface / atmospheric
    classes. These labelled pixels are training samples for the downstream SVM.

Pipeline position (Step 4 of AL-SVM):
    scale_and_mask_MOD09GA() → compute_all_indices() → compute_dust_fraction()
    → compute_edi_alpha() → attach_auxiliary_data() → apply_multi_index_thresholds()

Expected input image bands
--------------------------
    Reflectance:   sur_refl_b01 … sur_refl_b07   (scaled, 0–1)
    Indices:       NDDI, NDVI, NDWI, NDSI, MBWI, MBSCI  (from compute_all_indices)
    Alpha/EDI:     alpha_dust, alpha_cloud, alpha_water, EDI_alpha  (from edi_with_alpha)
    Auxiliary:     DEM, LCT  (from attach_auxiliary_data — defined in this file)

Output
------
    Single-band uint8 image 'class_label', values 0–11:
        0  = Unclassified
        1  = TcC   Thick Cloud
        2  = Cs    Cirrostratus Cloud
        3  = TnC   Thin Cloud
        4  = TcD   Thick Dust
        5  = TnD   Thin Dust
        6  = WD    Water Dust (dust over water)
        7  = Lb    Dry Lakebed
        8  = WB    Water Bodies
        9  = BL    Bare Land
        10 = VL    Vegetated Land
        11 = SI    Snow and Ice

Visualisation palette (CLASS_PALETTE, VIS_CLASS) is exported from this module
so notebook cells can import it directly without re-defining colours.
"""

import ee

# ==============================================================================
# CLASS LABEL CONSTANTS
# ==============================================================================
# Integer codes for 'class_label' band. Consistent across all scripts.

CLASS_UNCLASSIFIED = 0
CLASS_TCC = 1    # Thick Cloud
CLASS_CS  = 2    # Cirrostratus Cloud
CLASS_TNC = 3    # Thin Cloud
CLASS_TCD = 4    # Thick Dust
CLASS_TND = 5    # Thin Dust
CLASS_WD  = 6    # Water Dust
CLASS_LB  = 7    # Dry Lakebed
CLASS_WB  = 8    # Water Bodies
CLASS_BL  = 9    # Bare Land
CLASS_VL  = 10   # Vegetated Land
CLASS_SI  = 11   # Snow and Ice

# ==============================================================================
# VISUALISATION — importable by sds_analysis.ipynb
# ==============================================================================

CLASS_PALETTE = [
    'ffffff',   # 0  Unclassified       — white
    '4575b4',   # 1  TcC  Thick Cloud   — dark blue
    '74add1',   # 2  Cs   Cirrostratus  — mid blue
    'abd9e9',   # 3  TnC  Thin Cloud    — light blue
    'd73027',   # 4  TcD  Thick Dust    — dark red
    'fdae61',   # 5  TnD  Thin Dust     — orange
    'fee090',   # 6  WD   Water Dust    — pale yellow
    'f4a582',   # 7  Lb   Dry Lakebed   — salmon
    '313695',   # 8  WB   Water Bodies  — navy
    'c8a96e',   # 9  BL   Bare Land     — tan
    '1a9641',   # 10 VL   Vegetated     — green
    'e0f3f8',   # 11 SI   Snow/Ice      — pale cyan
]

VIS_CLASS = {'min': 0, 'max': 11, 'palette': CLASS_PALETTE}

# Human-readable labels for add_legend()
CLASS_LEGEND = {
    '0 — Unclassified':       '#ffffff',
    '1 — Thick Cloud':        '#4575b4',
    '2 — Cirrostratus':       '#74add1',
    '3 — Thin Cloud':         '#abd9e9',
    '4 — Thick Dust':         '#d73027',
    '5 — Thin Dust':          '#fdae61',
    '6 — Water Dust':         '#fee090',
    '7 — Dry Lakebed':        '#f4a582',
    '8 — Water Bodies':       '#313695',
    '9 — Bare Land':          '#c8a96e',
    '10 — Vegetated Land':    '#1a9641',
    '11 — Snow and Ice':      '#e0f3f8',
}

# ==============================================================================
# AUXILIARY DATASETS — loaded once at module import
# ==============================================================================

# SRTM DEM: required for SI class (elevation > 2500 m).
# Dubai max elevation ≈ 300 m — SI pixels will not occur unless ROI is expanded
# to include the Hajar Mountains (Oman/UAE, up to ~3000 m).
srtm_band = (
    ee.Image('USGS/SRTMGL1_003')
    .select('elevation')
    .reproject(crs='EPSG:4326', scale=500)
    # .clip(roi)
    .rename('DEM')
)

# MCD12Q1 land cover — IGBP Type 1. Water body = class 17.
# Year 2020 used as default; adjust if analysing a different year.
# Used for WD class (dust over water bodies).
lct_band = (
    ee.ImageCollection('MODIS/061/MCD12Q1')
    .filter(ee.Filter.calendarRange(2020, 2020, 'year'))
    .first()
    .select('LC_Type1')
    .reproject(crs='EPSG:4326', scale=500)
    # .clip(roi)
    .rename('LCT')
)

IGBP_WATER = 17   # IGBP water body class code


# ==============================================================================
# FUNCTION 1 — ATTACH AUXILIARY DATA
# ==============================================================================

def attach_auxiliary_data(image, lct_year=2020):
    """
    Attach SRTM DEM and MCD12Q1 land cover type as bands to the image.

    Both datasets are reprojected to match the MOD09GA 500 m pixel grid.
    DEM uses bilinear resampling (continuous elevation values).
    LCT uses nearest-neighbour (preserves integer class codes).

    These auxiliary bands are required by three Table 3 criteria:
        SI  → DEM > 2500 m   (separates high-altitude snow from thick cloud)
        WD  → LCT == 17      (dust over IGBP water body land cover)

    Parameters
    ----------
    image : ee.Image
        Image with reflectance and index bands already attached.
    lct_year : int
        Year of MCD12Q1 land cover to use. Defaults to 2020.

    Returns
    -------
    ee.Image
        Input image with 'DEM' and 'LCT' bands added.
    """

# DEM — reproject only, no .resample() call needed
    dem_500m = (
        srtm_band
        .reproject(crs=image.projection(), scale=500)
        .rename('DEM')
    )

    # LCT — reproject only, GEE uses nearest neighbour by default
    # for integer/categorical images when no resample is specified
    lct_500m = (
        lct_band
        .reproject(crs=image.projection(), scale=500)
        .rename('LCT')
    )

    return image.addBands([dem_500m, lct_500m])


# ==============================================================================
# FUNCTION 2 — APPLY TABLE 3 MULTI-INDEX THRESHOLDS
# ==============================================================================

def apply_multi_index_thresholds(image):
    """
    Apply the empirical multi-index threshold criteria from Table 3
    (Wang et al. 2022) to assign a class label to every pixel.

    Logic:
        ALL stated criteria for a class must be satisfied simultaneously
        (AND logic). Priority order is enforced by applying class masks
        from lowest to highest priority — the last .where() wins.
        Unclassified pixels (no criteria met) receive label 0.

    Priority order (high → low):
        SI > TcC > Cs > TnC > WB > TcD > TnD > WD > Lb > BL > VL

    Parameters
    ----------
    image : ee.Image
        Image with bands: sur_refl_b01, sur_refl_b07, NDDI, NDVI, NDWI,
        NDSI, MBWI, MBSCI, alpha_dust, alpha_cloud, alpha_water, EDI_alpha,
        DEM, LCT.

    Returns
    -------
    ee.Image
        Single-band 'class_label' image (uint8, values 0–11).
    """

    # ── Pull all required bands ───────────────────────────────────────────────
    B1    = image.select('sur_refl_b01')
    B7    = image.select('sur_refl_b07')
    EDI   = image.select('EDI_alpha')
    NDDI  = image.select('NDDI')
    NDVI  = image.select('NDVI')
    NDWI  = image.select('NDWI')
    NDSI  = image.select('NDSI')
    MBWI  = image.select('MBWI')
    MBSCI = image.select('MBSCI')
    a_dust  = image.select('alpha_dust')
    a_cloud = image.select('alpha_cloud')
    a_water = image.select('alpha_water')
    DEM   = image.select('DEM')
    LCT   = image.select('LCT')

    # ── Helper: min ≤ band ≤ max ──────────────────────────────────────────────
    def in_range(band, lo, hi):
        return band.gte(lo).And(band.lte(hi))

    # ==========================================================================
    # SI — Snow and Ice
    # Key separators: NDSI > 0.6 (snow bright in Green, dark in SWIR-2),
    #   MBSCI > 10 (cloud-like), DEM > 2500 m (not a cloud floating above Dubai).
    # In Dubai: essentially never triggered (max elevation ~300 m).
    # ==========================================================================
    si_mask = (
        in_range(EDI,  -1.2, -0.8)
        .And(in_range(NDDI, -1.0, -0.8))
        .And(in_range(NDSI,  0.6,  1.0))
        .And(in_range(NDVI, -0.2,  0.1))
        .And(in_range(NDWI, -0.1,  0.2))
        .And(in_range(MBWI, -0.3,  1.0))
        .And(MBSCI.gt(10))
        .And(B7.gt(0.05))
        .And(a_cloud.lt(0.2))
        .And(DEM.gt(2500))
    )

    # ==========================================================================
    # TcC — Thick Cloud
    # Key separator: B1 (Red) > 0.9 — optically thick cloud reflects uniformly
    # bright across all bands. Almost nothing else reaches this reflectance.
    # ==========================================================================
    tcc_mask = (
        in_range(EDI,  -1.2, -0.8)
        .And(in_range(NDDI, -0.9, -0.6))
        .And(in_range(NDSI,  0.4,  0.8))
        .And(in_range(NDVI, -0.1,  0.1))
        .And(in_range(NDWI, -0.1,  0.1))
        .And(in_range(MBWI, -0.6, -0.2))
        .And(B1.gt(0.9))
    )

    # ==========================================================================
    # Cs — Cirrostratus
    # High-altitude semi-transparent ice cloud. B7 > 0.25 because the surface
    # below still contributes SWIR through the thin cloud layer.
    # ==========================================================================
    cs_mask = (
        in_range(NDDI, -0.6, -0.3)
        .And(in_range(NDVI, -0.1,  0.1))
        .And(in_range(NDWI, -0.1,  0.1))
        .And(in_range(MBSCI, 1.5,  5.0))
        .And(B7.gt(0.25))
    )

    # ==========================================================================
    # TnC — Thin Cloud
    # Hardest class — overlaps with thin dust. Key separators:
    # negative NDDI (cloud Blue is high), alpha_cloud > 0 (LSU detects cloud),
    # alpha_dust < 0.8 (not dust-dominated).
    # ==========================================================================
    tnc_mask = (
        in_range(NDDI, -0.7,  0.1)
        .And(in_range(NDSI, -0.15,  0.0))
        .And(in_range(NDVI,  0.0,   0.1))
        .And(in_range(NDWI, -0.15,  0.0))
        .And(B7.lt(0.4))
        .And(a_cloud.gt(0))
        .And(a_dust.lt(0.8))
    )

    # ==========================================================================
    # WB — Water Bodies
    # alpha_water > 0.85 (LSU water fraction) is the primary discriminator.
    # Negative NDDI because water absorbs SWIR strongly.
    # In Dubai: Arabian Gulf coast, Dubai Creek, reservoirs.
    # ==========================================================================
    wb_mask = (
        in_range(NDDI, -1.2, -0.9)
        .And(in_range(NDWI, -1.1, -0.1))
        .And(in_range(NDVI, -0.1,  0.4))
        .And(in_range(MBWI, -0.1,  0.4))
        .And(a_water.gt(0.85))
    )

    # ==========================================================================
    # TcD — Thick Dust
    # Dense dust plume core. MBWI strongly negative (−1.5 to −1.0) because
    # dust has high SWIR (B6, B7) and low Green (B4). EDI elevated by high α.
    # ==========================================================================
    tcd_mask = (
        in_range(EDI,  -0.2,  1.0)
        .And(in_range(NDDI,  0.0,  0.15))
        .And(in_range(NDSI, -0.4,  0.0))
        .And(in_range(NDVI, -0.2,  0.2))
        .And(in_range(NDWI, -0.2,  0.1))
        .And(in_range(MBWI, -1.5, -1.0))
    )

    # ==========================================================================
    # TnD — Thin Dust
    # Weaker dust signal. B1 < 0.55 separates from thick cloud (B1 > 0.9).
    # Most frequently confused class in Dubai — overlaps with BL and TnC.
    # ==========================================================================
    tnd_mask = (
        in_range(EDI,  -0.1,  0.8)
        .And(in_range(NDDI, -0.2,  0.3))
        .And(in_range(NDSI, -0.3,  0.1))
        .And(in_range(NDVI, -0.1,  0.2))
        .And(in_range(NDWI, -0.2,  0.0))
        .And(in_range(MBWI, -1.3, -0.8))
        .And(B1.lt(0.55))
    )

    # ==========================================================================
    # WD — Water Dust (dust plume over water body)
    # EDI positive (dust signal) but LCT == 17 (underlying surface is water).
    # Key occurrence: Dubai coastline during Gulf SDS events.
    # ==========================================================================
    wd_mask = (
        in_range(EDI,  -0.6,  0.4)
        .And(in_range(NDDI, -0.4, -0.2))
        .And(in_range(NDSI,  0.0,  0.2))
        .And(in_range(NDVI, -0.1,  0.1))
        .And(in_range(NDWI,  0.0,  0.2))
        .And(in_range(MBWI, -0.6, -0.2))
        .And(LCT.eq(IGBP_WATER))
    )

    # ==========================================================================
    # Lb — Dry Lakebed / Sabkha
    # Bright salt crust: moderate reflectance in both visible and SWIR.
    # Dubai analogue: sabkha (salt flats) in Abu Dhabi region.
    # ==========================================================================
    lb_mask = (
        in_range(EDI,  -0.2,  0.1)
        .And(in_range(NDDI,  0.0,  0.1))
        .And(in_range(NDSI, -0.3, -0.1))
        .And(in_range(NDVI,  0.0,  0.1))
        .And(in_range(NDWI, -0.1,  0.0))
        .And(in_range(MBWI, -0.8, -0.3))
    )

    # ==========================================================================
    # BL — Bare Land
    # Desert sand / bare soil. EDI upper bound −0.2 is the key separator
    # from thin dust: bare land has low α from LSU → EDI suppressed below dust.
    # Most common land cover type in Dubai's desert outskirts.
    # ==========================================================================
    bl_mask = (
        in_range(EDI,  -1.2, -0.2)
        .And(in_range(NDDI,  0.0,  0.5))
        .And(in_range(NDSI, -0.6, -0.1))
        .And(in_range(NDVI,  0.0,  0.2))
        .And(in_range(NDWI, -0.4,  0.0))
        .And(in_range(MBWI, -1.5, -0.8))
    )

    # ==========================================================================
    # VL — Vegetated Land
    # NDVI > 0.05 generous lower bound — Dubai vegetation is sparse and often
    # mixed with sand (golf courses, palm plantations, irrigated parks).
    # ==========================================================================
    vl_mask = (
        in_range(EDI,   0.0,  0.6)
        .And(in_range(NDDI, -0.7, -0.3))
        .And(in_range(NDVI,  0.05, 0.9))
        .And(in_range(NDWI, -0.8,  0.0))
        .And(in_range(MBWI, -0.8, -0.4))
    )

    # ==========================================================================
    # ASSIGN LABELS — lowest priority applied first, highest last (.where wins)
    # Final priority (highest last): VL → BL → Lb → WD → WB → TnD → TcD →
    #                                TnC → Cs → TcC → SI
    # ==========================================================================
    label = ee.Image.constant(CLASS_UNCLASSIFIED).rename('class_label')

    label = label.where(vl_mask,  CLASS_VL)
    label = label.where(bl_mask,  CLASS_BL)
    label = label.where(lb_mask,  CLASS_LB)
    label = label.where(wd_mask,  CLASS_WD)
    label = label.where(wb_mask,  CLASS_WB)
    label = label.where(tnd_mask, CLASS_TND)
    label = label.where(tcd_mask, CLASS_TCD)
    label = label.where(tnc_mask, CLASS_TNC)
    label = label.where(cs_mask,  CLASS_CS)
    label = label.where(tcc_mask, CLASS_TCC)
    label = label.where(si_mask,  CLASS_SI)

    return ee.Image(
        label
        .toUint8()
        .copyProperties(image, ['system:time_start', 'system:index'])
    )
