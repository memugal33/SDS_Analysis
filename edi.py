"""
edi.py — Enhanced Dust Index (EDI) computation for MODIS MOD09GA
================================================================

Methodology reference:
  Al-Hamdan et al. (2022). "Sand and dust storm monitoring and assessment using
  remote sensing and GIS." European Journal of Remote Sensing, 55(1).
  https://www.tandfonline.com/doi/full/10.1080/22797254.2022.2093278

Background
----------
Sand and Dust Storms (SDS) pose severe hazards across arid and semi-arid
regions.  Remote sensing with MODIS provides daily, near-global coverage at
500 m resolution — ideal for tracking fast-moving dust events.

The Enhanced Dust Index (EDI) is designed to isolate pixels affected by
airborne sand/dust while simultaneously suppressing confusion with:
  - Bright bare soil / sabkha (salt flats) — which are spectrally similar
    to dust in visible bands alone
  - Water bodies — which can have low reflectance across all bands
  - Clouds — which are bright in visible AND shortwave infrared

The core insight exploited by EDI
-----------------------------------
Dust aerosols have a distinctive spectral fingerprint in MODIS surface
reflectance:

  1. HIGH reflectance in SWIR bands (B6: 1628 nm, B7: 2105 nm)
     Dust particles scatter and absorb SWIR energy strongly, raising
     apparent surface reflectance in those bands during storm events.

  2. ELEVATED reflectance in the Blue band (B3: 459–479 nm)
     Aerosol scattering raises apparent Blue reflectance above what
     bare-ground would produce.

  3. SUPPRESSED NIR (B2: 841–876 nm)
     Unlike healthy vegetation (high NIR) or clouds (high across all),
     dust-laden skies moderate NIR less predictably — so the NIR/Red
     ratio (NDVI) stays near zero over desert + dust scenes.

EDI formula
-----------
EDI is defined as the difference between two normalised ratios:

    NDDi  = (B7 − B3) / (B7 + B3)   ← Normalised Difference Dust Index
                                        High when SWIR3 >> Blue (dust signal)

    NDVI  = (B2 − B1) / (B2 + B1)   ← Normalised Difference Vegetation Index
                                        High over vegetation, low over bare/dust

    EDI   = NDDi − NDVI

Interpretation:
  •  EDI > 0   : SWIR-dominated scene with little vegetation
                 → candidate dust / bare arid surface
  •  EDI ≈ 0   : Spectrally ambiguous
  •  EDI < 0   : Vegetation or water dominates; unlikely to be SDS

Why subtract NDVI?
  Subtracting NDVI double-penalises vegetated pixels (their NDVI is high,
  pulling EDI strongly negative) and amplifies bare/dusty pixels (NDVI
  near zero → minimal subtraction → EDI driven almost entirely by NDDi).
  This suppresses false detections over irrigated cropland or mangroves
  which can show elevated SWIR from soil moisture.

Thresholds (indicative — may need local calibration)
------------------------------------------------------
  EDI ≥  0.20  →  High dust likelihood
  EDI ∈ [0.05, 0.20)  →  Moderate / suspect
  EDI <  0.05  →  Non-dust (vegetation, water, cloud-free desert)

MODIS MOD09GA band mapping used here
--------------------------------------
  sur_refl_b01  →  B1  Red      620–670 nm
  sur_refl_b02  →  B2  NIR      841–876 nm
  sur_refl_b03  →  B3  Blue     459–479 nm
  sur_refl_b06  →  B6  SWIR2   1628–1652 nm
  sur_refl_b07  →  B7  SWIR3   2105–2155 nm
"""

import ee


def compute_edi(img):
    """
    Compute the Enhanced Dust Index (EDI) for a single MOD09GA image.

    The input image must already have been scaled to reflectance units
    (i.e. multiplied by 0.0001) and cloud-masked.  The function adds a
    new band 'EDI' and returns the image with that band attached.

    Parameters
    ----------
    img : ee.Image
        A single MOD09GA image with scaled surface-reflectance bands.

    Returns
    -------
    ee.Image
        The input image with an additional 'EDI' band.
    """

    # ------------------------------------------------------------------
    # 1. Pull the bands we need.
    #    All values should be in [0, 1] after the 0.0001 scaling step.
    # ------------------------------------------------------------------
    B1 = img.select("sur_refl_b01")   # Red
    B2 = img.select("sur_refl_b02")   # NIR
    B3 = img.select("sur_refl_b03")   # Blue
    B7 = img.select("sur_refl_b07")   # SWIR3

    # ------------------------------------------------------------------
    # 2. Normalised Difference Dust Index (NDDi)
    #
    #    NDDi = (B7 − B3) / (B7 + B3)
    #
    #    Dust aerosols in the MODIS SWIR3 channel appear brighter than
    #    they do in the Blue channel, so NDDi is positive over dusty
    #    surfaces and negative over water or dense vegetation.
    # ------------------------------------------------------------------
    nddi = (
        B7.subtract(B3)
          .divide(B7.add(B3))
          .rename("NDDI")
    )

    # ------------------------------------------------------------------
    # 3. Normalised Difference Vegetation Index (NDVI)
    #
    #    NDVI = (B2 − B1) / (B2 + B1)
    #
    #    Used here as a suppression factor: vegetated pixels will have
    #    high NDVI, causing EDI to be strongly negative and thus
    #    excluded from SDS classification.
    # ------------------------------------------------------------------
    ndvi = (
        B2.subtract(B1)
          .divide(B2.add(B1))
          .rename("NDVI_tmp")
    )

    # ------------------------------------------------------------------
    # 4. Enhanced Dust Index
    #
    #    EDI = NDDi − NDVI
    #
    #    •  Over active dust storms:   NDDi high (+), NDVI low (≈0)
    #                                  → EDI strongly positive
    #    •  Over bare desert (no SDS): NDDi moderate, NDVI near zero
    #                                  → EDI moderate
    #    •  Over vegetation:           NDDi low/negative, NDVI high
    #                                  → EDI strongly negative
    #    •  Over water:                Both ratios near −1
    #                                  → EDI ≈ 0 (ambiguous — use NDWI
    #                                     in downstream classification)
    # ------------------------------------------------------------------
    edi = nddi.subtract(ndvi).rename("EDI")

    return img.addBands(edi)


def classify_edi(edi_band):
    """
    Convert a continuous EDI band into a 3-class dust likelihood map.

    Classes
    -------
    2  →  High dust likelihood    (EDI ≥ 0.20)
    1  →  Moderate / suspect      (0.05 ≤ EDI < 0.20)
    0  →  Non-dust                (EDI < 0.05)

    Parameters
    ----------
    edi_band : ee.Image (single band named 'EDI')

    Returns
    -------
    ee.Image
        Single-band image 'EDI_class' with integer values 0, 1, or 2.
    """
    high     = edi_band.gte(0.20).multiply(2)
    moderate = edi_band.gte(0.05).And(edi_band.lt(0.20)).multiply(1)

    # Stack by taking the maximum — ensures high beats moderate beats 0
    edi_class = high.max(moderate).rename("EDI_class")
    return edi_class
