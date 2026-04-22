/**
 * computeAllIndices.js
 * Spectral indices from Table 2, Wang et al. (2022).
 * Input: scaled MOD09GA image (sur_refl_b* bands, 0–1).
 * Output: same image + NDDI, NDVI, NDWI, NDSI, MBWI, MBSCI bands.
 *
 * Usage in GEE Code Editor:
 *   var idx = require('users/YOUR_USERNAME/YOUR_REPO:javascript/computeAllIndices');
 *   var imgWithIndices = idx.computeAllIndices(img);
 */

var eps = ee.Image(1e-9);

/**
 * Compute all six SDS spectral indices and attach to image.
 * @param {ee.Image} image - Scaled MOD09GA image.
 * @return {ee.Image}
 */
var computeAllIndices = function(image) {
  var B1 = image.select('sur_refl_b01');  // Red
  var B2 = image.select('sur_refl_b02');  // NIR
  var B3 = image.select('sur_refl_b03');  // Blue
  var B4 = image.select('sur_refl_b04');  // Green
  var B6 = image.select('sur_refl_b06');  // SWIR-2
  var B7 = image.select('sur_refl_b07');  // SWIR-3

  // NDDI: (B7-B3)/(B7+B3) — dust aerosol detector [Qu et al. 2006]
  var nddi = B7.subtract(B3).divide(B7.add(B3).add(eps)).rename('NDDI');

  // NDVI: (B2-B1)/(B2+B1) — vegetation greenness [Tucker 1979]
  var ndvi = B2.subtract(B1).divide(B2.add(B1).add(eps)).rename('NDVI');

  // NDWI: (B4-B2)/(B4+B2) — open water [McFeeters 2007]
  var ndwi = B4.subtract(B2).divide(B4.add(B2).add(eps)).rename('NDWI');

  // NDSI: (B4-B6)/(B4+B6) — snow/ice [Salomonson & Appel 2004]
  var ndsi = B4.subtract(B6).divide(B4.add(B6).add(eps)).rename('NDSI');

  // MBWI: 2*B4-B1-B2-B6-B7 — water body index [X.X. Wang et al. 2018]
  var mbwi = B4.multiply(2).subtract(B1).subtract(B2)
               .subtract(B6).subtract(B7).rename('MBWI');

  // MBSCI: (B4+B1)/(B6+B7) — cloud/bare soil separator [Wang et al. 2022]
  var mbsci = B4.add(B1).divide(B6.add(B7).add(eps)).rename('MBSCI');

  return image.addBands([nddi, ndvi, ndwi, ndsi, mbwi, mbsci]);
};

exports.computeAllIndices = computeAllIndices;
