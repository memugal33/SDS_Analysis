/**
 * ediWithAlpha.js
 * Enhanced Dust Index via Linear Spectral Unmixing (Han et al. 2013 / Wang et al. 2022).
 * EDI = [α·B7 − (1−α)·B3] / [α·B7 + (1−α)·B3]
 * where α = dust endmember fraction from LSU.
 *
 * Usage:
 *   var edi = require('users/YOUR_USERNAME/YOUR_REPO:javascript/ediWithAlpha');
 *   var img2 = edi.computeDustFraction(img);   // adds alpha_dust, alpha_cloud, alpha_water
 *   var img3 = edi.computeEdiAlpha(img2);       // adds EDI_alpha
 *   var cls  = edi.classifyEdiAlpha(img3.select('EDI_alpha')); // adds EDI_alpha_class
 */

// Endmember spectra [B1,B2,B3,B4,B5,B6,B7] — approximated for Dubai/Gulf region.
// Calibrate against actual MOD09GA pixels for best results.
var ENDMEMBERS = [
  [0.35, 0.38, 0.30, 0.36, 0.42, 0.40, 0.38],  // dust
  [0.28, 0.32, 0.18, 0.26, 0.38, 0.42, 0.44],  // bare soil
  [0.80, 0.82, 0.82, 0.81, 0.70, 0.55, 0.45],  // cloud
  [0.04, 0.02, 0.06, 0.05, 0.01, 0.01, 0.01],  // water
];

/**
 * Linear Spectral Unmixing → dust fraction α (+ cloud, water fractions).
 * Adds bands: alpha_dust, alpha_cloud, alpha_water.
 * @param {ee.Image} image - Scaled MOD09GA image.
 * @return {ee.Image}
 */
var computeDustFraction = function(image) {
  var bands = image.select([
    'sur_refl_b01','sur_refl_b02','sur_refl_b03','sur_refl_b04',
    'sur_refl_b05','sur_refl_b06','sur_refl_b07'
  ]);

  // Constrained least-squares unmixing (sum-to-one, non-negative)
  var unmixed = bands.unmix(ENDMEMBERS, true, true);
  var alphas = unmixed.rename(['alpha_dust','alpha_soil','alpha_cloud','alpha_water']);

  return image.addBands(alphas.select(['alpha_dust','alpha_cloud','alpha_water']));
};

/**
 * Compute α-weighted EDI. Adds band: EDI_alpha (range −1 to +1).
 * @param {ee.Image} image - Image with sur_refl_b03, sur_refl_b07, alpha_dust.
 * @return {ee.Image}
 */
var computeEdiAlpha = function(image) {
  var B3    = image.select('sur_refl_b03');
  var B7    = image.select('sur_refl_b07');
  var alpha = image.select('alpha_dust');
  var oneMa = ee.Image(1).subtract(alpha);

  var alphaSWIR = alpha.multiply(B7);
  var bgBlue    = oneMa.multiply(B3);

  var edi = alphaSWIR.subtract(bgBlue)
              .divide(alphaSWIR.add(bgBlue).add(ee.Image(1e-6)))
              .rename('EDI_alpha');

  return image.addBands(edi);
};

/**
 * Classify EDI_alpha into 3 dust likelihood classes.
 * 0 = non-dust (< 0.05), 1 = moderate (0.05–0.20), 2 = high (≥ 0.20).
 * @param {ee.Image} ediBand - Single-band 'EDI_alpha' image.
 * @return {ee.Image} 'EDI_alpha_class' (0/1/2)
 */
var classifyEdiAlpha = function(ediBand) {
  var high     = ediBand.gte(0.20).multiply(2);
  var moderate = ediBand.gte(0.05).and(ediBand.lt(0.20)).multiply(1);
  return high.max(moderate).rename('EDI_alpha_class');
};

exports.computeDustFraction = computeDustFraction;
exports.computeEdiAlpha     = computeEdiAlpha;
exports.classifyEdiAlpha    = classifyEdiAlpha;
