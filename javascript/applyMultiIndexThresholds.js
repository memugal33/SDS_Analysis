/**
 * applyMultiIndexThresholds.js
 * Table 3 threshold labelling — Wang et al. (2022).
 * Labels each pixel as one of 11 surface/atmospheric classes.
 *
 * Usage:
 *   var mit = require('users/YOUR_USERNAME/YOUR_REPO:javascript/applyMultiIndexThresholds');
 *   var imgAux = mit.attachAuxiliaryData(img);
 *   var label  = mit.applyMultiIndexThresholds(imgAux);
 *   Map.addLayer(label, mit.VIS_CLASS, 'Classes');
 */

// Class codes (0–11)
var CLASS = {
  UNCLASSIFIED: 0,
  TCC: 1,   // Thick Cloud
  CS:  2,   // Cirrostratus
  TNC: 3,   // Thin Cloud
  TCD: 4,   // Thick Dust
  TND: 5,   // Thin Dust
  WD:  6,   // Water Dust
  LB:  7,   // Dry Lakebed
  WB:  8,   // Water Bodies
  BL:  9,   // Bare Land
  VL:  10,  // Vegetated Land
  SI:  11,  // Snow and Ice
};

var CLASS_PALETTE = [
  'ffffff',  // 0  Unclassified
  '4575b4',  // 1  TcC Thick Cloud
  '74add1',  // 2  Cs  Cirrostratus
  'abd9e9',  // 3  TnC Thin Cloud
  'd73027',  // 4  TcD Thick Dust
  'fdae61',  // 5  TnD Thin Dust
  'fee090',  // 6  WD  Water Dust
  'f4a582',  // 7  Lb  Dry Lakebed
  '313695',  // 8  WB  Water Bodies
  'c8a96e',  // 9  BL  Bare Land
  '1a9641',  // 10 VL  Vegetated Land
  'e0f3f8',  // 11 SI  Snow and Ice
];

var VIS_CLASS = {min: 0, max: 11, palette: CLASS_PALETTE};

var CLASS_LEGEND = {
  '0 — Unclassified':     '#ffffff',
  '1 — Thick Cloud':      '#4575b4',
  '2 — Cirrostratus':     '#74add1',
  '3 — Thin Cloud':       '#abd9e9',
  '4 — Thick Dust':       '#d73027',
  '5 — Thin Dust':        '#fdae61',
  '6 — Water Dust':       '#fee090',
  '7 — Dry Lakebed':      '#f4a582',
  '8 — Water Bodies':     '#313695',
  '9 — Bare Land':        '#c8a96e',
  '10 — Vegetated Land':  '#1a9641',
  '11 — Snow and Ice':    '#e0f3f8',
};

var IGBP_WATER = 17;

// Auxiliary datasets (loaded once)
var _srtm = ee.Image('USGS/SRTMGL1_003').select('elevation').rename('DEM');
var _lct  = ee.ImageCollection('MODIS/061/MCD12Q1')
              .filter(ee.Filter.calendarRange(2022, 2022, 'year'))
              .first().select('LC_Type1').rename('LCT');

/**
 * Attach SRTM DEM and MCD12Q1 LCT bands to the image.
 * @param {ee.Image} image
 * @return {ee.Image}
 */
var attachAuxiliaryData = function(image) {
  return image.addBands([_srtm, _lct]);
};

/**
 * Apply Table 3 threshold criteria. Returns 'class_label' band (uint8, 0–11).
 * Priority (highest last in .where chain): VL→BL→Lb→WD→WB→TnD→TcD→TnC→Cs→TcC→SI
 * @param {ee.Image} image - Image with all index, alpha, DEM, LCT bands.
 * @return {ee.Image}
 */
var applyMultiIndexThresholds = function(image) {
  var B1    = image.select('sur_refl_b01');
  var B7    = image.select('sur_refl_b07');
  var EDI   = image.select('EDI_alpha');
  var NDDI  = image.select('NDDI');
  var NDVI  = image.select('NDVI');
  var NDWI  = image.select('NDWI');
  var NDSI  = image.select('NDSI');
  var MBWI  = image.select('MBWI');
  var MBSCI = image.select('MBSCI');
  var aDust  = image.select('alpha_dust');
  var aCloud = image.select('alpha_cloud');
  var aWater = image.select('alpha_water');
  var DEM   = image.select('DEM');
  var LCT   = image.select('LCT');

  // Range test helper: lo <= band <= hi
  var inRange = function(band, lo, hi) {
    return band.gte(lo).and(band.lte(hi));
  };

  var si = inRange(EDI,-1.2,-0.8).and(inRange(NDDI,-1.0,-0.8))
    .and(inRange(NDSI,0.6,1.0)).and(inRange(NDVI,-0.2,0.1))
    .and(inRange(NDWI,-0.1,0.2)).and(inRange(MBWI,-0.3,1.0))
    .and(MBSCI.gt(10)).and(B7.gt(0.05))
    .and(aCloud.lt(0.2)).and(DEM.gt(2500));

  var tcc = inRange(EDI,-1.2,-0.8).and(inRange(NDDI,-0.9,-0.6))
    .and(inRange(NDSI,0.4,0.8)).and(inRange(NDVI,-0.1,0.1))
    .and(inRange(NDWI,-0.1,0.1)).and(inRange(MBWI,-0.6,-0.2))
    .and(B1.gt(0.9));

  var cs = inRange(NDDI,-0.6,-0.3).and(inRange(NDVI,-0.1,0.1))
    .and(inRange(NDWI,-0.1,0.1)).and(inRange(MBSCI,1.5,5.0))
    .and(B7.gt(0.25));

  var tnc = inRange(NDDI,-0.7,0.1).and(inRange(NDSI,-0.15,0.0))
    .and(inRange(NDVI,0.0,0.1)).and(inRange(NDWI,-0.15,0.0))
    .and(B7.lt(0.4)).and(aCloud.gt(0)).and(aDust.lt(0.8));

  var wb = inRange(NDDI,-1.2,-0.9).and(inRange(NDWI,-1.1,-0.1))
    .and(inRange(NDVI,-0.1,0.4)).and(inRange(MBWI,-0.1,0.4))
    .and(aWater.gt(0.85));

  var tcd = inRange(EDI,-0.2,1.0).and(inRange(NDDI,0.0,0.15))
    .and(inRange(NDSI,-0.4,0.0)).and(inRange(NDVI,-0.2,0.2))
    .and(inRange(NDWI,-0.2,0.1)).and(inRange(MBWI,-1.5,-1.0));

  var tnd = inRange(EDI,-0.1,0.8).and(inRange(NDDI,-0.2,0.3))
    .and(inRange(NDSI,-0.3,0.1)).and(inRange(NDVI,-0.1,0.2))
    .and(inRange(NDWI,-0.2,0.0)).and(inRange(MBWI,-1.3,-0.8))
    .and(B1.lt(0.55));

  var wd = inRange(EDI,-0.6,0.4).and(inRange(NDDI,-0.4,-0.2))
    .and(inRange(NDSI,0.0,0.2)).and(inRange(NDVI,-0.1,0.1))
    .and(inRange(NDWI,0.0,0.2)).and(inRange(MBWI,-0.6,-0.2))
    .and(LCT.eq(IGBP_WATER));

  var lb = inRange(EDI,-0.2,0.1).and(inRange(NDDI,0.0,0.1))
    .and(inRange(NDSI,-0.3,-0.1)).and(inRange(NDVI,0.0,0.1))
    .and(inRange(NDWI,-0.1,0.0)).and(inRange(MBWI,-0.8,-0.3));

  var bl = inRange(EDI,-1.2,-0.2).and(inRange(NDDI,0.0,0.5))
    .and(inRange(NDSI,-0.6,-0.1)).and(inRange(NDVI,0.0,0.2))
    .and(inRange(NDWI,-0.4,0.0)).and(inRange(MBWI,-1.5,-0.8));

  var vl = inRange(EDI,0.0,0.6).and(inRange(NDDI,-0.7,-0.3))
    .and(inRange(NDVI,0.05,0.9)).and(inRange(NDWI,-0.8,0.0))
    .and(inRange(MBWI,-0.8,-0.4));

  // Apply in reverse priority order — last .where() wins
  var label = ee.Image.constant(CLASS.UNCLASSIFIED).rename('class_label');
  label = label.where(vl,  CLASS.VL);
  label = label.where(bl,  CLASS.BL);
  label = label.where(lb,  CLASS.LB);
  label = label.where(wd,  CLASS.WD);
  label = label.where(wb,  CLASS.WB);
  label = label.where(tnd, CLASS.TND);
  label = label.where(tcd, CLASS.TCD);
  label = label.where(tnc, CLASS.TNC);
  label = label.where(cs,  CLASS.CS);
  label = label.where(tcc, CLASS.TCC);
  label = label.where(si,  CLASS.SI);

  return ee.Image(label.toUint8().copyProperties(image, ['system:time_start']));
};

exports.attachAuxiliaryData        = attachAuxiliaryData;
exports.applyMultiIndexThresholds  = applyMultiIndexThresholds;
exports.VIS_CLASS                  = VIS_CLASS;
exports.CLASS_LEGEND               = CLASS_LEGEND;
exports.CLASS                      = CLASS;
