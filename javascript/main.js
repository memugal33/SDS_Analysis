/**
 * main.js — SDS Analysis: Dubai
 * Translates sds_analysis.ipynb for GEE Code Editor.
 * Reference: Wang et al. (2022) https://doi.org/10.1080/22797254.2022.2093278
 *
 * SETUP: Replace YOUR_USERNAME and YOUR_REPO below with your GEE username
 *        and repository name, then paste each required script into that repo.
 */

var REPO = 'users/YOUR_USERNAME/YOUR_REPO';

var idxLib = require(REPO + ':javascript/computeAllIndices');
var ediLib = require(REPO + ':javascript/ediWithAlpha');
var mitLib = require(REPO + ':javascript/applyMultiIndexThresholds');

// =============================================================================
// 1. AREA OF INTEREST — Dubai
// =============================================================================
var adm2 = ee.FeatureCollection('WM/geoLab/geoBoundaries/600/ADM2');
var dubaiFC = adm2
  .filter(ee.Filter.eq('shapeGroup', 'ARE'))
  .filter(ee.Filter.eq('shapeName', 'Dubai'));
var roi = dubaiFC.geometry();

Map.centerObject(roi, 9);
Map.addLayer(dubaiFC, {color: '#FF6600'}, 'Dubai AOI');

print('Dubai features:', dubaiFC.size());

// =============================================================================
// 2. LOAD & SCALE MOD09GA
// =============================================================================
var scaleAndMask = function(img) {
  var b = img.select([
    'sur_refl_b01','sur_refl_b02','sur_refl_b03','sur_refl_b04',
    'sur_refl_b05','sur_refl_b06','sur_refl_b07'
  ]).multiply(0.0001);
  // Cloud masking deferred — dust pixels can trigger cloud QA flags.
  // Multi-index thresholding (Section 7) handles cloud/dust separation.
  return img.addBands(b, null, true).copyProperties(img, img.propertyNames());
};

// =============================================================================
// 3. IMAGE COLLECTION & SINGLE-DATE SELECTION
// =============================================================================
var START = '2022-01-01';
var END   = '2022-12-31';
var EVENT_DATE = '2022-04-18';  // known dust event

var icRaw = ee.ImageCollection('MODIS/061/MOD09GA')
  .filterDate(START, END)
  .filterBounds(roi)
  .map(scaleAndMask);

var eventDate = ee.Date(EVENT_DATE);
var dayCol    = icRaw.filterDate(eventDate, eventDate.advance(1, 'day'));
print('Images on ' + EVENT_DATE + ':', dayCol.size());

// img_raw: reflectance bands only — entry point for label_image()
var imgRaw = ee.Image(dayCol.sort('system:time_start').first()).clip(roi);

// img: reflectance + spectral indices — used for Section 4 maps
var img = idxLib.computeAllIndices(imgRaw);

// =============================================================================
// 4. SPECTRAL INDEX MAPS — NDDI, MBSCI, NDVI
// =============================================================================
Map.addLayer(
  img.select('NDDI'),
  {min:-0.3, max:0.6, palette:['#2166ac','#92c5de','#f7f7f7','#fddbc7','#d6604d','#b2182b']},
  'NDDI (Dust Index)'
);

Map.addLayer(
  img.select('MBSCI'),
  {min:0.3, max:1.5, palette:['#4d9221','#a1d76a','#f7f7f7','#e9a3c9','#c51b7d']},
  'MBSCI (Bare Soil)'
);

Map.addLayer(
  img.select('NDVI'),
  {min:-0.1, max:0.5, palette:['#8c510a','#d8b365','#f6e8c3','#c7eae5','#5ab4ac','#01665e']},
  'NDVI (Vegetation)'
);

// =============================================================================
// 5. EDI-α — Enhanced Dust Index with Dust Fraction
// =============================================================================
var imgAlpha    = ediLib.computeDustFraction(img);
var imgEdi      = ediLib.computeEdiAlpha(imgAlpha);
var ediAlphaBand = imgEdi.select('EDI_alpha');
var ediClass     = ediLib.classifyEdiAlpha(ediAlphaBand);

var ediVis = {
  min: -0.2, max: 0.4,
  palette: ['#313695','#4575b4','#74add1','#abd9e9','#e0f3f8',
            '#ffffbf','#fee090','#fdae61','#f46d43','#d73027','#a50026']
};

Map.addLayer(ediAlphaBand, ediVis, 'EDI-α (continuous)');
Map.addLayer(
  ediClass,
  {min:0, max:2, palette:['#d9d9d9','#fdae61','#d73027']},
  'EDI-α class (0=none, 1=mod, 2=high)'
);

// =============================================================================
// 6. FULL PIPELINE — label_image (Table 3 classification)
// =============================================================================

/**
 * Full labelling pipeline: scaled reflectance → class_label (0–11).
 * @param {ee.Image} image - Scaled MOD09GA image.
 * @return {ee.Image} class_label band (uint8, 0–11).
 */
var labelImage = function(image) {
  image = idxLib.computeAllIndices(image);
  image = ediLib.computeDustFraction(image);
  image = ediLib.computeEdiAlpha(image);
  image = mitLib.attachAuxiliaryData(image);
  return mitLib.applyMultiIndexThresholds(image);
};

var classLabel = labelImage(imgRaw).clip(roi);

// =============================================================================
// 7. VISUALISATION — Class Label Map
// =============================================================================
Map.addLayer(classLabel, mitLib.VIS_CLASS, 'Class Labels (Table 3)');

// Legend panel
var legendPanel = ui.Panel({style: {position: 'bottom-left', padding: '8px'}});
legendPanel.add(ui.Label('Wang et al. 2022 — Table 3', {fontWeight: 'bold'}));

var legendEntries = [
  {label: '0 Unclassified',    color: '#ffffff'},
  {label: '1 Thick Cloud',     color: '#4575b4'},
  {label: '2 Cirrostratus',    color: '#74add1'},
  {label: '3 Thin Cloud',      color: '#abd9e9'},
  {label: '4 Thick Dust',      color: '#d73027'},
  {label: '5 Thin Dust',       color: '#fdae61'},
  {label: '6 Water Dust',      color: '#fee090'},
  {label: '7 Dry Lakebed',     color: '#f4a582'},
  {label: '8 Water Bodies',    color: '#313695'},
  {label: '9 Bare Land',       color: '#c8a96e'},
  {label: '10 Vegetated Land', color: '#1a9641'},
  {label: '11 Snow and Ice',   color: '#e0f3f8'},
];

legendEntries.forEach(function(e) {
  legendPanel.add(ui.Panel([
    ui.Label('', {
      backgroundColor: e.color,
      padding: '8px',
      margin: '0 4px 0 0',
      border: '1px solid #ccc'
    }),
    ui.Label(e.label, {margin: '0'})
  ], ui.Panel.Layout.flow('horizontal')));
});

Map.add(legendPanel);
