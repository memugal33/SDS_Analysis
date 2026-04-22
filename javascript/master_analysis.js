/**
 * Master Analysis — SDS Detection: Dubai
 * Wang et al. (2022) https://doi.org/10.1080/22797254.2022.2093278
 */

// =============================================================================
// 1. AREA OF INTEREST
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
// 2. QA MASKING & SCALING
// =============================================================================

var scaleAndMask = function(img) {
  var b = img.select([
    'sur_refl_b01','sur_refl_b02','sur_refl_b03','sur_refl_b04',
    'sur_refl_b05','sur_refl_b06','sur_refl_b07'
  ]).multiply(0.0001);
  // Cloud masking deferred — dust triggers cloud QA flags; handled in Section 7.
  return img.addBands(b, null, true).copyProperties(img, img.propertyNames());
};

// =============================================================================
// 3. IMAGE COLLECTION & DATE SELECTION
// =============================================================================

var START      = '2022-01-01';
var END        = '2022-12-31';
var EVENT_DATE = '2022-04-18';

var icRaw = ee.ImageCollection('MODIS/061/MOD09GA')
  .filterDate(START, END)
  .filterBounds(roi)
  .map(scaleAndMask);

var eventDate = ee.Date(EVENT_DATE);
var dayCol    = icRaw.filterDate(eventDate, eventDate.advance(1, 'day'));
print('Images on ' + EVENT_DATE + ':', dayCol.size());

var imgRaw = ee.Image(dayCol.sort('system:time_start').first()).clip(roi);

// =============================================================================
// 4. SPECTRAL INDICES  (Table 2, Wang et al. 2022)
// =============================================================================

var computeAllIndices = function(image) {
  var B1 = image.select('sur_refl_b01');
  var B2 = image.select('sur_refl_b02');
  var B3 = image.select('sur_refl_b03');
  var B4 = image.select('sur_refl_b04');
  var B6 = image.select('sur_refl_b06');
  var B7 = image.select('sur_refl_b07');
  var eps = ee.Image(1e-9);

  var nddi  = B7.subtract(B3).divide(B7.add(B3).add(eps)).rename('NDDI');
  var ndvi  = B2.subtract(B1).divide(B2.add(B1).add(eps)).rename('NDVI');
  var ndwi  = B4.subtract(B2).divide(B4.add(B2).add(eps)).rename('NDWI');
  var ndsi  = B4.subtract(B6).divide(B4.add(B6).add(eps)).rename('NDSI');
  var mbwi  = B4.multiply(2).subtract(B1).subtract(B2).subtract(B6).subtract(B7).rename('MBWI');
  var mbsci = B4.add(B1).divide(B6.add(B7).add(eps)).rename('MBSCI');

  return image.addBands([nddi, ndvi, ndwi, ndsi, mbwi, mbsci]);
};

var img = computeAllIndices(imgRaw);

// =============================================================================
// 5. INDEX MAPS — NDDI, MBSCI, NDVI
// =============================================================================

Map.addLayer(
  img.select('NDDI'),
  {min:-0.3, max:0.6, palette:['#2166ac','#92c5de','#f7f7f7','#fddbc7','#d6604d','#b2182b']},
  'NDDI'
);
Map.addLayer(
  img.select('MBSCI'),
  {min:0.3, max:1.5, palette:['#4d9221','#a1d76a','#f7f7f7','#e9a3c9','#c51b7d']},
  'MBSCI'
);
Map.addLayer(
  img.select('NDVI'),
  {min:-0.1, max:0.5, palette:['#8c510a','#d8b365','#f6e8c3','#c7eae5','#5ab4ac','#01665e']},
  'NDVI'
);

// =============================================================================
// 6. EDI-α  (Han et al. 2013 / Wang et al. 2022)
//    EDI = [α·B7 − (1−α)·B3] / [α·B7 + (1−α)·B3]
//    α = dust endmember fraction from Linear Spectral Unmixing
// =============================================================================

// Endmember spectra [B1,B2,B3,B4,B5,B6,B7] — approximate for Gulf region.
var ENDMEMBERS = [
  [0.35, 0.38, 0.30, 0.36, 0.42, 0.40, 0.38],  // dust
  [0.28, 0.32, 0.18, 0.26, 0.38, 0.42, 0.44],  // bare soil
  [0.80, 0.82, 0.82, 0.81, 0.70, 0.55, 0.45],  // cloud
  [0.04, 0.02, 0.06, 0.05, 0.01, 0.01, 0.01],  // water
];

var computeDustFraction = function(image) {
  var bands = image.select([
    'sur_refl_b01','sur_refl_b02','sur_refl_b03','sur_refl_b04',
    'sur_refl_b05','sur_refl_b06','sur_refl_b07'
  ]);
  var unmixed = bands.unmix(ENDMEMBERS, true, true);
  var alphas  = unmixed.rename(['alpha_dust','alpha_soil','alpha_cloud','alpha_water']);
  return image.addBands(alphas.select(['alpha_dust','alpha_cloud','alpha_water']));
};

var computeEdiAlpha = function(image) {
  var B3    = image.select('sur_refl_b03');
  var B7    = image.select('sur_refl_b07');
  var alpha = image.select('alpha_dust');
  var oneMa = ee.Image(1).subtract(alpha);
  var num   = alpha.multiply(B7).subtract(oneMa.multiply(B3));
  var den   = alpha.multiply(B7).add(oneMa.multiply(B3)).add(ee.Image(1e-6));
  return image.addBands(num.divide(den).rename('EDI_alpha'));
};

var classifyEdiAlpha = function(ediBand) {
  var high = ediBand.gte(0.20).multiply(2);
  var mod  = ediBand.gte(0.05).and(ediBand.lt(0.20)).multiply(1);
  return high.max(mod).rename('EDI_alpha_class');
};

var imgAlpha   = computeDustFraction(img);
var imgEdi     = computeEdiAlpha(imgAlpha);
var ediBand    = imgEdi.select('EDI_alpha');
var ediClass   = classifyEdiAlpha(ediBand);

var ediVis = {
  min:-0.2, max:0.4,
  palette:['#313695','#4575b4','#74add1','#abd9e9','#e0f3f8',
           '#ffffbf','#fee090','#fdae61','#f46d43','#d73027','#a50026']
};

Map.addLayer(ediBand,  ediVis,                                               'EDI-α');
Map.addLayer(ediClass, {min:0, max:2, palette:['#d9d9d9','#fdae61','#d73027']}, 'EDI-α class');

// =============================================================================
// 7. MULTI-INDEX THRESHOLD LABELLING  (Table 3, Wang et al. 2022)
// =============================================================================

var IGBP_WATER = 17;

var _srtm = ee.Image('USGS/SRTMGL1_003').select('elevation').rename('DEM');
var _lct  = ee.ImageCollection('MODIS/061/MCD12Q1')
              .filter(ee.Filter.calendarRange(2022, 2022, 'year'))
              .first().select('LC_Type1').rename('LCT');

var attachAuxiliaryData = function(image) {
  return image.addBands([_srtm, _lct]);
};

var applyMultiIndexThresholds = function(image) {
  var B1     = image.select('sur_refl_b01');
  var B7     = image.select('sur_refl_b07');
  var EDI    = image.select('EDI_alpha');
  var NDDI   = image.select('NDDI');
  var NDVI   = image.select('NDVI');
  var NDWI   = image.select('NDWI');
  var NDSI   = image.select('NDSI');
  var MBWI   = image.select('MBWI');
  var MBSCI  = image.select('MBSCI');
  var aDust  = image.select('alpha_dust');
  var aCloud = image.select('alpha_cloud');
  var aWater = image.select('alpha_water');
  var DEM    = image.select('DEM');
  var LCT    = image.select('LCT');

  var inRange = function(b, lo, hi) { return b.gte(lo).and(b.lte(hi)); };

  var si = inRange(EDI,-1.2,-0.8).and(inRange(NDDI,-1.0,-0.8))
    .and(inRange(NDSI,0.6,1.0)).and(inRange(NDVI,-0.2,0.1))
    .and(inRange(NDWI,-0.1,0.2)).and(inRange(MBWI,-0.3,1.0))
    .and(MBSCI.gt(10)).and(B7.gt(0.05)).and(aCloud.lt(0.2)).and(DEM.gt(2500));

  var tcc = inRange(EDI,-1.2,-0.8).and(inRange(NDDI,-0.9,-0.6))
    .and(inRange(NDSI,0.4,0.8)).and(inRange(NDVI,-0.1,0.1))
    .and(inRange(NDWI,-0.1,0.1)).and(inRange(MBWI,-0.6,-0.2)).and(B1.gt(0.9));

  var cs = inRange(NDDI,-0.6,-0.3).and(inRange(NDVI,-0.1,0.1))
    .and(inRange(NDWI,-0.1,0.1)).and(inRange(MBSCI,1.5,5.0)).and(B7.gt(0.25));

  var tnc = inRange(NDDI,-0.7,0.1).and(inRange(NDSI,-0.15,0.0))
    .and(inRange(NDVI,0.0,0.1)).and(inRange(NDWI,-0.15,0.0))
    .and(B7.lt(0.4)).and(aCloud.gt(0)).and(aDust.lt(0.8));

  var wb = inRange(NDDI,-1.2,-0.9).and(inRange(NDWI,-1.1,-0.1))
    .and(inRange(NDVI,-0.1,0.4)).and(inRange(MBWI,-0.1,0.4)).and(aWater.gt(0.85));

  var tcd = inRange(EDI,-0.2,1.0).and(inRange(NDDI,0.0,0.15))
    .and(inRange(NDSI,-0.4,0.0)).and(inRange(NDVI,-0.2,0.2))
    .and(inRange(NDWI,-0.2,0.1)).and(inRange(MBWI,-1.5,-1.0));

  var tnd = inRange(EDI,-0.1,0.8).and(inRange(NDDI,-0.2,0.3))
    .and(inRange(NDSI,-0.3,0.1)).and(inRange(NDVI,-0.1,0.2))
    .and(inRange(NDWI,-0.2,0.0)).and(inRange(MBWI,-1.3,-0.8)).and(B1.lt(0.55));

  var wd = inRange(EDI,-0.6,0.4).and(inRange(NDDI,-0.4,-0.2))
    .and(inRange(NDSI,0.0,0.2)).and(inRange(NDVI,-0.1,0.1))
    .and(inRange(NDWI,0.0,0.2)).and(inRange(MBWI,-0.6,-0.2)).and(LCT.eq(IGBP_WATER));

  var lb = inRange(EDI,-0.2,0.1).and(inRange(NDDI,0.0,0.1))
    .and(inRange(NDSI,-0.3,-0.1)).and(inRange(NDVI,0.0,0.1))
    .and(inRange(NDWI,-0.1,0.0)).and(inRange(MBWI,-0.8,-0.3));

  var bl = inRange(EDI,-1.2,-0.2).and(inRange(NDDI,0.0,0.5))
    .and(inRange(NDSI,-0.6,-0.1)).and(inRange(NDVI,0.0,0.2))
    .and(inRange(NDWI,-0.4,0.0)).and(inRange(MBWI,-1.5,-0.8));

  var vl = inRange(EDI,0.0,0.6).and(inRange(NDDI,-0.7,-0.3))
    .and(inRange(NDVI,0.05,0.9)).and(inRange(NDWI,-0.8,0.0))
    .and(inRange(MBWI,-0.8,-0.4));

  // Apply lowest→highest priority; last .where() wins
  var label = ee.Image.constant(0).rename('class_label');
  label = label.where(vl, 10).where(bl, 9).where(lb, 7).where(wd, 6)
               .where(wb,  8).where(tnd,5).where(tcd,4).where(tnc,3)
               .where(cs,  2).where(tcc,1).where(si, 11);

  return ee.Image(label.toUint8().copyProperties(image, ['system:time_start']));
};

// Full pipeline: scaled reflectance → class_label (0–11)
var labelImage = function(image) {
  image = computeAllIndices(image);
  image = computeDustFraction(image);
  image = computeEdiAlpha(image);
  image = attachAuxiliaryData(image);
  return applyMultiIndexThresholds(image);
};

var classLabel = labelImage(imgRaw).clip(roi);

// =============================================================================
// 8. CLASS LABEL MAP & LEGEND
// =============================================================================

var CLASS_PALETTE = [
  'ffffff','4575b4','74add1','abd9e9','d73027',
  'fdae61','fee090','f4a582','313695','c8a96e','1a9641','e0f3f8'
];

Map.addLayer(classLabel, {min:0, max:11, palette:CLASS_PALETTE}, 'Class Labels (Table 3)');

var legendEntries = [
  ['0 Unclassified','#ffffff'],  ['1 Thick Cloud','#4575b4'],
  ['2 Cirrostratus','#74add1'],  ['3 Thin Cloud','#abd9e9'],
  ['4 Thick Dust','#d73027'],    ['5 Thin Dust','#fdae61'],
  ['6 Water Dust','#fee090'],    ['7 Dry Lakebed','#f4a582'],
  ['8 Water Bodies','#313695'],  ['9 Bare Land','#c8a96e'],
  ['10 Vegetated Land','#1a9641'], ['11 Snow & Ice','#e0f3f8'],
];

var legend = ui.Panel({style: {position:'bottom-left', padding:'8px'}});
legend.add(ui.Label('Wang et al. 2022 — Table 3', {fontWeight:'bold', fontSize:'13px'}));
legendEntries.forEach(function(e) {
  legend.add(ui.Panel([
    ui.Label('', {backgroundColor:e[1], padding:'8px', margin:'2px 6px 2px 0', border:'1px solid #ccc'}),
    ui.Label(e[0], {margin:'4px 0'})
  ], ui.Panel.Layout.flow('horizontal')));
});
Map.add(legend);
