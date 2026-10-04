// Export the MODIS BRDF shadow parameter for the albedo-based scheme (AEM).
//
// Runs in the Google Earth Engine Code Editor. For every month of YEAR it
// writes one GeoTIFF (brdf_YYYYMM.tif) to Google Drive with four bands:
// albedo, shadow, omega_n and f_iso, as the monthly mean of the daily values
// on a 0.25 degree grid. scripts/brdf_tif_to_nc.py converts the twelve files
// into the NetCDF file that odem.py reads with --modis-brdf.
//
// Input: MODIS MCD43A1 Collection 6.1, daily 500 m BRDF model parameters,
// Band 1 (doi:10.5067/MODIS/MCD43A1.061).
// Method: Chappell and Webb (2016). The rescaling of omega_n and the
// friction velocity ratio are computed in odem.py, not here.

var YEAR = 2006;
var MONTHS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
var EXPORT_SCALE = 27830;  // metres, about 0.25 degree at the equator
var DRIVE_FOLDER = 'ODEM_BRDF';

// Kernel integrals of the black-sky albedo at nadir (solar zenith 0) for the
// Ross-Thick / Li-Sparse-Reciprocal model (Schaaf et al., 2002)
var G_VOL_NADIR = -0.007574;
var G_GEO_NADIR = -1.284909;

var exportRegion = ee.Geometry.BBox(-179.875, -60, 179.875, 90);

function computeBRDF(img) {
  // The collection stores scaled integers (scale factor 0.001)
  var f_iso = img.select('BRDF_Albedo_Parameters_Band1_iso').multiply(0.001);
  var f_vol = img.select('BRDF_Albedo_Parameters_Band1_vol').multiply(0.001);
  var f_geo = img.select('BRDF_Albedo_Parameters_Band1_geo').multiply(0.001);

  // Black-sky albedo at nadir
  var albedo = f_iso
    .add(f_vol.multiply(G_VOL_NADIR))
    .add(f_geo.multiply(G_GEO_NADIR))
    .clamp(0.001, 1)
    .rename('albedo');

  var shadow = ee.Image(1).subtract(albedo).rename('shadow');

  // Normalised shadow: omega_n = (1 - albedo) / f_iso
  var omega_n = shadow.divide(f_iso.max(0.001))
    .clamp(0, 50)
    .rename('omega_n');

  return albedo
    .addBands(shadow)
    .addBands(omega_n)
    .addBands(f_iso.rename('f_iso'))
    .copyProperties(img, ['system:time_start']);
}

// One export task per month; start each task from the Tasks tab
MONTHS.forEach(function(month) {
  var startDate = ee.Date.fromYMD(YEAR, month, 1);
  var endDate = startDate.advance(1, 'month');
  var monthStr = (month < 10) ? '0' + month : '' + month;

  var monthly = ee.ImageCollection('MODIS/061/MCD43A1')
    .filterDate(startDate, endDate)
    .map(computeBRDF)
    .mean()
    .unmask(0)
    .toFloat();

  Export.image.toDrive({
    image: monthly,
    description: 'brdf_' + YEAR + monthStr,
    folder: DRIVE_FOLDER,
    fileNamePrefix: 'brdf_' + YEAR + monthStr,
    region: exportRegion,
    scale: EXPORT_SCALE,
    crs: 'EPSG:4326',
    maxPixels: 1e10,
    fileFormat: 'GeoTIFF'
  });
});
