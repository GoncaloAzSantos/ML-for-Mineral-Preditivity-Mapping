import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_bounds
from scipy.ndimage import uniform_filter
from shapely.geometry import Point, Polygon

# Load shapefiles
region = gpd.read_file(r"C:\\Users\\gonca\\Ambiente de Trabalho\\Aramo\\area.shp") #Area of study
lines = gpd.read_file(r"C:\\Users\\gonca\\Ambiente de Trabalho\\Aramo\\faults.shp") #Shapefile composed of lines

pixel_size = 15  # meters
minx, miny, maxx, maxy = region.total_bounds
width = 512
height = 512
transform = from_bounds(minx, miny, maxx, maxy, width, height)

line_raster = rasterize(
    [(geom, 1) for geom in lines.geometry],
    out_shape=(height, width),
    transform=transform,
    fill=0,
    dtype="uint8"
)

# Define moving window size
window_m = 500
window_px = int(window_m / pixel_size)
if window_px % 2 == 0:
    window_px += 1

line_sum = uniform_filter(line_raster.astype(float), size=window_px, mode='constant') * (window_px ** 2)
line_length = line_sum * pixel_size

# Calculate density: meters per km²
window_area_m2 = (window_px * pixel_size) ** 2
density = line_length / window_area_m2 * 1e6  # meters per km²

# Mask outside region
region_mask = rasterize(
    [(geom, 1) for geom in region.geometry],
    out_shape=(height, width),
    transform=transform,
    fill=0,
    dtype="uint8"
)
density[region_mask == 0] = np.nan

output_base = r"C:\\Users\\gonca\Ambiente de Trabalho\\Aramo512\\faults"
with rasterio.open(
        f"{output_base}.tif",
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype="float32",
        crs=region.crs,
        transform=transform,
        nodata=np.nan,
) as dst:
    dst.write(density.astype("float32"), 1)