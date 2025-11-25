import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from shapely.geometry import box
import numpy as np
from rasterio.transform import from_bounds
from scipy.ndimage import distance_transform_edt

region = gpd.read_file(r"C:\\Users\\gonca\\Ambiente de Trabalho\\Aramo_base\\area.shp")  # Polygon area
lines = gpd.read_file(r"C:\\Users\\gonca\\Ambiente de Trabalho\\Aramo_base\\contact.shp")  # Line shapefile

# Define resolution
pixel_size = 15  # meters

minx, miny, maxx, maxy = region.total_bounds

width = int(round((maxx - minx) / pixel_size))
height = int(round((maxy - miny) / pixel_size))
transform = from_bounds(minx, miny, maxx, maxy, width, height)

# Create buffer around the line
buffer_distance = 1500  # meters, adjust as needed
lines_buffered = lines.buffer(buffer_distance)

# Rasterize the buffer (1 inside buffer, 0 outside)
buffer_raster = rasterize(
    [(geom, 1) for geom in lines_buffered],
    out_shape=(height, width),
    transform=transform,
    fill=0,
    dtype="uint8"
)

# Rasterize the line itself (1 on line, 0 elsewhere)
line_raster = rasterize(
    [(geom, 1) for geom in lines.geometry],
    out_shape=(height, width),
    transform=transform,
    fill=0,
    dtype="uint8"
)

# Calculate distance from each pixel in buffer to the nearest line pixel
# Invert line_raster: 0 where line, 1 elsewhere
distance = distance_transform_edt(1 - line_raster) * pixel_size

# Mask: only keep distances inside the buffer, set outside to 0
distance[buffer_raster == 0] = 0

# Optional: Mask everything outside the study region
region_mask = rasterize(
    [(geom, 1) for geom in region.geometry],
    out_shape=(height, width),
    transform=transform,
    fill=0,
    dtype="uint8"
)
distance[region_mask == 0] = np.nan # Set outside region to NaN

with rasterio.open(
    r"C:\\Users\\gonca\\Ambiente de Trabalho\\Aramo\\Contact_Buffer_Distance.tif",
    "w",
    driver="GTiff",
    height=height,
    width=width,
    count=1,
    dtype="float32",
    crs=region.crs,
    transform=transform,
    nodata=0,
) as dst:
    dst.write(distance.astype("float32"), 1)