import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from shapely.geometry import box
from rasterio.transform import from_bounds

# Load shapefiles
region = gpd.read_file(r"C:\\Users\\gonca\\Ambiente de Trabalho\\Aramo\\area.shp")
labels = gpd.read_file(r"C:\\Users\\gonca\\Ambiente de Trabalho\\Aramo\\litologias.shp")

# Convert categorical labels to integers
labels["litologia_code"] = labels["litologia"].astype("category").cat.codes

label_mapping = dict(enumerate(labels["litologia"].astype("category").cat.categories))
print("Label mapping:", label_mapping)

pixel_size = 15  # meters

minx, miny, maxx, maxy = region.total_bounds

width = int(round((maxx - minx) / pixel_size))
height = int(round((maxy - miny) / pixel_size))
transform = from_bounds(minx, miny, maxx, maxy, width, height)

shapes = ((geom, value) for geom, value in zip(labels.geometry, labels["litologia_code"]))
label_raster = rasterize(
    shapes=shapes,
    out_shape=(height, width),
    transform=transform,
    fill=-1,
    dtype="int16"
)

region_mask = rasterize(
    [(geom, 1) for geom in region.geometry],
    out_shape=(height, width),
    transform=transform,
    fill=0,
    dtype="uint8"
)
label_raster[region_mask == 0] = -1

with rasterio.open(
    r"C:\\Users\\gonca\\Ambiente de Trabalho\\Aramo\\litologia.tif",
    "w",
    driver="GTiff",
    height=height,
    width=width,
    count=1,
    dtype="int16",
    crs=region.crs,
    transform=transform,
    nodata=-1,
) as dst:
    dst.write(label_raster, 1)