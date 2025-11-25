import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_bounds

area = gpd.read_file(r"C:\\Users\\gonca\\Ambiente de Trabalho\\Aramo\\area.shp")
labels = gpd.read_file(r"C:\\Users\\OneDrive\\Ambiente de Trabalho\\Aramo\\mask.shp")

pixel_size = 15  # meters

minx, miny, maxx, maxy = area.total_bounds
width = int(round((maxx - minx) / pixel_size))
height = int(round((maxy - miny) / pixel_size))
transform = from_bounds(minx, miny, maxx, maxy, width, height)

shapes = ((geom, value) for geom, value in zip(labels.geometry, labels["id"]))
label_raster = rasterize(
    shapes=shapes,
    out_shape=(height, width),
    transform=transform,
    fill=-1,
    dtype="int16"
)

area_mask = rasterize(
    [(geom, 1) for geom in area.geometry],
    out_shape=(height, width),
    transform=transform,
    fill=0,
    dtype="uint8"
)
label_raster[area_mask == 0] = -1

with rasterio.open(
    r"C:\\Users\\gonca\\Ambiente de Trabalho\\Aramo\\mask.tif",
    "w",
    driver="GTiff",
    height=height,
    width=width,
    count=1,
    dtype="int16",
    crs=area.crs,
    transform=transform,
    nodata=-1,
) as dst:
    dst.write(label_raster, 1)