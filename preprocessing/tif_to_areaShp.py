import geopandas as gpd
import rasterio
from shapely.geometry import box

base_raster = r"C:\\Users\\gonca\\Ambiente de Trabalho\Aramo\\EnMap_512.tif"
output_shapefile = r"C:\\Users\\gonca\\Ambiente de Trabalho\\Aramo\\area.shp"

with rasterio.open(base_raster) as src:
    raster_bounds = src.bounds
    raster_crs = src.crs

    # Create rectangular polygon from raster bounds
    raster_bbox = box(
        raster_bounds.left,
        raster_bounds.bottom,
        raster_bounds.right,
        raster_bounds.top
    )

    area_gdf = gpd.GeoDataFrame(geometry=[raster_bbox], crs=raster_crs)
    area_gdf.to_file(output_shapefile)

print(f"Created matching area shapefile at: {output_shapefile}")
print(f"CRS: {raster_crs}")
print(f"Bounds: {raster_bounds}")