import argparse
import random
from pathlib import Path

import geopandas as gpd
import gmaps
import pandas as pd

from googlemaps import Client

import constants


def get_gmaps_client(api_key: str) -> Client:
    gmaps.configure(api_key=api_key)
    g_maps_client = Client(key=api_key)
    return g_maps_client


def get_cities_coordinates(
    g_maps_client, use_saved_coordinates=False
) -> gpd.GeoDataFrame:
    file_name = "data/east_africa/cities.geojson"
    if Path(file_name).is_file() and use_saved_coordinates:
        return gpd.read_file(file_name)
    cities_locations = []
    for city in constants.CITIES:
        geocode_result = g_maps_client.geocode(city)
        location = {"Label": city}
        try:
            location.update(geocode_result[0]["geometry"]["location"])
            cities_locations.append(location)
        except Exception:
            print(f"Failed to get coordinates for {city}")
    cities_locations_df = pd.DataFrame(cities_locations)
    cities_locations_gdf = gpd.GeoDataFrame(
        cities_locations_df,
        geometry=gpd.points_from_xy(
            cities_locations_df["lng"], cities_locations_df["lat"]
        ),
        crs="EPSG:4326",
    )
    cities_locations_gdf.to_file(file_name)
    return cities_locations_gdf


def main(api_key: str) -> None:
    g_maps_client = get_gmaps_client(api_key)
    cities_locations_gdf = get_cities_coordinates(
        g_maps_client, use_saved_coordinates=True
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_key", type=str, required=True)

    args = parser.parse_args()
    main(args.api_key)
