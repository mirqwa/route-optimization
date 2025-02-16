import argparse
import random
from pathlib import Path

import contextily as cx
import geopandas as gpd
import gmaps
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from googlemaps import Client

import constants


def plot_cities(cities_gdf: gpd.GeoDataFrame, path: list = []) -> None:
    _, ax = plt.subplots(1, figsize=(15, 15))
    cities_gdf["markersize"] = np.where(
        cities_gdf["Label"].isin(["Nairobi", "Kampala"]), 150, 50
    )
    cities_gdf["color"] = np.where(
        cities_gdf["Label"].isin(["Nairobi", "Kampala"]), "red", "purple"
    )

    cities_gdf.plot(
        ax=ax, color=cities_gdf["color"], markersize=cities_gdf["markersize"], alpha=0.5
    )

    # Add basemap
    cx.add_basemap(
        ax, crs=cities_gdf.crs, zoom=8, source=cx.providers.OpenStreetMap.Mapnik
    )

    for lon, lat, label in zip(
        cities_gdf.geometry.x, cities_gdf.geometry.y, cities_gdf.Label
    ):
        city_label = constants.CITIES_TO_LABEL.get(label)
        if city_label:
            ax.annotate(
                city_label,
                xy=(lon, lat),
                xytext=(-20, -20),
                textcoords="offset points",
                size=15,
                color="blue",
            )
    ax.set_axis_off()
    plt.show()


def get_gmaps_client(api_key: str) -> Client:
    gmaps.configure(api_key=api_key)
    g_maps_client = Client(key=api_key)
    return g_maps_client


def get_coordinates_for_available_city(
    cities_coordinates: gpd.GeoDataFrame, city: str
) -> dict:
    city_gdf = cities_coordinates[cities_coordinates["Label"] == city].reset_index()
    location = city_gdf.iloc[0].to_dict()
    del location["index"]
    del location["geometry"]
    return location


def get_cities_coordinates(
    g_maps_client, use_saved_coordinates=False
) -> gpd.GeoDataFrame:
    file_name = "data/east_africa/cities.geojson"
    cities_coordinates = gpd.read_file(file_name) if Path(file_name).is_file() else None
    if use_saved_coordinates and cities_coordinates is not None:
        return cities_coordinates
    available_cities = (
        cities_coordinates["Label"].tolist() if cities_coordinates is not None else []
    )
    cities_locations = []
    for city in constants.CITIES:
        location = {"Label": city}
        if city in available_cities:
            location = get_coordinates_for_available_city(cities_coordinates, city)
            cities_locations.append(location)
            continue
        geocode_result = g_maps_client.geocode(city)
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
        g_maps_client, use_saved_coordinates=False
    )
    plot_cities(cities_locations_gdf)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_key", type=str, required=True)

    args = parser.parse_args()
    main(args.api_key)
