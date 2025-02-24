import argparse
from pathlib import Path

import contextily as cx
import geopandas as gpd
import gmaps
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from googlemaps import Client

import constants


def initialize_q_table(distances: np.ndarray, no_of_neighbors: int) -> np.ndarray:
    q_table = np.zeros((distances.shape[0], distances.shape[0]))
    for city in range(distances.shape[0]):
        city_distances = pd.Series(distances[city, :])
        min_distance = city_distances.sort_values().to_list()[no_of_neighbors]
        possible_actions = np.where(distances[city, :] < min_distance)[0]
        for action in range(distances.shape[1]):
            q_table[city][action] = 0 if action in possible_actions else -float("inf")
    return q_table


def select_next_action(
    distances: np.ndarray,
    current_city: int,
    q_table: np.ndarray,
    no_of_neighbors: int,
    epsilon: float,
) -> int:
    city_distances = pd.Series(distances[current_city, :])
    min_distance = city_distances.sort_values().to_list()[no_of_neighbors]
    possible_actions = (
        np.where(distances[current_city, :] < min_distance)[0]  # exploration
        if np.random.uniform(0, 1) < epsilon
        else np.where(
            q_table[current_city, :] == np.max(q_table[current_city, :])  # exploitation
        )[0]
    )
    if len(possible_actions) == 0:
        return
    return np.random.choice(possible_actions)


def annotate_route(
    ax: plt.Axes, data_gdf: gpd.GeoDataFrame, i: int, j: int, color: str
) -> None:
    arrowprops = dict(arrowstyle="->", connectionstyle="arc3", edgecolor=color)
    ax.annotate(
        "",
        xy=[data_gdf.iloc[j].geometry.x, data_gdf.iloc[j].geometry.y],
        xytext=[data_gdf.iloc[i].geometry.x, data_gdf.iloc[i].geometry.y],
        arrowprops=arrowprops,
    )


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
    for i, j in path:
        annotate_route(ax, cities_gdf, i, j, "darkblue")
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


def get_intercity_distances(
    cities_locations_gdf: gpd.GeoDataFrame, g_maps_client, use_saved_distances=False
) -> np.ndarray:
    file_name = f"data/east_africa/distances.csv"
    cities_distances = pd.read_csv(file_name) if Path(file_name).is_file() else None
    if use_saved_distances and cities_distances is not None:
        return cities_distances.values[:, 1:]
    distances = np.zeros((len(cities_locations_gdf), len(cities_locations_gdf)))
    cities_locations_gdf["coord"] = (
        cities_locations_gdf.lat.astype(str)
        + ","
        + cities_locations_gdf.lng.astype(str)
    )
    available_cities = (
        cities_distances["Label"].tolist() if cities_distances is not None else []
    )
    for lat in range(len(cities_locations_gdf)):
        for lon in range(len(cities_locations_gdf)):
            origin = cities_locations_gdf["Label"][lat]
            destination = cities_locations_gdf["Label"][lon]
            if origin in available_cities:
                city_distances = cities_distances[
                    cities_distances["Label"] == origin
                ].reset_index()
                city_distances = city_distances.iloc[0].to_dict()
                distance = city_distances.get(destination)
                if distance is not None:
                    distances[lat][lon] = distance
                    print(
                        f"Distance for {origin} -> {destination} already fetched, skipping"
                    )
                    continue
            print(f"Getting distance for {origin} -> {destination}")
            maps_api_result = g_maps_client.directions(
                cities_locations_gdf["coord"].iloc[lat],
                cities_locations_gdf["coord"].iloc[lon],
                mode="driving",
            )
            distances[lat][lon] = maps_api_result[0]["legs"][0]["distance"]["value"]
    distances_df = pd.DataFrame(
        data=distances,
        columns=cities_locations_gdf["Label"],
        index=cities_locations_gdf["Label"],
    )
    distances_df.to_csv(file_name)
    return distances


def get_shortest_path(
    state_action_values: np.ndarray, start_state: int, end_state: int
) -> list:
    shortest_path = [start_state]
    current_state = start_state
    while current_state != end_state:
        next_state = np.argmax(state_action_values[current_state, :])
        shortest_path.append(next_state)
        current_state = next_state
    route = [(start, dest) for start, dest in zip(shortest_path, shortest_path[1:])]
    return shortest_path, route


def get_distance(distances: np.array, route: list) -> int:
    route_distance = 0
    for origin, destination in route:
        route_distance += distances[origin][destination]
    return int(route_distance)


def get_training_data(api_key: str) -> tuple:
    g_maps_client = get_gmaps_client(api_key)
    cities_locations_gdf = get_cities_coordinates(
        g_maps_client, use_saved_coordinates=True
    )
    distances = get_intercity_distances(
        cities_locations_gdf, g_maps_client, use_saved_distances=True
    )
    distances = distances / 1000
    distances = np.where(distances == 0, float("inf"), distances)
    return cities_locations_gdf, distances


def get_optimal_path_and_distance(
    cities_locations_gdf: gpd.GeoDataFrame,
    distances: np.ndarray,
    q_table: np.ndarray,
    start_city_index: int,
    end_city_index: str,
    file_name: str,
) -> tuple:
    q_table_df = pd.DataFrame(
        data=q_table,
        index=cities_locations_gdf["Label"],
        columns=cities_locations_gdf["Label"],
    )
    q_table_df.to_csv(
        file_name
    )
    shortest_path, route = get_shortest_path(
        q_table, start_city_index, end_city_index
    )
    route_distance = get_distance(distances, route)
    print("The route distance", route_distance)
    shortest_path = [
        cities_locations_gdf["Label"][city_index] for city_index in shortest_path
    ]
    shortest_path = " -> ".join(shortest_path)
    return shortest_path, route


def main(api_key: str) -> None:
    g_maps_client = get_gmaps_client(api_key)
    cities_locations_gdf = get_cities_coordinates(
        g_maps_client, use_saved_coordinates=False
    )
    plot_cities(cities_locations_gdf)
    distances = get_intercity_distances(
        cities_locations_gdf, g_maps_client, use_saved_distances=False
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_key", type=str, required=True)

    args = parser.parse_args()
    main(args.api_key)
