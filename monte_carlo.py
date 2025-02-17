import argparse
from collections import defaultdict

import geopandas as gpd
import numpy as np

import utils


def initialize_policy(distances: np.ndarray) -> dict:
    policy = defaultdict(list)
    for state in range(distances.shape[0]):
        for action in range(distances.shape[1]):
            if state == action:
                continue
            policy[state].append({action: 1 / (distances.shape[1] - 1)})
    return dict(policy)


def initialize_state_action_values(
    cities_locations_gdf: gpd.GeoDataFrame,
) -> np.ndarray:
    state_action_values = np.zeros(
        (cities_locations_gdf.shape[0], cities_locations_gdf.shape[0])
    )
    return state_action_values


def initialize_returns(distances: np.ndarray) -> dict:
    returns = {}
    for state in range(distances.shape[0]):
        returns[state] = []
    return returns


def main(api_key: str) -> None:
    g_maps_client = utils.get_gmaps_client(api_key)
    cities_locations_gdf = utils.get_cities_coordinates(
        g_maps_client, use_saved_coordinates=True
    )
    utils.plot_cities(cities_locations_gdf)
    distances = utils.get_intercity_distances(
        cities_locations_gdf, g_maps_client, use_saved_distances=True
    )
    distances = distances / 1000
    distances = np.where(distances == 0, float("inf"), distances)
    policy = initialize_policy(distances)
    state_action_values = initialize_state_action_values(cities_locations_gdf)
    returns = initialize_returns(distances)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_key", type=str, required=True)

    args = parser.parse_args()
    main(args.api_key)
