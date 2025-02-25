import argparse

import geopandas as gpd
import numpy as np

import utils


EPISODES = 10000
NO_OF_NEIGHBORS = 10


def get_optimal_path(
    cities_locations_gdf: gpd.GeoDataFrame, distances: np.ndarray
) -> tuple:
    policy = utils.initialize_policy(distances, NO_OF_NEIGHBORS)
    state_action_values = utils.initialize_state_action_values(
        cities_locations_gdf, policy
    )

    for episode in range(EPISODES):
        pass


def main(api_key: str) -> None:
    cities_locations_gdf, distances = utils.get_training_data(api_key)
    utils.plot_cities(cities_locations_gdf)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_key", type=str, required=True)

    args = parser.parse_args()
    main(args.api_key)
