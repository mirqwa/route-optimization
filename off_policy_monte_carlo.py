import argparse

import geopandas as gpd
import numpy as np

import utils


EPISODES = 10000
NO_OF_NEIGHBORS = 10


def get_optimal_path(
    cities_locations_gdf: gpd.GeoDataFrame,
    distances: np.ndarray,
    start_city: str,
    end_city: str,
) -> tuple:
    start_city_index = cities_locations_gdf[
        cities_locations_gdf["Label"] == start_city
    ].index[0]
    end_city_index = cities_locations_gdf[
        cities_locations_gdf["Label"] == end_city
    ].index[0]
    behavior_policy = utils.initialize_policy(distances, NO_OF_NEIGHBORS)
    target_policy = utils.initialize_policy(distances, NO_OF_NEIGHBORS)
    state_action_values = utils.initialize_state_action_values(
        cities_locations_gdf, behavior_policy
    )
    C = np.zeros((distances.shape[0], distances.shape[1]))

    for episode in range(EPISODES):
        print(f"Episode {episode + 1}")

    shortest_path = route = None
    return shortest_path, route


def main(api_key: str) -> None:
    cities_locations_gdf, distances = utils.get_training_data(api_key)
    # utils.plot_cities(cities_locations_gdf)
    shortest_path, route = get_optimal_path(
        cities_locations_gdf, distances, "Nairobi", "Kampala"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_key", type=str, required=True)

    args = parser.parse_args()
    main(args.api_key)
