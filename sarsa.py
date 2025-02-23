import argparse

import geopandas as gpd
import numpy as np

import utils


EPSILON = 0.2
LEARNING_RATE = 0.01
DISCOUNT_FACTOR = 0.9
NO_OF_NEIGHBORS = 10
MAX_STEPS = 500


def update_q_table(
    q_table: np.ndarray,
    distances: np.ndarray,
    current_city: int,
    action: int,
    next_city: int,
    next_action: list,
) -> None:
    # the reward is negative since the goal is to have minimum distance
    reward = -distances[current_city, next_city]
    current_state_action_value = q_table[current_city, action]
    next_state_action_value = q_table[next_city][next_action]

    q_table[current_city, action] = (
        1 - LEARNING_RATE
    ) * current_state_action_value + LEARNING_RATE * (
        reward + DISCOUNT_FACTOR * next_state_action_value
    )


def train_agent(
    cities_locations_gdf: gpd.GeoDataFrame,
    num_episodes: int,
    start_city_index: str,
    end_city_index: str,
    distances: np.ndarray,
) -> np.ndarray:
    q_table = utils.initialize_q_table(distances, NO_OF_NEIGHBORS)
    for episode in range(num_episodes):
        print(f"Episode {episode + 1}")
        current_city = start_city_index
        action = utils.select_next_action(
            distances, current_city, q_table, NO_OF_NEIGHBORS, EPSILON
        )
        steps = 0
        while current_city != end_city_index and steps <= MAX_STEPS:
            steps += 1
            next_city = action
            next_action = utils.select_next_action(
                distances, next_city, q_table, NO_OF_NEIGHBORS, EPSILON
            )
            update_q_table(
                q_table, distances, current_city, action, next_city, next_action
            )
            current_city = next_city
            action = next_action


def get_optimal_path(
    cities_locations_gdf: gpd.GeoDataFrame,
    distances: np.ndarray,
    start_city: str,
    end_city: str,
) -> list:
    start_city_index = cities_locations_gdf[
        cities_locations_gdf["Label"] == start_city
    ].index[0]
    end_city_index = cities_locations_gdf[
        cities_locations_gdf["Label"] == end_city
    ].index[0]
    train_agent(cities_locations_gdf, 1000, start_city_index, end_city_index, distances)
    shortest_path = None
    route = None
    return shortest_path, route


def main(api_key: str):
    g_maps_client = utils.get_gmaps_client(api_key)
    cities_locations_gdf = utils.get_cities_coordinates(
        g_maps_client, use_saved_coordinates=True
    )
    # utils.plot_cities(cities_locations_gdf)
    distances = utils.get_intercity_distances(
        cities_locations_gdf, g_maps_client, use_saved_distances=True
    )
    distances = distances / 1000
    distances = np.where(distances == 0, float("inf"), distances)
    shortest_path, route = get_optimal_path(
        cities_locations_gdf, distances, "Nairobi", "Kampala"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_key", type=str, required=True)

    args = parser.parse_args()
    main(args.api_key)
