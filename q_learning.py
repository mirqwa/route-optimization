import argparse
from collections import defaultdict

import geopandas as gpd
import numpy as np
import pandas as pd

import utils


np.random.seed(0)

EPSILON = 0.2
LEARNING_RATE = 0.01
DISCOUNT_FACTOR = 0.9
EPISODES = 10000
MAX_STEPS = 500
NO_OF_NEIGHBORS = 10


def update_q_table(
    q_table: np.ndarray,
    distances: np.ndarray,
    current_city: int,
    action: int,
    next_city: int,
    visited_cities: list,
) -> None:
    # the reward is negative since the goal is to have minimum distance
    reward = (
        -1000
        if visited_cities.count((current_city, action)) > 1
        else -distances[current_city, next_city]
    )
    current_state_action_value = q_table[current_city, action]
    next_state_action_value = np.max(q_table[next_city, :])

    q_table[current_city, action] = (
        1 - LEARNING_RATE
    ) * current_state_action_value + LEARNING_RATE * (
        reward + DISCOUNT_FACTOR * next_state_action_value
    )


def get_q_learning_cost_table(
    cities_locations_gdf: gpd.GeoDataFrame,
    num_episodes: int,
    start_city_index: str,
    end_city_index: str,
    distances: np.ndarray,
) -> np.ndarray:
    q_table = utils.initialize_q_table(distances, NO_OF_NEIGHBORS)
    number_of_visits = defaultdict(int)
    for epidode in range(num_episodes):
        print(f"Episode {epidode + 1}")
        visited_cities = []
        current_city = start_city_index
        steps = 0
        while current_city != end_city_index and steps <= MAX_STEPS:
            steps += 1
            action = utils.select_next_action(
                distances, current_city, q_table, NO_OF_NEIGHBORS, EPSILON
            )
            visited_cities.append((current_city, action))
            if action is None:
                break
            next_city = action
            update_q_table(
                q_table, distances, current_city, action, next_city, visited_cities
            )

            current_city = next_city
            if current_city == end_city_index:
                break
        for city, action in visited_cities:
            number_of_visits[cities_locations_gdf.iloc[city].to_dict()["Label"]] += 1
    # print(dict(number_of_visits))
    return q_table


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
    q_table = get_q_learning_cost_table(
        cities_locations_gdf, EPISODES, start_city_index, end_city_index, distances
    )

    shortest_path, route = utils.get_optimal_path_and_distance(
        cities_locations_gdf,
        distances,
        q_table,
        start_city_index,
        end_city_index,
        f"data/east_africa/{start_city}_{end_city}_q_learning_q_table_{EPISODES}.csv",
    )
    return shortest_path, route


def main(api_key: str) -> None:
    cities_locations_gdf, distances = utils.get_training_data(api_key)
    utils.plot_cities(cities_locations_gdf)
    shortest_path, route = get_optimal_path(
        cities_locations_gdf, distances, "Nairobi", "Kampala"
    )
    print(shortest_path)
    print(route)
    utils.plot_cities(cities_locations_gdf, route)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_key", type=str, required=True)

    args = parser.parse_args()
    main(args.api_key)
