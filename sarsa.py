import argparse

import geopandas as gpd
import numpy as np
import pandas as pd

import utils


EPSILON = 0.2
LEARNING_RATE = 0.01
DISCOUNT_FACTOR = 0.8
NO_OF_NEIGHBORS = 10
MAX_STEPS = 500
EPISODES = 10000


def update_q_table(
    q_table: np.ndarray,
    distances: np.ndarray,
    current_city: int,
    action: int,
    next_city: int,
    next_action: list,
    visited_cities: list,
) -> None:
    # the reward is negative since the goal is to have minimum distance
    reward = (
        -1000
        if visited_cities.count((current_city, action)) > 1
        else -distances[current_city, next_city]
    )
    current_state_action_value = q_table[current_city, action]
    next_state_action_value = q_table[next_city][next_action]

    q_table[current_city, action] = (
        1 - LEARNING_RATE
    ) * current_state_action_value + LEARNING_RATE * (
        reward + DISCOUNT_FACTOR * next_state_action_value
    )


def train_agent(
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
        visited_cities = []
        while current_city != end_city_index and steps <= MAX_STEPS:
            steps += 1
            visited_cities.append((current_city, action))
            next_city = action
            next_action = utils.select_next_action(
                distances, next_city, q_table, NO_OF_NEIGHBORS, EPSILON
            )
            update_q_table(
                q_table, distances, current_city, action, next_city, next_action, visited_cities
            )
            current_city = next_city
            action = next_action
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
    q_table = train_agent(EPISODES, start_city_index, end_city_index, distances)
    q_table_df = pd.DataFrame(
        data=q_table,
        index=cities_locations_gdf["Label"],
        columns=cities_locations_gdf["Label"],
    )
    q_table_df.to_csv(
        f"data/east_africa/{start_city}_{end_city}_q_table_{EPISODES}.csv"
    )
    shortest_path, route = utils.get_shortest_path(
        q_table, start_city_index, end_city_index
    )
    route_distance = utils.get_distance(distances, route)
    print("The route distance", route_distance)
    shortest_path = [
        cities_locations_gdf["Label"][city_index] for city_index in shortest_path
    ]
    shortest_path = " -> ".join(shortest_path)
    return shortest_path, route


def main(api_key: str):
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
