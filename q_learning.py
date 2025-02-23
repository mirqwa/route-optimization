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
EPIDODES = 50000
MAX_STEPS = 200
NO_OF_NEIGHBORS = 10


def initialize_q_table(distances: np.ndarray) -> np.ndarray:
    q_table = np.zeros((distances.shape[0], distances.shape[0]))
    for city in range(distances.shape[0]):
        city_distances = pd.Series(distances[city, :])
        min_distance = city_distances.sort_values().to_list()[NO_OF_NEIGHBORS]
        possible_actions = np.where(distances[city, :] < min_distance)[0]
        for action in range(distances.shape[1]):
            q_table[city][action] = 0 if action in possible_actions else -float("inf")
    return q_table


def select_next_action(
    distances: np.ndarray, current_city: int, q_table: np.ndarray
) -> int:
    city_distances = pd.Series(distances[current_city, :])
    min_distance = city_distances.sort_values().to_list()[NO_OF_NEIGHBORS]
    possible_actions = (
        np.where(distances[current_city, :] < min_distance)[0]  # exploration
        if np.random.uniform(0, 1) < EPSILON
        else np.where(
            q_table[current_city, :] == np.max(q_table[current_city, :])  # exploitation
        )[0]
    )
    if len(possible_actions) == 0:
        return
    return np.random.choice(possible_actions)


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
    # reward = -distances[
    #     current_city, next_city
    # ]  * visited_cities.count((current_city, action))
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
    q_table = initialize_q_table(distances)
    # q_table = np.zeros((cities_locations_gdf.shape[0], cities_locations_gdf.shape[0]))
    number_of_visits = defaultdict(int)
    for epidode in range(num_episodes):
        print(f"Episode {epidode + 1}")
        visited_cities = []
        current_city = start_city_index
        steps = 0
        while current_city != end_city_index and steps <= MAX_STEPS:
            steps += 1
            action = select_next_action(distances, current_city, q_table)
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
        cities_locations_gdf, EPIDODES, start_city_index, end_city_index, distances
    )
    q_table_df = pd.DataFrame(
        data=q_table,
        index=cities_locations_gdf["Label"],
        columns=cities_locations_gdf["Label"],
    )
    q_table_df = q_table_df[["Nairobi", "Kampala", "Mau Summit", "Londiani Junction"]]
    q_table_df.to_csv(
        f"data/east_africa/{start_city}_{end_city}_q_table_{EPIDODES}.csv"
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


def main(api_key: str) -> None:
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
    print(shortest_path)
    print(route)
    utils.plot_cities(cities_locations_gdf, route)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_key", type=str, required=True)

    args = parser.parse_args()
    main(args.api_key)
