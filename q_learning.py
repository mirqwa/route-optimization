import argparse

import geopandas as gpd
import numpy as np
import pandas as pd

import constants
import utils


def select_next_action(
    distances: np.ndarray, current_city: int, q_table: np.ndarray
) -> int:
    possible_actions = (
        np.where(distances[current_city, :] > 0)[0]  # exploration
        if np.random.uniform(0, 1) < constants.EPSILON
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
) -> None:
    # the reward is negative since the goal is to have minimum distance
    reward = -distances[current_city, next_city]
    current_state_action_value = q_table[current_city, action]
    next_state_action_value = np.max(q_table[next_city, :])

    q_table[current_city, action] = (
        1 - constants.LEARNING_RATE
    ) * current_state_action_value + constants.LEARNING_RATE * (
        reward + constants.DISCOUNT_FACTOR * next_state_action_value
    )


def get_q_learning_cost_table(
    cities_locations_gdf: gpd.GeoDataFrame,
    num_episodes: int,
    start_city_index: str,
    end_city_index: str,
    distances: np.ndarray,
) -> np.ndarray:
    q_table = np.zeros((cities_locations_gdf.shape[0], cities_locations_gdf.shape[0]))
    for epidode in range(num_episodes):
        print(f"Episode {epidode + 1}")
        current_city = start_city_index
        while current_city != end_city_index:
            action = select_next_action(distances, current_city, q_table)
            if action is None:
                break
            next_city = action
            update_q_table(q_table, distances, current_city, action, next_city)

            current_city = next_city
            if current_city == end_city_index:
                break
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
        cities_locations_gdf, 1000, start_city_index, end_city_index, distances
    )
    q_table_df = pd.DataFrame(
        data=q_table,
        index=cities_locations_gdf["Label"],
        columns=cities_locations_gdf["Label"],
    )
    q_table_df.to_csv(f"data/east_africa/{start_city}_{end_city}_q_table.csv")
    shortest_path, route = utils.get_shortest_path(
        q_table, start_city_index, end_city_index
    )
    route_distance = utils.get_distance(distances, route)
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
