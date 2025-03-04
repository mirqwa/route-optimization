import argparse

import geopandas as gpd
import numpy as np

import utils


np.random.seed(32)


EPISODES = 10000
EPSILON = 0.2
DISCOUNT_FACTOR = 0.95
NO_OF_NEIGHBORS = 10


def initialize_returns(distances: np.ndarray) -> dict:
    returns = {}
    for state in range(distances.shape[0]):
        for action in range(distances.shape[1]):
            returns[(state, action)] = []
    return returns


def state_action_pair_exists_earlier(
    state_action: tuple, episode_results: list, current_time_step: int
) -> bool:
    if state_action in episode_results[:current_time_step]:
        return True
    return False


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
    policy = utils.initialize_policy(distances, NO_OF_NEIGHBORS)
    state_action_values = utils.initialize_state_action_values(
        cities_locations_gdf, policy
    )
    returns = initialize_returns(distances)

    for episode in range(EPISODES):
        print(f"Episode {episode + 1}")
        episode_results = utils.generate_episode(
            policy, start_city_index, end_city_index
        )
        G = 0
        current_time_step = len(episode_results) - 1
        for state, action in reversed(episode_results):
            G = DISCOUNT_FACTOR * G - distances[state][action]
            if state_action_pair_exists_earlier(
                (state, action), episode_results, current_time_step
            ):
                current_time_step -= 1
                continue
            multiple = episode_results[current_time_step:].count((state, action))
            returns[(state, action)].append(G * multiple)
            state_action_values[state][action] = sum(returns[(state, action)]) / len(
                returns[(state, action)]
            )
            action_with_max_value = np.argmax(state_action_values[state])
            policy = utils.update_policy(policy, EPSILON, state, action_with_max_value)
            current_time_step -= 1

    shortest_path, route = utils.get_shortest_path(
        state_action_values, start_city_index, end_city_index
    )

    route_distance = utils.get_distance(distances, route)
    shortest_path = [
        cities_locations_gdf["Label"][city_index] for city_index in shortest_path
    ]
    shortest_path = " -> ".join(shortest_path)

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
