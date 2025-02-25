import argparse

import geopandas as gpd
import numpy as np

import utils


np.random.seed(0)

EPISODES = 100000
EPSILON = 0.1
DISCOUNT_FACTOR = 0.95
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
        episode_results = utils.generate_episode(
            behavior_policy, start_city_index, end_city_index
        )
        G = 0
        W = 1
        current_time_step = len(episode_results) - 1
        for state, action in reversed(episode_results):
            multiple = episode_results[current_time_step:].count((state, action))
            reward = -1000 if multiple > 1 else -distances[state][action]
            G = DISCOUNT_FACTOR * G + reward
            C[state][action] += W
            state_action_values[state][action] = (
                1 - W / C[state][action]
            ) * state_action_values[state][action] + (W / C[state][action]) * G
            action_with_max_value = np.argmax(state_action_values[state])
            behavior_policy = utils.update_policy(
                behavior_policy, EPSILON, state, action_with_max_value
            )
            target_policy = utils.update_policy(
                target_policy, 0, state, action_with_max_value
            )
            if action != action_with_max_value:
                break
            state_probs = {}
            for action_prob in behavior_policy[state]:
                state_probs.update(action_prob)
            W = W / state_probs[action]
            current_time_step -= 1

    shortest_path, route = utils.get_shortest_path(
        state_action_values, start_city_index, end_city_index
    )

    route_distance = utils.get_distance(distances, route)
    print(route_distance)
    shortest_path = [
        cities_locations_gdf["Label"][city_index] for city_index in shortest_path
    ]
    shortest_path = " -> ".join(shortest_path)
    return shortest_path, route


def main(api_key: str) -> None:
    cities_locations_gdf, distances = utils.get_training_data(api_key)
    # utils.plot_cities(cities_locations_gdf)
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
