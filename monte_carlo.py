import argparse
from collections import defaultdict

import geopandas as gpd
import numpy as np

import utils


np.random.seed(32)


EPISODES = 100


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


def select_action(policy: dict, current_state: int) -> int:
    actions = [list(state_policy.keys())[0] for state_policy in policy[current_state]]
    probs = [list(state_policy.values())[0] for state_policy in policy[current_state]]
    action = np.random.choice(actions, 1, replace=False, p=probs)[0]
    next_state = action
    return action, next_state


def generate_episode(policy: dict, origin: int, destination: int) -> list:
    episode_results = []
    current_state = origin
    while current_state != destination:
        action, next_state = select_action(policy, current_state)
        episode_results.append({current_state: action})
        current_state = next_state
    return episode_results


def get_optimal_path(
    cities_locations_gdf: gpd.GeoDataFrame, distances: np.ndarray
) -> tuple:
    policy = initialize_policy(distances)
    state_action_values = initialize_state_action_values(cities_locations_gdf)
    returns = initialize_returns(distances)

    shortest_path = None
    route = None

    for episode in range(EPISODES):
        print(f"Episode {episode + 1}")
        episode_results = generate_episode(policy, 0, 15)

    return shortest_path, route


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
    shortest_path, route = get_optimal_path(cities_locations_gdf, distances)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_key", type=str, required=True)

    args = parser.parse_args()
    main(args.api_key)
