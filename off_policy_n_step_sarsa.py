import argparse

import geopandas as gpd
import numpy as np

import utils


np.random.seed(0)

EPSILON = 0.2
LEARNING_RATE = 0.01
DISCOUNT_FACTOR = 0.9
EPISODES = 10000
NO_OF_NEIGHBORS = 10
MAX_STEPS = 1500
N_STEPS = 3


def get_importance_sampling_ratio(
    q_table: np.ndarray, states: list, actions: list
) -> float:
    pass


def update_q_table(
    q_table: np.ndarray,
    states: list,
    actions: list,
    rewards: list,
    tau: int,
    n: int,
    t: int,
    T: int,
) -> np.ndarray:
    current_states = [states[i] for i in range(tau, min(tau + n - 1, T - 1))]
    current_actions = [actions[i] for i in range(tau, min(tau + n - 1, T - 1))]
    importance_sampling_ratio = get_importance_sampling_ratio(
        q_table, current_states, current_actions
    )
    G = sum(
        [DISCOUNT_FACTOR ** (i - tau) * rewards[i] for i in range(tau, min(tau + n, T))]
    )
    G = (
        G + DISCOUNT_FACTOR**n * q_table[states[tau + n]][actions[tau + n]]
        if t + n < T
        else G
    )

    q_table[states[tau]][actions[tau]] = (1 - LEARNING_RATE) * q_table[states[tau]][
        actions[tau]
    ] + LEARNING_RATE * G
    return q_table


def train_agent(
    num_episodes: int,
    start_city_index: str,
    end_city_index: str,
    distances: np.ndarray,
    n: int,
) -> np.ndarray:
    behavior_policy = utils.initialize_policy(distances, NO_OF_NEIGHBORS)
    target_policy = utils.initialize_policy(distances, NO_OF_NEIGHBORS)
    q_table = utils.initialize_q_table(distances, NO_OF_NEIGHBORS)
    for episode in range(num_episodes):
        print(f"Episode {episode + 1}")
        current_city = start_city_index
        states = [current_city]
        rewards = []
        action, _ = utils.select_action_from_policy(behavior_policy, current_city)
        actions = [action]
        visited_cities = []
        T = float("inf")
        for t in range(MAX_STEPS):
            if t < T:
                visited_cities.append((current_city, action))
                reward = (
                    -1000
                    if visited_cities.count((current_city, action)) > 1
                    else -distances[current_city, action]
                )
                current_city = action
                states.append(current_city)
                rewards.append(reward)
                if current_city == end_city_index:
                    T = t + 1
                else:
                    next_action, _ = utils.select_action_from_policy(
                        behavior_policy, current_city
                    )
                    actions.append(next_action)
                    action = next_action
            tau = t - n + 1
            if tau >= 0:
                q_table = update_q_table(
                    q_table,
                    states,
                    actions,
                    rewards,
                    tau,
                    n,
                    t,
                    T,
                )
            if tau == T - 1:
                break
    return q_table


def get_optimal_path(
    cities_locations_gdf: gpd.GeoDataFrame,
    distances: np.ndarray,
    start_city: str,
    end_city: str,
    n: int,
) -> tuple:
    start_city_index = cities_locations_gdf[
        cities_locations_gdf["Label"] == start_city
    ].index[0]
    end_city_index = cities_locations_gdf[
        cities_locations_gdf["Label"] == end_city
    ].index[0]
    q_table = train_agent(
        EPISODES, start_city_index, end_city_index, distances, N_STEPS
    )
    shortest_path, route = utils.get_optimal_path_and_distance(
        cities_locations_gdf,
        distances,
        q_table,
        start_city_index,
        end_city_index,
        f"data/east_africa/{start_city}_{end_city}_off_policy_n_step_sarsa_q_table_{EPISODES}.csv",
    )
    return shortest_path, route


def main(api_key: str) -> None:
    cities_locations_gdf, distances = utils.get_training_data(api_key)
    # utils.plot_cities(cities_locations_gdf)
    shortest_path, route = get_optimal_path(
        cities_locations_gdf, distances, "Nairobi", "Kampala", 3
    )
    print(shortest_path)
    print(route)
    utils.plot_cities(cities_locations_gdf, route)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_key", type=str, required=True)

    args = parser.parse_args()
    main(args.api_key)
