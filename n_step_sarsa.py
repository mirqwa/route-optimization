import argparse

import geopandas as gpd
import numpy as np

import utils


np.random.seed(32)

EPSILON = 0.2
LEARNING_RATE = 0.01
DISCOUNT_FACTOR = 0.8
EPISODES = 1000
NO_OF_NEIGHBORS = 10
MAX_STEPS = 1000


def train_agent(
    num_episodes: int,
    start_city_index: str,
    end_city_index: str,
    distances: np.ndarray,
    n: int,
) -> np.ndarray:
    q_table = utils.initialize_q_table(distances, NO_OF_NEIGHBORS)
    for episode in range(num_episodes):
        print(f"Episode {episode + 1}")
        current_city = start_city_index
        states = [current_city]
        rewards = []
        action = utils.select_next_action(
            distances, current_city, q_table, NO_OF_NEIGHBORS, EPSILON
        )
        actions = [action]
        T = float("inf")
        for t in range(MAX_STEPS):
            if t < T:
                next_city = action
                reward = -distances[current_city, action]
                states.append(next_city)
                rewards.append(reward)
                if next_city == end_city_index:
                    T = t + 1
                else:
                    next_action = utils.select_next_action(
                        distances, next_city, q_table, NO_OF_NEIGHBORS, EPSILON
                    )
                    actions.append(next_action)
                    action = next_action
            tau = t - n + 1
            if tau >= 0:
                G = sum(
                    [
                        DISCOUNT_FACTOR ** (i - tau) * rewards[i]
                        for i in range(tau, min(tau + n, T))
                    ]
                )
                G = (
                    G
                    + DISCOUNT_FACTOR**n * q_table[states[tau + n]][actions[tau + n]]
                    if t + n < T
                    else G
                )

                q_table[states[tau]][actions[tau]] = (1 - LEARNING_RATE) * q_table[
                    states[tau]
                ][actions[tau]] + LEARNING_RATE * G
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
    q_table = train_agent(EPISODES, start_city_index, end_city_index, distances, 3)


def main(api_key: str) -> None:
    cities_locations_gdf, distances = utils.get_training_data(api_key)
    # utils.plot_cities(cities_locations_gdf)
    get_optimal_path(cities_locations_gdf, distances, "Nairobi", "Kampala", 3)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_key", type=str, required=True)

    args = parser.parse_args()
    main(args.api_key)
