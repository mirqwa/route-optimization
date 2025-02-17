import argparse

import numpy as np

import utils


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_key", type=str, required=True)

    args = parser.parse_args()
    main(args.api_key)
