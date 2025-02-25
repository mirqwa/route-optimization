import argparse

import utils


def main(api_key: str) -> None:
    cities_locations_gdf, distances = utils.get_training_data(api_key)
    utils.plot_cities(cities_locations_gdf)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_key", type=str, required=True)

    args = parser.parse_args()
    main(args.api_key)
