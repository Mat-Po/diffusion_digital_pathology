import argparse
from data_generation.utils import tiles_generation_utils
import json
import os


def get_args():
    r"""Parse command line arguments."""

    parser = argparse.ArgumentParser(
        prog="python -m data_generation",
        description="Tiles generation and normalization from slides data",
    )

    # parser arguments
    parser.add_argument(
        "--normalize",
        "-n",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Normalize pre existing tiles in the path given in settings.json",
    )

    parser.add_argument(
        "--preview",
        "-p",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Generate rescaled slides with possible extracted tiles outlined",
    )
    # parse arguments
    parsed_args = parser.parse_args()

    return parsed_args


def main(args):
    dirname = os.path.dirname(__file__)

    try:
        with open(os.path.join(dirname, "settings.json"), "r") as f:
            settings = json.load(f)
    except FileNotFoundError:
        print("Missing settings.json file")
        exit()

    # targets = []
    # for key in settings['normalization_targets']:
    #    targets.append((key,settings['normalization_targets'][key]))

    if args.normalize:
        tiles_that_cause_error = []
        # just normalize
        # for target in targets:
        #    tiles_generation_utils.normalize(target[1],target[0],tiles_that_cause_error,settings)
        tiles_generation_utils.normalize_tiles(tiles_that_cause_error, settings)

        with open(
            settings["general_parameters"]["normalization_path"]
            + "tiles_not_normalized.json",
            "w",
        ) as f:
            json.dump(tiles_that_cause_error, f)
    elif args.preview:
        tiles_generation_utils.preview_tiles_generation(settings)
    else:
        tiles_generation_utils.generate_tiles(settings)

        # generate first and then normalize


if __name__ == "__main__":
    main(get_args())
