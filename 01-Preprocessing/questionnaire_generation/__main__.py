import argparse
from questionnaire_generation.utils import questionnaire_utils
import json


def get_args():
    r"""Parse command line arguments."""

    parser = argparse.ArgumentParser(
        prog="python -m questionnaire_generation",
        description="Questionnaires generation from tiles",
    )

    # parse arguments
    parsed_args = parser.parse_args()

    return parsed_args


def main(args):
    try:
        with open("questionnaire_generation/settings.json", "r") as f:
            settings = json.load(f)
    except FileNotFoundError:
        print("Missing settings.json file")
        exit()

    questionnaire_utils.generate_questionnaire(settings)


if __name__ == "__main__":
    main(get_args())
