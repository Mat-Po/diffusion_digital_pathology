import shutil
import random
from questionnaire_generation.utils import image_generation_utils
from PIL import Image
import os
import tqdm


def generate_images_to_question(
    tissues,
    tiles_path,
    realPath,
    number_images_first_task,
    tiles_per_image,
    number_images_second_task,
    output_path,
    directory,
):
    for tissue in tissues:
        # get tiles
        tiles_names = []

        for root, dirs, files in os.walk(tiles_path + tissue):
            tiles_names.extend(files)

        tiles = random.sample(
            tiles_names,
            number_images_first_task * tiles_per_image + number_images_second_task,
        )

        # generating guess the tissue tiles
        for i in range(
            tiles_per_image,
            number_images_first_task * tiles_per_image + tiles_per_image,
            tiles_per_image,
        ):
            tiles_to_merge = tiles[i - tiles_per_image : i]
            tiles_to_merge_as_img = [
                Image.open(tiles_path + tissue + "/" + tile) for tile in tiles_to_merge
            ]
            merged_tiles = image_generation_utils.merge_tiles(tiles_to_merge_as_img)
            merged_tiles.save(
                f"{output_path+directory}/first_task-{tissue}-{realPath}-image-{i}-{tiles_per_image}_tiles_per_image.png"
            )

        for tile in tiles[number_images_first_task * tiles_per_image :]:
            shutil.copy(tiles_path + tissue + "/" + tile, output_path + directory + "/"+tissue+"_"+tile)


def generate_questionnaire(settings):
    n_questionnaires = int(settings["number_questionnaires"])

    if not (os.path.isdir(settings["output_path"])):
        os.mkdir(settings["output_path"])

    if not (os.path.isdir(f"{settings['output_path']}fixed_questions/")):
        os.mkdir(f"{settings['output_path']}fixed_questions/")

    generate_images_to_question(
        settings["tissues"],
        settings["real_tiles_path"],
        "real",
        settings["number_images_first_task_per_tissue"],
        settings["tiles_per_image_first_task"],
        settings["number_images_second_task_per_tissue"],
        f"{settings['output_path']}",
        "fixed_questions",
    )

    generate_images_to_question(
        settings["tissues"],
        settings["fake_tiles_path"],
        "fake",
        settings["number_images_first_task_per_tissue"],
        settings["tiles_per_image_first_task"],
        settings["number_images_second_task_per_tissue"],
        settings["output_path"],
        "fixed_questions",
    )

    for i in tqdm.tqdm(
        range(n_questionnaires),
        desc="Generating Questionnaires",
        unit="it",
        position=0,
        leave=True,
    ):
        if not (os.path.isdir(f"{settings['output_path']}questionaire_number_{i+1}/")):
            os.mkdir(f"{settings['output_path']}questionaire_number_{i+1}/")

        generate_images_to_question(
            settings["tissues"],
            settings["real_tiles_path"],
            "real",
            settings["questionnaire_specific_number_first_task_questions_per_tissue"],
            settings["tiles_per_image_first_task"],
            settings["questionnaire_specific_number_second_task_questions_per_tissue"],
            settings["output_path"],
            f"questionaire_number_{i+1}",
        )

        generate_images_to_question(
            settings["tissues"],
            settings["fake_tiles_path"],
            "fake",
            settings["questionnaire_specific_number_first_task_questions_per_tissue"],
            settings["tiles_per_image_first_task"],
            settings["questionnaire_specific_number_second_task_questions_per_tissue"],
            settings["output_path"],
            f"questionaire_number_{i+1}",
        )
