from histolab.slide import Slide
from histolab.masks import TissueMask,BiggestTissueBoxMask
from histolab.tiler import GridTiler,ScoreTiler

from histolab.filters.image_filters import (
    ApplyMaskImage,
    Invert,
    OtsuThreshold,
    RgbToGrayscale,
    FilterEntropy
)

from histolab.filters.morphological_filters import BinaryDilation,BinaryFillHoles

from histolab.stain_normalizer import MacenkoStainNormalizer

from histolab.filters.morphological_filters import RemoveSmallHoles, RemoveSmallObjects

from PIL import Image

import os

import tqdm

import pandas as pd
import abc
from histolab.scorer import Scorer
import cv2
import numpy as np
from PIL import Image


class ComplexityScorer(Scorer):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, tile) -> float:
        tile_array = np.array(tile.image)
    
        tile_array = tile_array[:,:,:-1]
        tile_array = tile_array[:,:,::-1]
        #print(tile_array.shape)
        hog = cv2.HOGDescriptor()
        gradient_im,_ = hog.computeGradient(tile_array,5,8)
        return (1/(gradient_im.shape[0]**2))*np.sum(gradient_im)

def show_slide_with_tiles(slide, tiles_extractor, mask):
    return tiles_extractor.locate_tiles(
        slide=slide,
        scale_factor=24,  # default
        alpha=128,  # default
        outline="red",
        extraction_mask=mask,
    )


def get_slides_from_dir(slides_path):
    slides_filenames = []
    for root, dirs, files in os.walk(slides_path):
        slides_filenames.extend(files)

    return slides_filenames


def get_custom_mask_for_slide(slide):
    return TissueMask(
        RgbToGrayscale(),
        OtsuThreshold(),
        BinaryDilation(),
        BinaryFillHoles()
    )


def normalize_tiles(tiles_that_cause_error, settings):
    for tissue in settings["tissues"]:
        tissue_tile_names = []
        for root, dirs, files in os.walk(
            settings["general_parameters"]["processed_path"]
            + "/"
            + settings["general_parameters"]["prefix"]
            + tissue
        ):
            tissue_tile_names.extend(files)

        tile_target_name = settings["normalization_parameters"]["targets"][tissue]
        normalizer = MacenkoStainNormalizer()
        normalizer.fit(
            Image.open(
                settings["general_parameters"]["processed_path"]
                + "/"
                + settings["general_parameters"]["prefix"]
                + tissue
                + "/"
                + tile_target_name
            )
        )

        for tile_name in tqdm.tqdm(
            tissue_tile_names,
            desc=f"Normalizing {tissue} Tiles",
            unit="it",
            position=0,
            leave=True,
        ):
            tile_image = Image.open(
                settings["general_parameters"]["processed_path"]
                + "/"
                + settings["general_parameters"]["prefix"]
                + tissue
                + "/"
                + tile_name
            )
            if tile_name != tile_target_name:
                try:
                    normalized_tile = normalizer.transform(tile_image)
                    normalized_tile.save(
                        settings["general_parameters"]["normalization_path"]
                        + tissue
                        + "/"
                        + tile_name
                    )
                except:
                    # errore nella normalizzazione
                    tiles_that_cause_error.append(tile_name)
                    print(f"tile {tile_name} not normalized, an error occured")
            else:
                tile_image.save(
                    settings["general_parameters"]["normalization_path"]
                    + tissue
                    + "/"
                    + tile_name
                )


def preview_tiles_generation(settings):
    slide_files = get_slides_from_dir(settings["general_parameters"]["slides_path"])
    scorer = ComplexityScorer()
    to_care_about = []
    for slide_file in tqdm.tqdm(
        slide_files,
        desc="Generating tiled slide images",
        unit="it",
        position=0,
        leave=True,
    ):
        
        slide = Slide(
            settings["general_parameters"]["slides_path"] + slide_file,
            processed_path=settings["general_parameters"]["processed_path"],
        )
        # if(slide.level_magnification_factor(level=settings["general_parameters"]["level"]) != "1.25X"):
        #     continue

        grid_extractor = GridTiler(
            tile_size=(
                settings["general_parameters"]["tile_size"],
                settings["general_parameters"]["tile_size"],
            ),
            level=settings["general_parameters"]["level"],
            check_tissue=settings["general_parameters"]["check_tissue"],
            pixel_overlap=settings["general_parameters"]["pixel_overlap"],
            prefix="",
            suffix=f".png",
        )


        # grid_extractor = ScoreTiler( 
        #     scorer=scorer,
        #     n_tiles=0,
        #     tile_size=(
        #         settings["general_parameters"]["tile_size"],
        #         settings["general_parameters"]["tile_size"],
        #     ),
        #     level=settings["general_parameters"]["level"],
        #     check_tissue=settings["general_parameters"]["check_tissue"],
        #     pixel_overlap=settings["general_parameters"]["pixel_overlap"],
        #     prefix="",
        #     suffix=f".png",
        # )

        slide_mask = get_custom_mask_for_slide(slide)
        preview_tiled_slide = show_slide_with_tiles(slide, grid_extractor, slide_mask)
        print(settings["general_parameters"]["tiling_preview_path"] + slide_file[:-4]+".png")
        preview_tiled_slide.save(
            settings["general_parameters"]["tiling_preview_path"] + slide_file[:-4]+".png"
        )


def generate_tiles(settings):
    dirname = os.path.dirname(__file__)

    slides_metadata = None
    scorer = ComplexityScorer()

    try:
        with open(
            os.path.join(dirname, "../", "GTEx Portal.csv"), "r"
        ) as f:
            slides_metadata = pd.read_csv(f)
            slides_metadata['Tissue'] = slides_metadata['Tissue'].replace({"Kidney - Cortex":"Kidney","Brain - Cortex":"Brain"})
    except FileNotFoundError:
        print("Missing slides info file")
        exit(1)

    slide_files = get_slides_from_dir(settings["general_parameters"]["slides_path"])

    for slide_file in tqdm.tqdm(
        slide_files, desc="Generating Tiles", unit="it", position=0, leave=True
    ):

        slide_metadata = slides_metadata.loc[
            slides_metadata["Tissue Sample ID"] == slide_file[:-4]
        ]
        if(slide_metadata["Tissue"].values[0] not in settings['general_parameters']['tissues']):
            continue
        slide = Slide(
            settings["general_parameters"]["slides_path"] + slide_file,
            processed_path=settings["general_parameters"]["processed_path"],
        )
        # if(slide.level_magnification_factor(level=settings["general_parameters"]["level"]) != "1.25X"):
        #     continue
        
        filler = "_Endometrium" if slide_metadata["Tissue"].values[0] == "Uterus" else ""

        grid_extractor = GridTiler(
            tile_size=(
                settings["general_parameters"]["tile_size"],
                settings["general_parameters"]["tile_size"],
            ),
            level=settings["general_parameters"]["level"],
            check_tissue=settings["general_parameters"]["check_tissue"],
            pixel_overlap=settings["general_parameters"]["pixel_overlap"],
            prefix=settings["general_parameters"]["prefix"]
            + slide_metadata["Tissue"].values[0]
            + "/",
            suffix=f"-level-{settings['general_parameters']['level']}-tilesize-{settings['general_parameters']['tile_size']}-slidename-{slide_file[:-4]}-tissuetype-{slide_metadata['Tissue'].values[0]}{filler}{settings['general_parameters']['suffix']}",
        )


        # grid_extractor = ScoreTiler(
        #     scorer=scorer,
        #     tile_size=(
        #         settings["general_parameters"]["tile_size"],
        #         settings["general_parameters"]["tile_size"],
        #     ),
        #     level=settings["general_parameters"]["level"],
        #     check_tissue=settings["general_parameters"]["check_tissue"],
        #     pixel_overlap=settings["general_parameters"]["pixel_overlap"],
        #     prefix=settings["general_parameters"]["prefix"]
        #     + slide_metadata["Tissue"].values[0]
        #     + "/",
        #     suffix=f"-level-{settings['general_parameters']['level']}-tilesize-{settings['general_parameters']['tile_size']}-slidename-{slide_file[:-4]}-tissuetype-{slide_metadata['Tissue'].values[0]}{filler}{settings['general_parameters']['suffix']}",
        # )
        grid_extractor.extract(slide, get_custom_mask_for_slide(slide))
