from PIL import Image


def concatenate_tiles(tile1, tile2):
    dst = Image.new("RGB", (tile1.width + tile2.width, tile1.height))
    dst.paste(tile1, (0, 0))
    dst.paste(tile2, (tile1.width, 0))
    return dst


def merge_tiles(tiles):
    tmp_image = tiles[0]
    for i in range(1, len(tiles)):
        tmp_image = concatenate_tiles(tmp_image, tiles[i])
    return tmp_image
