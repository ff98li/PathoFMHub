# %%
import pyvips
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import json
import multiprocessing as mp
import tqdm
import argparse
join = os.path.join

# %%
parser = argparse.ArgumentParser()
parser.add_argument("--data_root", type=str, default="./UBC-OCEAN")
parser.add_argument("--tile_size", type=int, default=224)
parser.add_argument("--non_bg_threshold", type=float, default=0.5)
parser.add_argument("--overlap", type=int, default=0)
parser.add_argument("--num_workers", type=int, default=2)
args = parser.parse_args()

# %%
np.random.seed(2023)
non_bg_threshold = args.non_bg_threshold
overlap = args.overlap
tile_size = args.tile_size
num_workers = args.num_workers
# %%
data_root = args.data_root
train_dir = join(data_root, "train_images")
train_csv = join(data_root, "train.csv")
train_df = pd.read_csv(train_csv)
# %%
def background_ratio(rgb):
    bg_mask = (
        (rgb[..., 0] >= 200) & (rgb[..., 1] >= 200) & (rgb[..., 2] >= 200)
    ) | (
        (rgb[..., 0] <= 10) & (rgb[..., 1] <= 10) & (rgb[..., 2] <= 10)
    )
    bg_pixel_count = np.sum(bg_mask)
    try:
        ratio = bg_pixel_count / np.prod(rgb.shape[:2])
    except:
        print(f"rgb.shape: {rgb.shape}")
        print(f"bg_pixel_count: {bg_pixel_count}")
        print(f"np.prod(rgb.shape[:2]): {np.prod(rgb.shape[:2])}")
        ratio = 0

    return ratio

def pad_tile(tile, tile_size=224):
    tile_height = tile.shape[0]
    tile_width = tile.shape[1]
    if tile_height < tile_size:
        pad_height = tile_size - tile_height
        tile = np.pad(tile, ((0, pad_height), (0, 0), (0, 0)), mode="constant")
    if tile_width < tile_size:
        pad_width = tile_size - tile_width
        tile = np.pad(tile, ((0, 0), (0, pad_width), (0, 0)), mode="constant")
    return tile
# %%
#idx = np.random.randint(0, len(train_df))
#if True:
def main(idx):
    image_id = train_df["image_id"][idx]
    image_height = train_df["image_height"][idx]
    image_width = train_df["image_width"][idx]
    label = train_df["label"][idx]
    is_tma = train_df["is_tma"][idx]
    image_arr = pyvips.Image.new_from_file(join(train_dir, str(image_id) + ".png")).numpy()

    json_slide_dict = {
        "image_id": int(image_id),
        "label": label,
        "image_width": int(image_width),
        "image_height": int(image_height),
        "is_tma": bool(is_tma),
        "tiles": []
    }
    for x_min in range(0, image_width, tile_size):
        for y_min in range(0, image_height, tile_size):
            if y_min+tile_size > image_height:
                slice_row = slice(y_min, image_height)
                w = image_height-y_min
            else:
                slice_row = slice(y_min, y_min+tile_size)
                w = tile_size
            if x_min+tile_size > image_width:
                slice_col = slice(x_min, image_width)
                h = image_width-x_min
            else:
                slice_col = slice(x_min, x_min+tile_size)
                h = tile_size

            tile = image_arr[slice_row, slice_col, :]
            if w < tile_size or h < tile_size:
                tile = pad_tile(tile, tile_size=tile_size)
                assert tile.shape == (tile_size, tile_size, 3)

            bg_ratio = background_ratio(tile)
            tissue_ratio = 1 - bg_ratio
            if tissue_ratio < non_bg_threshold:
                continue
            else:
                json_tile_dict = {
                    "x_min": x_min,
                    "y_min": y_min,
                    "w": w,
                    "h": h,
                    "tissue_ratio": tissue_ratio
                }
                json_slide_dict["tiles"].append(json_tile_dict)

    return json_slide_dict

# %%
if __name__ == "__main__":
    json_dataset_dict = {
         "labels": {
           "Other": 0, ## Negative
           "CC": 1,
           "EC": 2,
           "HGSC": 3,
           "LGSC": 4,
           "MC": 5
         }, 
         "numTraining": len(train_df), 
         "file_ending": ".png",
         "overlap": overlap,
         "slides": []
    }

    if num_workers > 1:
        with mp.Pool(num_workers) as pool:
            for json_slide_dict in tqdm.tqdm(pool.imap_unordered(main, range(len(train_df))), total=len(train_df)):
                json_dataset_dict["slides"].append(json_slide_dict)
    else:
        for json_slide_dict in tqdm.tqdm(map(main, range(len(train_df))), total=len(train_df)):
            json_dataset_dict["slides"].append(json_slide_dict)
    
    with open(join('.', "patch.json"), "w") as f:
        json.dump(json_dataset_dict, f, indent=4)