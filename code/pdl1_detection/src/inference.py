# Ignore deprecation warnings from torch.cuda.amp.autocast
import warnings

warnings.filterwarnings(
    "ignore",
    message=r"`torch\.cuda\.amp\.autocast\(.*\)` is deprecated",
    category=FutureWarning,
)

# Generic imports
import torch
import numpy as np
import pandas as pd
from collections import defaultdict
from itertools import combinations
from tqdm import tqdm
from pathlib import Path

# Imports from codebase
from src.eval_utils import calculate_confusion_matrix, get_metrics
from yolov5.utils.general import non_max_suppression, xyxy2xywh


def _process_batch(
    images, model, tile_size=128, conf_thres=0.5, iou_thres=0.5, max_det=1500
):
    """
    Inference on a batch of images. To inference images that are larger than the model input size,
    the images are split into tiles of size `tile_size` x `tile_size`. The tiles are then processed
    by the model and the predictions are adjusted to the original image coordinates.
    """
    stride = tile_size
    output = []

    for image in images:
        _, height, width = image.shape
        tiles = []

        # Generate tiles
        for y in range(0, height, stride):
            for x in range(0, width, stride):
                tile = image[:, y : y + tile_size, x : x + tile_size]
                tiles.append(tile)

        tiles = torch.stack(tiles)
        pred = model(tiles)
        pred = non_max_suppression(
            pred,
            conf_thres=conf_thres,
            iou_thres=iou_thres,
            classes=None,
            agnostic=True,
            max_det=max_det,
        )

        # Process predictions for each tile
        image_output = []
        for i, p in enumerate(pred):
            if p is not None:
                xyxy, _, c = p[:, :4], p[:, 4], p[:, 5]
                xywh = xyxy2xywh(xyxy)
                xyc = torch.cat(
                    (xywh[:, :2], c.unsqueeze(1)), dim=1
                ).cpu()  # Centered (x,y) coordinates + class

                # Adjust coordinates to global image space
                tile_x = (i % (width // stride)) * stride
                tile_y = (i // (width // stride)) * stride
                xyc[:, 0] += tile_x
                xyc[:, 1] += tile_y

                image_output.append(xyc)

        output.append(
            torch.cat(image_output, dim=0) if image_output else torch.empty((0, 3))
        )

    return output


def run_inference(
    data_loader,
    model,
    test_set,
    output_folder,
    device,
    batch_size,
    tile_size=512,
    iou_thres=0.5,
    conf_thres=0.5,
):
    results = dict()

    for i, (images, _) in tqdm(
        enumerate(data_loader), desc="Running inference", total=len(data_loader)
    ):
        images = images.to(device)
        xyc = _process_batch(
            images,
            model,
            tile_size=tile_size,
            conf_thres=conf_thres,
            iou_thres=iou_thres,
        )
        image_ids = batch_size * i + np.arange(batch_size)
        image_ids = image_ids[: len(images)]  # Handle len(images) < batch_size
        image_info = test_set.coco.loadImgs(image_ids)

        for patch, xyc_i, info in zip(images, xyc, image_info):

            file_name = Path(info["file_name"]).stem
            img_id = info["id"]

            # Discard predictions outside of annotated area
            annotated_xyxy = info["annotated_xyxy"]
            xyc_i = xyc_i.cpu()
            xyc_i = xyc_i[
                (xyc_i[:, 0] >= annotated_xyxy[0])
                & (xyc_i[:, 0] <= annotated_xyxy[2])
                & (xyc_i[:, 1] >= annotated_xyxy[1])
                & (xyc_i[:, 1] <= annotated_xyxy[3])
            ]
            if len(xyc_i) == 0:
                print(f"No predictions for image {file_name}")
                continue

            # Save XYC
            xyc_i = xyc_i.round(decimals=2).cpu().numpy()
            output_path = Path(output_folder, file_name).with_suffix(".npy")
            np.save(output_path, xyc_i)

            results[img_id] = xyc_i
    return results


def get_annotations(test_set, readers, label_func):
    annotations_per_reader = defaultdict(dict)
    center_offset = (
        test_set.coco.anns[0]["bbox"][-1] // 2
    )  # COCO (x,y) coordinates are measured from the top left, so we need to center the bounding boxes
    for reader in readers:
        cat_ids = test_set.coco.getCatIds(supNms=reader)
        img_ids = {
            img_id for id in cat_ids for img_id in test_set.coco.getImgIds(catIds=[id])
        }  # OR filter for img ids
        for img_id in img_ids:
            anno_ids = test_set.coco.getAnnIds(imgIds=img_id, catIds=cat_ids)
            annos = test_set.coco.loadAnns(anno_ids)
            annos = [
                (
                    anno["bbox"][0] + center_offset,
                    anno["bbox"][1] + center_offset,
                    label_func(anno["category_id"]),  # Convert category_id to [0,1,2]
                )
                for anno in annos
            ]
            annotations_per_reader[reader][img_id] = np.array(annos)
    return annotations_per_reader


def calculate_f1_scores(annotations_per_reader, readers, hit_criterion, n_classes):
    """
    For every pair of corresponding readers, consider one reader as ground truth and the other as predictions.
    Then calculate TP/TN/FP/FN for every ROI and calculate the F1 scores for all classes.
    """

    conf_matrixes_per_pair = defaultdict(dict)
    pairs = list(combinations(readers, 2))

    for pair in tqdm(pairs, desc="Calculating confusion matrices"):
        reader1, reader2 = pair
        for img_id in annotations_per_reader[reader1].keys():
            ground_truth = annotations_per_reader[reader1][img_id]
            predictions = annotations_per_reader[reader2][img_id]
            confusion_matrix = calculate_confusion_matrix(
                ground_truth=ground_truth,
                predictions=predictions,
                n_classes=n_classes,
                radius=hit_criterion,
            )
            conf_matrixes_per_pair[pair][img_id] = confusion_matrix

    summed_conf_matrixes_per_pair = dict()  # Sum over all images for each pair
    for pair, cfms_ in conf_matrixes_per_pair.items():
        summed_conf_matrixes_per_pair[pair] = sum(cfms_.values())

    categories = ["f1 macro"] + [f"f1: {c}" for c in range(n_classes)]
    dfs_f1 = {
        category: pd.DataFrame(index=readers, columns=readers, dtype=float)
        for category in categories
    }

    for pair, cfm in summed_conf_matrixes_per_pair.items():
        reader_1, reader_2 = pair
        metrics = get_metrics(cfm, range(n_classes))
        for category in categories:
            dfs_f1[category].loc[reader_1, reader_2] = round(metrics[category], 2)

    return dfs_f1
