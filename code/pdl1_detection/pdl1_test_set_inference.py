# Generic imports
import torch
import matplotlib.pyplot as plt
from simple_cocotools.utils.coco import CocoDetection
from torch.utils.data import DataLoader
from pathlib import Path

# Imports from codebase
from src.eval_utils import plot_f1_matrix
from src.inference_utils import transform_to_tensors, collate_fn, visualize_inference
from src.inference import run_inference, calculate_f1_scores, get_annotations

# Constants
RUMC_READERS = ["reader P1", "reader P2", "reader P3"]
SCDC_READERS = ["reader P4", "reader P5", "reader P6"]
HIT_CRITERION = 20  # At spacing 0.5 um/px, this corresponds to an 10 um margin
CLASSES = [
    "pd-l1 negative tumor cell",
    "pd-l1 positive tumor cell",
    "non-tumor cell",
]
TILE_SIZE = 512
CONF_THRES = 0.5
IOU_THRES = 0.6
LABEL_FUNC = (
    lambda id: (id - 1) % 3
)  # Map category IDs to class indices (0, 1, 2 for the three classes)


def main():

    # Make output folders if it does not exist
    Path(inference_output_folder).mkdir(parents=True, exist_ok=True)
    Path(figures_output_folder).mkdir(parents=True, exist_ok=True)

    # Load test set annotations
    test_set = CocoDetection(
        root=img_folder, annFile=annotation_file, transforms=transform_to_tensors
    )
    batch_size = 16
    data_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    # Initialize model
    torch.hub.set_dir("/tmp/.cache/torch")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.hub.load(
        "ultralytics/yolov5",
        "custom",
        path=weight_path,
        trust_repo=True,
        force_reload=True,
    ).to(device)

    inference_results = run_inference(
        data_loader,
        model,
        test_set,
        inference_output_folder,
        device,
        batch_size,
        tile_size=TILE_SIZE,
        conf_thres=CONF_THRES,
        iou_thres=IOU_THRES,
    )
    readers = RUMC_READERS + SCDC_READERS
    annotations_per_reader = get_annotations(test_set, readers, LABEL_FUNC)
    annotations_per_reader["AI"] = inference_results
    visualize_inference(
        test_set,
        annotations_per_reader,
        readers,
        CLASSES,
        img_folder,
        figures_output_folder,
    )
    rumc_f1_scores = calculate_f1_scores(
        annotations_per_reader, RUMC_READERS + ["AI"], HIT_CRITERION, len(CLASSES)
    )
    scdc_f1_scores = calculate_f1_scores(
        annotations_per_reader, SCDC_READERS + ["AI"], HIT_CRITERION, len(CLASSES)
    )

    reader_groups = [
        (rumc_f1_scores, RUMC_READERS + ["AI"], "RUMC", plt.get_cmap("Blues")),
        (scdc_f1_scores, SCDC_READERS + ["AI"], "SCDC", plt.get_cmap("Reds")),
    ]
    for f1_scores, readers, group_name, cmap in reader_groups:
        df_f1 = f1_scores["f1 macro"].loc[readers, readers].T
        output_path = Path(figures_output_folder, f"f1_scores_{group_name}.png")
        plot_f1_matrix(
            df_f1,
            readers,
            output_path,
            title=f"F1 scores per reader ({group_name})",
            cmap=cmap,
        )

    print(
        "Inference completed and results saved, see output folders:",
        inference_output_folder,
        figures_output_folder,
        sep="\n",
    )


if __name__ == "__main__":

    # Paths to data and model
    img_folder = "../../data/images/pdl1/pdl1"
    annotation_file = "../../data/annotations/pdl1/pdl1/pdl1_test_set_all_readers.json"
    weight_path = "../../data/models/pdl1/pdl1/pdl1_detection_weights.pt"
    inference_output_folder = "../../data/inference/pdl1/pdl1/"
    figures_output_folder = "../../data/figures/pdl1/pdl1/"

    print("Running inference on PDL1 positive tumor cell detection test set")
    main()
