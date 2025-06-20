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
READERS = ["reader R1", "reader R2", "reader R3", "reader R4"]
HIT_CRITERION = 8  # At spacing 0.5 um/px, this corresponds to an 4 um margin
CLASSES = ["nucleus"]
TILE_SIZE = 128
CONF_THRES = 0.6
IOU_THRES = 0.4
LABEL_FUNC = lambda id: 0  # Map category IDs to class indices (0 for nucleus)


def main():

    # Make output folder if it does not exist
    Path(inference_output_folder).mkdir(parents=True, exist_ok=True)
    Path(figures_output_folder).mkdir(parents=True, exist_ok=True)

    # Load test set annotations
    test_set = CocoDetection(
        root=img_folder, annFile=annotation_file, transforms=transform_to_tensors
    )
    batch_size = 1
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
        data_loader, model, test_set, inference_output_folder, device, batch_size
    )
    annotations_per_reader = get_annotations(test_set, READERS, LABEL_FUNC)
    annotations_per_reader["AI"] = inference_results
    visualize_inference(
        test_set,
        annotations_per_reader,
        READERS,
        CLASSES,
        img_folder,
        figures_output_folder,
    )
    readers = READERS + ["AI"]
    f1_scores = calculate_f1_scores(
        annotations_per_reader, readers, HIT_CRITERION, len(CLASSES)
    )

    df_f1 = f1_scores["f1 macro"].loc[readers, readers].T
    output_path = Path(figures_output_folder, f"f1_scores_nuclei.png")
    plot_f1_matrix(
        df_f1,
        readers,
        output_path,
        title=f"F1 scores per reader",
        cmap=plt.get_cmap("Blues"),
    )

    print(
        "Inference completed and results saved, see output folder:",
        inference_output_folder,
        figures_output_folder,
        sep="\n",
    )


if __name__ == "__main__":

    # Paths to data and model
    img_folder = "../../data/images/pdl1/nuclei/"
    annotation_file = "../../data/annotations/pdl1/nuclei/nuclei_test_set_all_readers.json"
    weight_path = "../../data/models/pdl1/nuclei/nuclei_detection_weights.pt"
    inference_output_folder = "../../data/inference/pdl1/nuclei/"
    figures_output_folder = "../../data/figures/pdl1/nuclei/"

    print("Running inference on PDL1 nuclei detection test set")
    main()
