import torch
import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_tensor
from tqdm import tqdm
from pathlib import Path

# (PD-L1 negative tumor cells/nucleus, PD-L1 positive tumor cells, non-tumor cells)
CMAP = {
    0: "blue",  # PD-L1 negative tumor cell and nucleus
    1: "red",  # PD-L1 positive tumor cell
    2: "yellow",  # Non-tumor cell
}
READER_SYMBOLS = ["s", "o", "d", "h"]


def transform_to_tensors(image, targets):
    out_image = to_tensor(image)
    out_targets = torch.tensor(
        [(t["bbox"][0], t["bbox"][1], t["category_id"]) for t in targets]
    )
    return out_image, out_targets


def collate_fn(batch):
    imgs, annotations = tuple(zip(*batch))
    imgs = torch.stack(imgs, dim=0)
    return imgs, annotations


def _visualize_inference(patch, xyc, output, annotated_xyxy=None):
    x_hat, y_hat, c_hat = xyc.T
    c = [CMAP[int(c)] for c in c_hat.numpy()]
    plt.figure(figsize=(8, 8))
    plt.imshow(patch)
    plt.axis("off")
    plt.scatter(x_hat.int(), y_hat.int(), c=c, s=70, marker="^", edgecolors="black")

    # Draw annotated bounding box if provided
    if annotated_xyxy is not None:
        x_min, y_min, x_max, y_max = annotated_xyxy
        plt.plot([x_min, x_max], [y_min, y_min], linestyle="--", color="black")
        plt.plot([x_min, x_max], [y_max, y_max], linestyle="--", color="black")
        plt.plot([x_min, x_min], [y_min, y_max], linestyle="--", color="black")
        plt.plot([x_max, x_max], [y_min, y_max], linestyle="--", color="black")
    plt.savefig(output, bbox_inches="tight", pad_inches=0.1)
    plt.close()


def _visualize_inference_multireader(
    patch,
    xyc_readers,
    xyc_ai,
    output,
    classes,
    annotated_xyxy=None,
    marker_size=30,
    title="AI Predictions",
):
    n_readers = len(xyc_readers)
    reader_symbols = READER_SYMBOLS[:n_readers]

    ratio = [14]
    fig, axs = plt.subplots(
        1,
        3,
        figsize=(15, 5),
        gridspec_kw={"width_ratios": 3 * [ratio[0]]},
    )
    fig.suptitle(title)
    fig.subplots_adjust(wspace=0.5)

    # Subplot 1: Show the patch
    axs[0].imshow(patch)
    axs[0].set_title("Input image")

    # Subplot 2: Predictions of all readers
    axs[1].imshow(patch)
    for i, xyc_reader in enumerate(xyc_readers):
        x, y, c = xyc_reader.T
        colors = [CMAP[int(cls)] for cls in c]
        axs[1].scatter(
            x,
            y,
            c=colors,
            s=marker_size,
            marker=reader_symbols[i],
            edgecolors="black",
            label=f"Reader {i+1}",
        )
    axs[1].set_title("Reader predictions")
    axs[1].legend(loc="upper right", fontsize=8)

    # Subplot 3: Predictions of AI with legend
    axs[2].imshow(patch)
    x_ai, y_ai, c_ai = xyc_ai.T
    colors_ai = [CMAP[int(cls)] for cls in c_ai]
    axs[2].scatter(
        x_ai, y_ai, c=colors_ai, s=marker_size, marker="^", edgecolors="black"
    )
    axs[2].set_title("AI predictions")

    # Add legend for class colors
    for cls, color in zip(classes, CMAP.values()):
        axs[2].scatter(
            [],
            [],
            color=color,
            marker="^",
            label=cls,
            s=marker_size,
            edgecolors="black",
        )
    axs[2].legend(loc="lower right", fontsize=8)

    # Remove ticks from all subplots
    for ax in axs:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.tick_params(labelbottom=False, labelleft=False)

    # Show annotated bounding box if provided
    if annotated_xyxy is not None:
        x_min, y_min, x_max, y_max = annotated_xyxy
        for ax in axs:  # Draw bounding box on all subplots
            ax.plot([x_min, x_max], [y_min, y_min], linestyle="--", color="black")
            ax.plot([x_min, x_max], [y_max, y_max], linestyle="--", color="black")
            ax.plot([x_min, x_min], [y_min, y_max], linestyle="--", color="black")
            ax.plot([x_max, x_max], [y_min, y_max], linestyle="--", color="black")

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)  # Adjust space for the title

    plt.savefig(output, bbox_inches="tight", pad_inches=0.1, dpi=150)
    plt.close()


def visualize_inference(
    dataset, annotations_per_reader, readers, classes, image_folder, output_folder
):

    for id, img_info in tqdm(
        dataset.coco.imgs.items(), desc="Visualizing inference results"
    ):

        img_path = Path(image_folder, img_info["file_name"])
        img = plt.imread(img_path)
        annotated_xyxy = img_info["annotated_xyxy"]
        xyc_ai = annotations_per_reader["AI"].get(id)
        xyc_readers = [annotations_per_reader[reader].get(id) for reader in readers]
        xyc_readers = [
            xyc for xyc in xyc_readers if xyc is not None
        ]  # Not all images have annotations from all readers

        output_path = Path(output_folder, img_info["file_name"]).with_suffix(".png")
        _visualize_inference_multireader(
            img,
            xyc_readers,
            xyc_ai,
            output=output_path,
            annotated_xyxy=annotated_xyxy,
            classes=classes,
            title=f"Evaluation plot for {img_info['file_name']}",
        )
