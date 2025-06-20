import os
from pathlib import Path
from tqdm import tqdm
import numpy as np
from PIL import Image
from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

print('RUNNING EVALUATION SCRIPT')


##################################################
### Labels and plotting
##################################################

labels={
        'Unannotated':0,
        'Background':1, 
        'Tumor cells':2,
        'Reactive epithelium':3,
        'Stroma':4, 
        'Inflammation':5,
        'Alveolar tissue':6,
        'Fatty tissue':7,
        'Necrotic tissue':8, 
        'Erythrocytes':9,
        'Bronchial epithelium':10,
        'Mucus/Plasma/Fluids':11,
        'Cartilage/Bone':12,
        'Macrophages':13,
        'Muscle':14,
        'Liver':15,
        'Keratinization':16,
        }

colormap = {v: [128, 128, 128, 255] for v in range(256)}
colors ={
    0: [0, 0, 0, 255], # unannotated BLACK
    1: [255, 255, 255, 255], # background # WHITE
    2: [0, 253, 255, 255], # tumor CYAN #[0, 150, 255, 255], # tumor BLUE # [0, 253, 255, 255], # tumor CYAN #
    3: [0, 249, 0, 255], # reactive epithelium # GREEN
    4: [255, 251, 0, 255], # stroma # YELLOW [142, 250, 0, 255], # stroma # GREEN    #[255, 251, 0, 255], # stroma # YELLOW    
    5: [255, 147, 0, 255], # inflammation # ORANGE
    6: [255, 47, 146, 255], # healthy parenchyma # PINK
    7: [148, 55, 255, 255], # fatty tissue # PURPLE
    8: [1, 25, 147, 255], # necrotic tissue # DARK BLUE
    9: [255, 38, 0, 255], # erytrocytes # RED
    10: [0, 143, 0, 255], # healthy epithelium # GREEN
    11: [146, 144, 0, 255], # mucus # MUSTARD BROWN
    12: [169, 169, 169, 255], # cartilage bone GRAY #[115, 253, 255, 255], # cartilage # LIGHT RED
    13: [83, 27, 147, 255], # macrophages # DARK PURPLE #[0, 145, 147, 255], # macrophages # TEAL #[83, 27, 147, 255], # macrophages # DARK PURPLE #
    14: [255, 126, 121, 255], # muscle # LIGHT RED #[148, 82, 0, 255], # muscle ORANGE BROWN
    15: [215, 131, 255, 255], # liver # LIGHT PINK #[255, 138, 216, 255], # liver # LIGHT PINK
    16: [148, 23, 81, 255], # keratinization # DARK PINK
}

remap_dict = {
    0: 0, # unannotated
    1: 1, # bg
    2: 2, # tumor
    3: 6, # reactive
    4: 3, # stroma
    5: 3, # inflammation
    6: 6, # healthy parenchyma
    7: 6, # fatty tissue
    8: 4, # necrotic tissue
    9: 6, # erytrocytes
    10: 6, # healthy epithelium
    11: 6, # mucus
    12: 6, # cartilage/bone
    13: 5, # macrophages 
    14: 6, # mucle
    15: 6, # liver
    16: 6 # kera
}
labels_remapped = {
    'Unannotated': 0,
    'Background': 1,
    'Tumor cells': 2,
    'Stroma and Inflammation': 3,
    'Necrotic tissue': 4,
    'Macrophages': 5,
    'Rest': 6,
    }

colormap.update(colors)
colormap = np.array(list(colormap.values()))/255.
cmap = LinearSegmentedColormap.from_list('my_cmap', colors=colormap)
label_plot_args = {"cmap":cmap, "vmin":0, "vmax":255, "interpolation":"none"}

def get_nonzero_bbox(mask):
    rows = np.any(mask != 0, axis=1)
    cols = np.any(mask != 0, axis=0)
    if not np.any(rows) or not np.any(cols):
        return None
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return rmin, rmax+1, cmin, cmax+1

def crop_to_bbox(arr, bbox):
    rmin, rmax, cmin, cmax = bbox
    return arr[rmin:rmax, cmin:cmax]

def plot_img_gt_pred(img, gt, pred, labels, stem):
    ratio = [14, 2]
    fig, axs = plt.subplots(1, 4, figsize=(15, 4),
        gridspec_kw={'width_ratios': 3*[ratio[0]] + [ratio[-1]]})
    fig.suptitle(f'Evaluation plot for {stem}', fontsize=16)
    fig.subplots_adjust(top=0.85)  # Adjusts space to make room for suptitle

    axs[0].imshow(img)
    axs[0].set_title("Input image")
    axs[0].set_xticks([])
    axs[0].set_yticks([])
    axs[0].tick_params(labelbottom=False, labelleft=False)

    axs[1].imshow(img)
    axs[1].imshow(gt, **label_plot_args, alpha=0.5)
    axs[1].set_title("Ground truth")
    axs[1].set_xticks([])
    axs[1].set_yticks([])
    axs[1].tick_params(labelbottom=False, labelleft=False)

    axs[2].imshow(img)
    axs[2].imshow(pred, **label_plot_args, alpha=0.5)
    axs[2].set_title("Prediction")
    axs[2].set_xticks([])
    axs[2].set_yticks([])
    axs[2].tick_params(labelbottom=False, labelleft=False)

    axs[-1].imshow([[i] for i in list(range(len(labels)))], **label_plot_args)
    axs[-1].set_yticks(list(range(len(labels))))
    axs[-1].set_yticklabels(labels.keys())
    axs[-1].yaxis.tick_right()
    axs[-1].get_xaxis().set_visible(False)
    axs[-1].set_title("Labels")


##################################################
### Confusion matrix: Utils
##################################################

# Default values
CMAP = 'gray_r'  # or 'viridis'
VMIN = -0.25
TEXT_COLOR = 'white'
TEXT_SIZE = 8

def get_label_dict_key_index(label_dict, key):
    # Needed when your labels dont start at 0 or are not consecutive
    return list(label_dict.keys()).index(key)

def reconstruct_cm(cm_dict, label_dict):
    value_to_label_dict = {v: k for k, v in label_dict.items()}
    cm = np.zeros((len(value_to_label_dict), len(value_to_label_dict)))
    for (label_true, label_pred), count in cm_dict.items():
        idx_true = get_label_dict_key_index(value_to_label_dict, label_true)
        idx_pred = get_label_dict_key_index(value_to_label_dict, label_pred)
        cm[idx_true, idx_pred] = count
    return cm

def normalize_cm(cm, normalize_type=None):
    """Function to normalize confusion matrix."""
    if normalize_type is None:
        cm_normalized = cm
    elif normalize_type == "true":
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        cm_normalized = cm / row_sums
        cm_normalized[np.isnan(cm_normalized)] = 0
    elif normalize_type == "pred":
        col_sums = cm.sum(axis=0, keepdims=True)  
        col_sums[col_sums == 0] = 1
        cm_normalized = cm / col_sums
        cm_normalized[np.isnan(cm_normalized)] = 0
    elif normalize_type == "total":
        total_sum = np.sum(cm)
        cm_normalized = cm / total_sum
    elif normalize_type == "max":
        cm_normalized = cm / np.max(cm)
    else:
        raise ValueError("Invalid normalization type. Choose None, 'true', 'pred', 'total' or 'max'.")
    return cm_normalized

def filter_confusion_matrix(cm, labels, ignore_gt_values):
    keep_values = [value for value in labels.values() if value not in ignore_gt_values]
    filtered_cm = cm[np.ix_(keep_values, keep_values)]
    filtered_labels = {k: v for k, v in labels.items() if v not in ignore_gt_values}
    return filtered_cm, filtered_labels

def plot_confusion_matrices(cm, labels, title_list=None, ignore_gt_values=None):
    """Plot confusion matrix and its normalized versions (true and predicted)."""

    # Compute normalized matrices if not provided
    cm_total_norm = normalize_cm(cm, normalize_type="total")
    cm_true_norm = normalize_cm(cm, normalize_type="true")
    cm_pred_norm = normalize_cm(cm, normalize_type="pred")
    
    matrices = [cm_total_norm, cm_true_norm, cm_pred_norm]
    titles = title_list if title_list else ['Total Normalized', 'True Label Normalized', 'Predicted Label Normalized']
    
    # Create subplots with 3 rows and 1 column
    fig, axes = plt.subplots(3, 1, figsize=(8, 24))
    
    for ax, matrix, title in zip(axes, matrices, titles):
        matrix_color = normalize_cm(matrix, normalize_type='max')
        im = ax.imshow(matrix_color, cmap=CMAP, vmin=VMIN)
        ax.set_title(title, fontsize=16)
        ax.set_ylabel('True Labels', fontsize=12)
        ax.set_xlabel('Predicted Labels', fontsize=12)
        ax.set_yticks(range(len(labels)))
        ax.set_xticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=10)
        ax.set_xticklabels(labels, fontsize=10, rotation=90)
        
        # Add values inside the squares
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                value = matrix[i, j]
                text = f"{value:.2f}"
                ax.text(j, i, text, ha='center', va='center', color=TEXT_COLOR, fontsize=TEXT_SIZE)
    
    plt.tight_layout()

def remap_confusion_matrix(orig_cm, remap_dict):
    # Get the set of new class IDs (remapped targets)
    new_labels = sorted(set(remap_dict.values()))
    label_to_idx = {label: i for i, label in enumerate(new_labels)}
    N_new = len(new_labels)

    # Create new confusion matrix
    new_cm = np.zeros((N_new, N_new), dtype=np.int64)

    # Loop through original confusion matrix and aggregate into remapped version
    for gt_old in range(orig_cm.shape[0]):
        for pred_old in range(orig_cm.shape[1]):
            gt_new = remap_dict.get(gt_old, None)
            pred_new = remap_dict.get(pred_old, None)
            if gt_new is not None and pred_new is not None:
                i = label_to_idx[gt_new]
                j = label_to_idx[pred_new]
                new_cm[i, j] += orig_cm[gt_old, pred_old]
    
    return new_cm


##################################################
### Confusion matrix: Prep
##################################################

# roi_folder = Path('../../data/images/he')
# img_paths = list(roi_folder.glob('*_with_context.png'))

# gt_folder = Path('../../data/annotations/he')
# gt_paths = list(gt_folder.glob('*_with_context.png'))

# pred_folder = Path('../../data/inference/he')
# pred_paths = list(pred_folder.glob('*_with_context.png'))

csv_path = '../../data/data_overview.csv'
df = pd.read_csv(csv_path)

img_paths = sorted(df[(df['task'] == 'he_tissue_segmentation') & (df['split'] == 'test')]['image_path'].str.replace('.png', '_with_context.png').tolist())
gt_paths = sorted(df[(df['task'] == 'he_tissue_segmentation') & (df['split'] == 'test')]['annotation_path'].str.replace('.png', '_with_context.png').tolist())
pred_paths = sorted('../../data/inference/he/' + Path(p).name for p in img_paths)

stem_img_dict = {Path(img).stem: img for img in img_paths}
stem_gt_dict = {Path(img).stem: img for img in gt_paths}
stem_pred_dict = {Path(img).stem: img for img in pred_paths}
assert set(stem_img_dict.keys()) == stem_gt_dict.keys() == set(stem_pred_dict.keys()), "Mismatch in image stems between GT, Pred, and ROI folders."
sorted_stems = sorted(stem_gt_dict.keys())
stem_img_gt_pred_list = [(stem, stem_img_dict[stem], stem_gt_dict[stem], stem_pred_dict[stem]) for stem in sorted_stems]

plot = True # Master switch for plotting and saving
show_plot = False
save_plot = True
save_folder = Path('../../data/figures/he')
os.makedirs(save_folder, exist_ok=True)
os.makedirs(save_folder / 'cases', exist_ok=True)
overwrite = True
print('Creating images in:', save_folder)

##################################################
### Confusion matrix: Process
##################################################

print('> Generating confusion matrix data and saving case evaluation images')
cm = np.zeros((len(labels), len(labels)), dtype=int)
if not os.path.exists(os.path.join(save_folder, 'cm.npy')) or overwrite:
    for stem, img, gt, pred in tqdm(stem_img_gt_pred_list):
        gt = np.array(Image.open(gt))
        pred = np.array(Image.open(pred))

        if plot:
            bbox = get_nonzero_bbox(gt)
            img = np.array(Image.open(img))
            img_bbox = crop_to_bbox(img, bbox)
            pred_bbox = crop_to_bbox(pred, bbox)
            gt_bbox = crop_to_bbox(gt, bbox)

            plot_img_gt_pred(img_bbox, gt_bbox, pred_bbox, labels, stem)
            if save_plot:
                plt.savefig(save_folder / 'cases' / f'{stem}.png', bbox_inches='tight', dpi=150)
            if show_plot:
                plt.show()
            plt.close()

        gt_flat = gt.flatten()
        pred_flat = pred.flatten()

        # Add to overall confusion matrix
        cm += confusion_matrix(gt_flat, pred_flat, labels=list(labels.values()))

    np.save(os.path.join(save_folder, 'cm.npy'), cm)
else:
    cm = np.load(os.path.join(save_folder, 'cm.npy'))


##################################################
### Confusion matrix: All classes
##################################################

print('> Creating confusion matrix images:')
cm_non_ignore, labels_non_ignore = filter_confusion_matrix(cm, labels, [0])
plot_confusion_matrices(cm_non_ignore, labels_non_ignore)
plt.savefig(os.path.join(save_folder, 'cm_all.png'), dpi=150)
plt.close()


##################################################
### Confusion matrix: TIL use case
##################################################

cm_remapped = remap_confusion_matrix(cm, remap_dict)
cm_remapped_non_ignore, labels_remapped_non_ignore = filter_confusion_matrix(cm_remapped, labels_remapped, [0])
plot_confusion_matrices(cm_remapped_non_ignore, labels_remapped_non_ignore)
plt.savefig(os.path.join(save_folder, 'cm_til_use_case.png'), dpi=150)
plt.close()


##################################################
### F1 / Dice: Utils
##################################################

def micro_dice(confusion_matrix):
    TP = np.trace(confusion_matrix)
    FP = confusion_matrix.sum(axis=0) - np.diag(confusion_matrix)
    FN = confusion_matrix.sum(axis=1) - np.diag(confusion_matrix)

    FP_total = FP.sum()
    FN_total = FN.sum()

    denom = 2 * TP + FP_total + FN_total
    dice_micro = 2 * TP / denom if denom > 0 else 0.0
    return dice_micro

def dice_per_class(confusion_matrix):
    dice_scores = []
    for k in range(confusion_matrix.shape[0]):
        TP = confusion_matrix[k, k]
        FP = confusion_matrix[:, k].sum() - TP
        FN = confusion_matrix[k, :].sum() - TP
        denom = 2 * TP + FP + FN
        dice = 2 * TP / denom if denom > 0 else 0.0
        dice_scores.append(dice)
    return dice_scores

def get_dice_data(cm, labels):
    micro_dice_score = micro_dice(cm)
    dpc = dice_per_class(cm)
    assert len(labels) == len(dpc)
    dice_data = [["Overall", round(micro_dice_score, 2)]] + [[label, round(f1, 2)] for label, f1 in list(zip(labels.keys(), dpc))]
    return dice_data

subset_colors = {
    "Overall": (0, 0, 0),                # Black
    "Tumor cells": (0, 253/255, 255/255),      # Cyan
    "Reactive epithelium": (0, 249/255, 0),  # Green
    "Stroma": (255/255, 251/255, 0),     # Yellow
    "Inflammation": (255/255, 147/255, 0),  # Orange
    "Alveolar tissue": (255/255, 47/255, 146/255),  # Pink
    "Fatty tissue": (148/255, 55/255, 255/255),  # Purple
    "Necrotic tissue": (1/255, 25/255, 147/255),  # Dark Blue
    "Erythrocytes": (255/255, 38/255, 0),  # Red
    "Bronchial epithelium": (0, 143/255, 0),  # Green
    "Mucus/Plasma/Fluids": (146/255, 144/255, 0),  # Mustard Brown
    "Cartilage/Bone": (169/255, 169/255, 169/255),  # Gray
    "Macrophages": (83/255, 27/255, 147/255),  # Dark Purple
    "Muscle": (255/255, 126/255, 121/255),  # Light Red
    "Liver": (215/255, 131/255, 255/255),  # Light Pink
    "Keratinization": (148/255, 23/255, 81/255),  # Dark Pink
    
    "Stromal classes": (255/255, 251/255, 0),     # Yellow
    "Other": (180/255, 180/255, 180/255),   # Gray
    "Rest": (180/255, 180/255, 180/255),   # Gray
    "Stroma and Inflammation": (255/255, 199/255, 0), # blend of yellow and orange
}

def plot_dice_scores(dice_data, subset_colors, title):
    # Filter the DataFrame for only 'F1' metric
    df_f1 = pd.DataFrame(dice_data, columns=['class', 'f1'])
    df_f1 = df_f1[~df_f1['class'].isin(['Unannotated', 'Background'])]

    # Create a catplot to compare F1 values across models and subsets
    g = sns.catplot(
        data=df_f1,
        x='class', y='f1',
        kind='bar',
        height=4, aspect=2,  # Adjust plot size
    )

    # Iterate through each axis in the grid
    for ax in g.axes.flat:
        # Rotate x-tick labels for better visibility
        ax.tick_params(axis='x', labelrotation=45)
        for tick in ax.get_xticklabels():
            tick.set_ha('right')  # Align to the right
            tick.set_rotation_mode('anchor')

        # Change colors of bars based on the subset
        for bar, category in zip(ax.patches, df_f1['class'].unique()):  
            if category in subset_colors:
                bar.set_facecolor(subset_colors[category])  # Apply color
                bar.set_edgecolor('black')
                bar.set_linewidth(0.5)

        # Add values above the bars
        for bar in ax.patches:
            if bar.get_height() > 0:
                ax.annotate(
                    f"{bar.get_height():.2f}",  # Format as two decimals
                    (bar.get_x() + bar.get_width() / 2, bar.get_height()),  # Position annotation
                    ha='center', va='bottom',  # Align center and bottom
                    fontsize=8, color='black'  # Styling
                )

    # Set titles and overall formatting
    g.set_titles("{col_name}: {row_name}", size=12, weight='bold')
    g.set_axis_labels("", "F1 Score")

    plt.title(title, fontsize=16)
    plt.tight_layout()


##################################################
### F1 / Dice: All classes
##################################################

print('> Creating F1 images')
cm_dice = cm_non_ignore
cm_dice[0] = 0
labels_dice = labels_non_ignore
dice_data = get_dice_data(cm_dice, labels_dice)
plot_dice_scores(dice_data, subset_colors, "H&E Segmentation Performance on All Classes")
plt.savefig(os.path.join(save_folder, 'f1_all.png'), dpi=150)
plt.close()


##################################################
### F1 / Dice: TIL use case
##################################################

cm_dice = cm_remapped_non_ignore
cm_dice[0] = 0
labels_dice = labels_remapped_non_ignore
dice_data = get_dice_data(cm_dice, labels_dice)
plot_dice_scores(dice_data, subset_colors, "H&E Segmentation Performance on TIL Use Case")
plt.savefig(os.path.join(save_folder, 'f1_til_use_case.png'), dpi=150)
plt.close()

print('DONE')
