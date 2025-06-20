from pathlib import Path
from PIL import Image
import pandas as pd
import numpy as np
from tqdm import tqdm
from nnunet.training.model_restore import load_model_and_checkpoint_files

print('RUNNING INFERENCE SCRIPT')


##################################################
### Import model
##################################################

def norm_01(x_batch): # Use this for models trained on 0-1 scaled data
    x_batch = x_batch / 255
    x_batch = x_batch.transpose(3, 0, 1, 2)
    return x_batch

def ensemble_softmax_list(trainer, params, x_batch):
    softmax_list = []
    for p in params:
        trainer.load_checkpoint_ram(p, False)
        softmax_list.append(
            trainer.predict_preprocessed_data_return_seg_and_softmax(x_batch.astype(np.float32), verbose=False,
                                                                     do_mirroring=False, mirror_axes=[])[
                -1].transpose(1, 2, 3, 0).squeeze())
    return softmax_list

model_base_path = '../../data/models/he/nnUNetTrainerV2_BN_pathology_DA_ignore0_hed005__nnUNet_RGB_scaleTo_0_1_bs8_ps512'
norm = norm_01 
output_minus_1 = False 

print('\n> Model path:')
print(model_base_path, '\n')

folds = (0, 1, 2, 3, 4)
mixed_precision = None
checkpoint_name = "model_best"

print('> Loading model')
trainer, params = load_model_and_checkpoint_files(model_base_path, folds, mixed_precision=mixed_precision,
                                                  checkpoint_name=checkpoint_name)


##################################################
### Run inference
##################################################

print('> Running inference')
print('Saving inference in:', "../../data/inference/he/")

csv_path = '../../data/data_overview.csv'
df = pd.read_csv(csv_path)
inference_images = sorted(df[(df['task'] == 'he_tissue_segmentation') & (df['split'] == 'test')]['image_path'].str.replace('.png', '_with_context.png').tolist())

for i in tqdm(inference_images):
    stem = Path(i).stem

    image = Image.open(i)
    i = np.array(image)
    i = np.expand_dims(i, axis=0)
    i = norm(i)

    softmax_list = ensemble_softmax_list(trainer, params, i)
    softmax_mean = np.array(softmax_list).mean(0)
    pred_output = softmax_mean.argmax(axis=-1)-(1 if output_minus_1 else 0)
    Image.fromarray(pred_output.astype(np.uint8)).save(f'../../data/inference/he/{stem}.png')

print('DONE')
