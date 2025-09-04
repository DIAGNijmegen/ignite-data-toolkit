# IGNITE data toolkit: a tissue and cell-level annotated H&E and PD-L1 histopathology image dataset in non-small cell lung cancer

>The tumor-immune micro-environment (TIME) in non-small cell lung cancer (NSCLC) histopathology contains morphological and molecular characteristics predictive of immunotherapy response. Computational quantification of TIME characteristics, such as cell detection and tissue segmentation, can support biomarker development. However, currently available digital pathology datasets of NSCLC for the development of cell detection or tissue segmentation algorithms are limited in scope, lack annotations of clinically prevalent metastatic sites, and forgo molecular information such as PD-L1 immunohistochemistry (IHC). To fill this gap, we introduce the IGNITE data toolkit, a multi-stain, multi-centric, and multi-scanner dataset of annotated NSCLC whole-slide images. We publicly release 887 fully annotated regions of interest from 155 unique patients across three complementary tasks: (i) multi-class semantic segmentation of tissue compartments in H&E-stained slides, with 16 classes spanning primary and metastatic NSCLC, (ii) nuclei detection, and (iii) PD-L1 positive tumor cell detection in PD-L1 IHC slides. To the best of our knowledge, this is the first public NSCLC dataset with manual annotations of H&E in metastatic sites and PD-L1.

### Repository layout

Welcome to the Github repository for the IGNITE data toolkit. Here we store the code to download all data from the associated [Zenodo data repository](https://zenodo.org/records/15674785), in addition to code used for the technical validation of the toolkit. The repository is laid out as follows:

* The [`data/`](data/) folder starts out empty and is populated with files after running the [`download_all.sh`](download_all.sh) shell script. Each of the subdirectories of [`data/`](data/) is layed like this:
```bash
.
└── data/
    └── {images,annotations,models,inference,figures}/
        ├── he/            # Files pertaining to the H&E tissue compartment segmentation dataset...
        └── pdl1/
            ├── nuclei/    # ... the PD-L1 IHC nuclei detection dataset..
            └── pdl1/      # ... and the PD-L1 positive tumor cell detection dataset
```
with `images/` containing PNG images of the regions of interest (ROIs) released in the toolkit; `annotations/` contains single-channel PNG masks for the H&E tissue compartment segmentation dataset and MS COCO-formatted JSON files for the nuclei/PD-L1 positive tumor cell detection datasets; `models/` contains the weights for our final models used for the technical validation of the toolkit. `inference/` contains raw inference of the models for the respective datasets; `figures/` contains neatly visualized inference and evaluation metric figures from our paper.

* The [`code/`](code/) contains Python code for running inference on the respective test sets of each of the three datasets.

We describe additional details regarding the datasets on our [Zenodo data repository](https://zenodo.org/records/15674785).

### Quickstart guide
1. Download all data from Zenodo by running the `download_all.sh` shell script. All data is automatically organized in the directory layout as described above.
2. To run code for running inference, build either the [H&E segmentation Docker](code/he_segmentation/docker/Dockerfile) or the [PD-L1 detection docker](code/pdl1_detection/docker/Dockerfile). Then use the scripts:
* [`code/he_segmentation/docker/run_inference_and_evaluation.sh`](code/he_segmentation/docker/run_inference_and_evaluation.sh)
* [`code/pdl1_detection/docker/run_inference_and_evaluation.sh`](code/pdl1_detection/docker/run_inference_and_evaluation.sh)


### Citation & license
This Github repository is released under the [Apache-2.0 license](LICENSE) license. The data of the IGNITE data toolkit is released under the [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) license.

If you use this dataset, please cite: Spronck, J., van Eekelen, L., et al. A tissue and cell-level annotated H&E and PD-L1 histopathology image dataset in non-small cell lung cancer. https://doi.org/10.48550/arXiv.2507.16855

```
@misc{spronck2025tissuecelllevelannotatedhe,
      title={A tissue and cell-level annotated H&E and PD-L1 histopathology image dataset in non-small cell lung cancer}, 
      author={Joey Spronck and Leander van Eekelen and Dominique van Midden and Joep Bogaerts and Leslie Tessier and Valerie Dechering and Muradije Demirel-Andishmand and Gabriel Silva de Souza and Roland Nemeth and Enrico Munari and Giuseppe Bogina and Ilaria Girolami and Albino Eccher and Balazs Acs and Ceren Boyaci and Natalie Klubickova and Monika Looijen-Salamon and Shoko Vos and Francesco Ciompi},
      year={2025},
      eprint={2507.16855},
      archivePrefix={arXiv},
      primaryClass={q-bio.QM},
      url={https://arxiv.org/abs/2507.16855}, 
}
```
