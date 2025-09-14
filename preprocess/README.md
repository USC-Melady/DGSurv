# Datasets

## Data Processing
Prior to the execution of any of the files, your folder structure should be in the following way:
```
DGSurv/
├── preprocess/
│   ├── data/
│   │  ├── EBPlusPlusAdjustPANCAN_IlluminaHiSeq_RNASeqV2.geneExp.tsv
│   │  ├── pancanMiRs_EBadjOnProtocolPlatformWithoutRepsWithUnCorrectMiRs_08_04_16.csv
│   │  ├── TCGA-CDR-SupplementalTableS1.xlsx
│   │  └── UNI2
│   │     ├── TCGA-KICH
│   │     ├── TCGA-KIRC
│   │     └── TCGA-KIRP
│   └── preprocessed_data/
└── ...
```
Where the source for each dataset can be found in the main `README.md` file.<br>
After this setup, you can execute the files in order mentioned in the main `README.md` file.

## Other Datasets

As mentioned in the main `README.md` file, the current code only supports the Kidney cancer dataset. However, with only minor modifications, the same code can be used for other datasets mentioned in the paper.

You can extract the Clinical data for other datasets by modifying the `dataset_index` variable in the last cell of the `clinical.ipynb` file according to the following indices:
- Bladder Cancer: `[1]` -> TCGA-BLCA
- Breast Cancer: `[2]` -> TCGA-BRCA
- Colorectal Cancer: `[5, 23]` -> TCGA-COAD, TCGA-READ
- Kidney Cancer: `[10, 11, 12]` -> TCGA-KICH, TCGA-KIRC, and TCGA-KIRP 

It is also recommended to change the save locations based on the datasets.

For mRNA and miRNA modality, the only required change is updating the `clinical_path` and the save location variable.

For the WSI modality, you need to download features for the datasets mentioned above and update the `datasets` variable in `wsi.ipynb`. For the Colorectal and Kidney dataset, you also need to run the combination step.

For splitting, you need to update the `clinical_path` and `save_path` variables in the `Args` class of the `/splits/create_splits_kidney.ipynb` notebook.

After this, you can run our proposed method or baselines by updating the variables `modality_data_path`, `input_modality_dim`, `result_path`, and `split_path` in the `Args` class of each notebook. For the `input_modality_dim` variable, the only change is for the mRNA modality dimension, you can update this variable based on the following values:
- Bladder Cancer: `'mRNA':2859`
- Breast Cancer: `'mRNA':2740`
- Colorectal Cancer: `'mRNA':1959`
- Kidney Cancer: `'mRNA':2746`

By making these modifications, you should be able to replicate the main paper results on other datasets.

# Acknowledgement
The preprocessing step was partially adapted from this [repository](https://github.com/ZhangqiJiang07/MultimodalSurvivalPrediction).