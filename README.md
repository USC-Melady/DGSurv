# DGSurv: Dynamic Graph-Based Multimodal Learning for Interpretable Cancer Survival Prediction

Official PyTorch implementation of "DGSurv: Dynamic Graph-Based Multimodal Learning for Interpretable Cancer Survival Prediction" by 
[Sajjad Shahabi](https://scholar.google.com/citations?user=wcvGxAIAAAAJ&hl=en), 
[Zijun Cui](https://scholar.google.com/citations?user=hxkHMJIAAAAJ&hl=en), 
[Ruishan Liu](https://scholar.google.com/citations?user=7UkqY6gAAAAJ&hl=en), 
[Joseph Carlson](https://www.cityofhope.org/patients/find-a-doctor/joseph-carlson), and 
[Yan Liu](https://scholar.google.com/citations?user=UUKLPMYAAAAJ&hl=en).

<img width="1046" height="460" alt="model_overview_updated" src="https://github.com/user-attachments/assets/c9786660-27bd-44a2-9d40-08155f807ea5" />

## 1. Requirements
### Environment
The current version was tested on the following Python and CUDA versions:
- Python 3.10.11
- CUDA 11.3

We recommend using [Conda](https://anaconda.org/anaconda/conda), to setup the Python environment based on the `environment.yml` file. <br>
Use the following command to setup the evironment through Conda:
```
conda env create -f environment.yml -n int_env
```
### Dataset
To replicate the results of the experiments, please download the following datasets:
- Clinical: [TCGA - PanCanAtlas](https://api.gdc.cancer.gov/data/1b5f413e-a8d1-4d10-92eb-7c4ae739ed81)
- mRNA: [TCGA - PanCanAtlas](https://api.gdc.cancer.gov/data/3586c0da-64d0-4b74-a449-5ff4d9136611)
- miRNA: [TCGA - PanCanAtlas](https://api.gdc.cancer.gov/data/1c6174d9-8ffb-466e-b5ee-07b204c15cf8)
- WSI: [UNI2-h Pretrained Features](https://huggingface.co/datasets/MahmoodLab/UNI2-h-features)

The source for the Clinical, mRNA, and miRNA data is [Genomic Data Commons (GDC)](https://gdc.cancer.gov/about-data/publications/pancanatlas) website, and for the WSI data is [UNI](https://github.com/mahmoodlab/UNI) model github.

After downloading the datasets, move them to `/preprocess/data`.

Note that the current repository only works for Kidney cancer (KICH, KIRP, and KIRC). However, the same code can be used for training on other datasets with minimal modification. More details on how the code should be modified for other datasets are provided at `/preprocess/README.md`.

## 2. Preprocessing & Splitting
### Preprocessing
After moving the datasets to `/preprocess/data`, you can preprocess the data by executing `clinical.ipynb`, `mirna.ipynb`, `mrna.ipynb`, and `wsi.ipynb` under the `preprocess` subfolder. Note that you should execute `clinical.ipynb` before `mirna.ipynb` and `mrna.ipynb`.

### Splitting
To generate the splits used for 5-fold cross validation, run the `/splits/create_splits_kidney.ipynb` script. Note that this should be done after executing `/preprocess/clinical.ipynb` script.

## 3. Training & Validation
After preprocessing the data and generating the splits, you can train and evaluate the models by running their respective IPython notebooks.
- Proposed Method (DGSurv): `dgsurv.ipynb`
- Baselines: (Maximization)`max.ipynb`, (Attention) `attention.ipynb`, (Graph Attention) `graph_att.ipynb`

Note that the execution setting of each script can be modified by updating the `Args` class at the beginning of each file.
The results of the execution will be saved under the `/logs` folder, and you can use `/logs/plot_results.ipynb` to summarize the results.

To generate single modality results, execute `single_modality.ipynb` script and display the results using `/logs/plot_results_single.ipynb`.

# 4. Results
Table below displays the performance of our proposed method and baselines on the Kidney cancer dataset:

|        	|   Maximization  	|    Attention    	| Graph Attention 	|      DGSurv     	|
|--------	|:---------------:	|:---------------:	|:---------------:	|:---------------:	|
| Kidney 	| 0.7511 (0.0244) 	| 0.7711 (0.0208) 	| 0.7489 (0.0169) 	| 0.7725 (0.0177) 	|

And the single modality performance:

|        	|     Clinical    	|       mRNA      	|      miRNA      	|       WSI       	|
|--------	|:---------------:	|:---------------:	|:---------------:	|:---------------:	|
| Kidney 	| 0.5670 (0.0750) 	| 0.7502 (0.0313) 	| 0.7052 (0.0270) 	| 0.7304 (0.0141) 	|

# 5. Interpretability
We use a modified version of the [SHAP](https://github.com/shap/shap) package to extract feature attribution.<br>
You can execute `/interpretability/extract_shap_values.ipynb` to extract Shapley values. Note that prior to this, the model should be trained and saved at `/logs` folder. <br>
To display the extracted Shapley values, you can use the `/interpretability/plot_shap_values.ipynb` script.

<div align="center">
  
| Modality Attribution | Feature Attribution |
|---------|---------|
| <img height="270" alt="modality" src="https://github.com/user-attachments/assets/83db2fa1-a494-4923-bc95-c2a8e96f2e0e" /> | <img height="270" alt="feature" src="https://github.com/user-attachments/assets/7fe20494-90c0-49ab-a6d4-ba6b249aeae4" /> |

</div>

# 6. Citation
<!-- If you find this useful for your research, please cite the following paper: -->
Citation will be provided after publication
