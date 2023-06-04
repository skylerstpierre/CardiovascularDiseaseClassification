# Cardiovascular Disease Classification
## CS230 Deep Learning Project

**Note**: This repository contains only the code, not the data. Researchers can [apply](https://www.ukbiobank.ac.uk/enable-your-research/apply-for-access) to access the UK Biobank to complete health-related research that is in the public interest.

## Our contribution
We used the UK Biobank data to train sex-specific classifiers of cardiovascular disease. Three different model types were evaluated: MLP (baseline), XGBoost, and SAINT. The SAINT implementation is adapted from the article [SAINT: Improved Neural Networks for Tabular Data via Row Attention and Contrastive Pre-Training](https://arxiv.org/abs/2106.01342) with the corresponding repository: [Saint GitHub](https://github.com/somepago/saint).

We implemented the following scripts:
- `preprocess_datasets.py`
- `train_mlp_models.py`
- `train_xgb_models.py`
- `build_saint_datasets.py`
- `evaluate_all.py`
- `shap_analysis.py`
- `cross_evaluation.py`
- `aux_functions_data.py`
- `aux_functions_mlp.py`
- `aux_functions_xgb.py`

The following scripts were modified from the original [Saint repository](https://github.com/somepago/saint):
- `SAINT/train_robust.py`
- `SAINT/data_openml.py`

Our main adjustments to the SAINT implementation allow the use of custom pickled tabular datasets (with explicit train-val-test splits, and training set oversampling) and to streamline the evaluation pipeline.

## Environment
We provide an `environment.yml` file for use with `miniconda` or `anaconda`. You can create the environment required for executing the code by running

```
conda env create -f environment.yml
```

## Data preparation

`CardioPhenoBiobank.py` extracts the cardiovascular features and disease diagnoses we selected from the entire UK Biobank database. Spark SQL is used to gather the features and then after converting to a Pandas dataframe we remove missing values and consolidate any arrayed features into one column, e.g. taking the mean of four consecutive blood pressure measurements.

`CardioPhenoExtract.py` takes in pre-filtered UK Biobank data and adds a column indicating whether a person has been diagnosed with cardiovascular disease (1) or not (0). Diagnosis is based on ICD10 codes. Smoking and diabetes status are also simplified to a binary representation with a (1) if diagnosed with diabetes or a current/previous smoker and (0) if else, e.g. participant selected "prefer not to answer". A spreadsheet showing the count by sex for the four cardiovascular disease variants is also generated.

## Dataset pre-processing

`preprocess_datasets.py`

`build_saint_datasets.py`

## Classifier training
### MLP training

### XGBoost training

### SAINT training

## Classifier evaluation

## SHAP analysis

## Classifier cross-evaluation

