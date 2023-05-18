# Cardiovascular Disease Classification
## CS230 Deep Learning Project

**Note**: This repository contains only the code, not the data. Researchers can [apply](https://www.ukbiobank.ac.uk/enable-your-research/apply-for-access) to access the UK Biobank to complete health-related research that is in the public interest.

`CardioPhenoExtract.py` takes in pre-filtered UK Biobank data and adds a column indicating whether a person has been diagnosed with cardiovascular disease (1) or not (0). Diagnosis is based on ICD10 codes.

`CardiacDiseaseClassifiers_All.ipynb` is the main Jupyter notebook. It further processes the tabular data by splitting it into 12 datasets (3 input groups, 4 output label subsets). It trains an MLP deep learning baseline (standard Sequential model) and an XGBoost model (with hyperparameter tuning) for each of the 12 dataset variants. It contains an evaluation (ROC-AUC) of all 36 resulting models. SHAP feature importance analysis is also performed for the MLP and XGBoost models for the BA dataset (both sexes, any disease).

