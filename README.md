# CardiovascularDiseaseClassification
## CS230 Deep Learning Project

**Note**: This repository contains only the code, not the data. Researchers can [apply](https://www.ukbiobank.ac.uk/enable-your-research/apply-for-access) to access the UK Biobank to complete health-related research.

`CardioPhenoExtract.py` takes in pre-filtered UK Biobank data and adds a column indicating whether a person has been diagnosed with cardiovascular disease (1) or not (0). Diagnosis is based on ICD10 codes.

`SHAP_DNN_baseline.ipynb` further processes the tabular data, builds a deep neural network baseline (standard Sequential model), and computes the Shapley values of the features.

`XGBOOST_SHAP.ipynb` also processes and takes in the tabuler data, builds and tunes an xgboost model, and computes the Shapley values of the features.
