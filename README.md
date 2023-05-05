# CardiovascularDiseaseClassification
CS230 Deep Learning Project

CardioPhenoExtract.py takes in pre-selected UK Biobank data and adds a column indicating whether a person has been diagnosed with cardiovascular disease (1) or not (0). Diagnosis is based on ICD10 codes.

SHAP_DNN_baseline.ipynb further processes the tabular data, builds a deep neural network baseline (standard Sequential model), and computes the Shapley scores of the features.

XGBOOST_SHAP.ipynb also processes and takes in the tabuler data, builds and tunes an xgboost model, and computes the Shapley scores of the features.
