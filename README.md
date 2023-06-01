# Cardiovascular Disease Classification
## CS230 Deep Learning Project

**Note**: This repository contains only the code, not the data. Researchers can [apply](https://www.ukbiobank.ac.uk/enable-your-research/apply-for-access) to access the UK Biobank to complete health-related research that is in the public interest.

`CardioPhenoBiobank.py` extracts the cardiovascular features and disease diagnoses we selected from the entire UK Biobank database. Spark SQL is used to gather the features and then after converting to a Pandas dataframe we remove missing values and consolidate any arrayed features into one column, e.g. taking the mean of four consecutive blood pressure measurements.

`CardioPhenoExtract.py` takes in pre-filtered UK Biobank data and adds a column indicating whether a person has been diagnosed with cardiovascular disease (1) or not (0). Diagnosis is based on ICD10 codes. Smoking and diabetes status are also simplified to a binary representation with a (1) if diagnosed with diabetes or a current/previous smoker and (0) if else, e.g. participant selected "prefer not to answer". A spreadsheet showing the count by sex for the four cardiovascular disease variants is also generated.

`CardiacDiseaseClassifiers_All.ipynb` is the main Jupyter notebook. It further processes the tabular data by splitting it into 12 datasets (3 input groups, 4 output label subsets). It trains an MLP deep learning baseline (standard Sequential model) and an XGBoost model (with hyperparameter tuning) for each of the 12 dataset variants. It contains an evaluation (ROC-AUC) of all 36 resulting models. SHAP feature importance analysis is also performed for the MLP and XGBoost models for the BA dataset (both sexes, any disease).

