# #### Resources used:
# - **UK Biobank (data)**: https://www.ukbiobank.ac.uk/
# - **Tensorflow**: https://www.tensorflow.org/tutorials
# - **Keras**: https://keras.io/examples/
# - **XGBoost**: repo: https://github.com/dmlc/xgboost, doc: https://xgboost.readthedocs.io/en/stable/tutorials/index.html
# - **SHAP**: https://shap.readthedocs.io/en/latest/tabular_examples.html
# - **joblib Parallel**: https://joblib.readthedocs.io/en/latest/generated/joblib.Parallel.html
# - **scikit-learn**: https://scikit-learn.org/stable/modules/classes.html

#######################
###### Packages #######
#######################
import numpy as np
import pandas as pd
import tensorflow as tf
import seaborn as sns
# import keras_tuner as kt
import xgboost as xgb
# import shap
from joblib import Parallel, delayed
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV

import openml
from openml.datasets.functions import create_dataset

from matplotlib import pyplot as plt
import pickle

from aux_functions_data import preprocess_df, buildXY, oversample_train, save_dataset_saint
from aux_functions_mlp import build_baseline_MLP
######################################

####################################
###### Data and model import #######
####################################
N_DATASETS = 12

# Load the relevant datasets
print("Loading preprocessed datasets...")
with open('preprocessed_datasets/data_all.pkl', 'rb') as f:
    features_df, features, targets = pickle.load(f)
print("Done.")

# Contextual row-wise and column-wise index bounds: 
models_I = len(features) # Row-wise: for the number of sex-specific splits
models_J = len(targets[0]) # Column-wise: for the number of disease-specific subset splits

# Load the 12 MLP models
print("Loading MLP models...")
models_mlp = []
for i in range(N_DATASETS) :
    models_mlp.append([tf.keras.models.load_model('models_MLP/mlp' + str(i)), 
                       tf.data.Dataset.load('models_MLP/test_ds' + str(i))])
print("Done.")

# Load the 36 XGBoost models (24 (12 untuned + 12 tuned) for all features, 12 tuned for Framingham only)
# and the corresponding 24 (12 all features + 12 Framingham) datasets
print("Loading XGBoost models and datasets...")
xgb_classifiers = [xgb.XGBClassifier(objective = 'binary:logistic') for i in range(N_DATASETS)]
[xgb_classifiers[i].load_model("models_XGB/untuned/model" + str(i) + ".json") for i in range(N_DATASETS)]
xgb_classifiers_tunedF = [xgb.XGBClassifier(objective = 'binary:logistic') for i in range(N_DATASETS)]
[xgb_classifiers_tunedF[i].load_model("models_XGB_Fram/tuned/model" + str(i) + ".json") for i in range(N_DATASETS)]
xgb_classifiers_tuned = [xgb.XGBClassifier(objective = 'binary:logistic') for i in range(N_DATASETS)]
[xgb_classifiers_tuned[i].load_model("models_XGB/tuned/model" + str(i) + ".json") for i in range(N_DATASETS)]

XY_xgb = []
XY_xgbF = []
for i in range(N_DATASETS) :
    with open('models_XGB/datasets/dataset' + str(i) + '.pkl', 'rb') as f:
        XY = pickle.load(f)
        XY_xgb.append(XY)
    with open('models_XGB_Fram/datasets/dataset' + str(i) + '.pkl', 'rb') as f:
        XYF = pickle.load(f)
        XY_xgbF.append(XYF)
print("Done.")

##################################
###### Evaluate all models #######
##################################
################
## MLP models ##
################
print("Evaluating MLP models...")
# Make predictions on the corresponding TEST SET
print("Computing predictions and extracting ground truth on the test set...")
y_pred = [mlp[0].predict(mlp[1], verbose=0) for mlp in models_mlp]
y_pred = np.reshape(y_pred, (models_I, models_J)) # Reshape collection to 2D model grid, as before
print("Done.")

# Extract the TEST SET ground truth
XY_test = [tuple(zip(*mlp[1])) for mlp in models_mlp]
y_true = [np.array(xy_test[1]) for xy_test in XY_test]
y_true = np.reshape(y_true, (models_I, models_J))

# Plot the ROC curve and report the AUC metric for all 12 MLPs
print("Plotting the ROC curves...")
fig, ax = plt.subplots(models_I, models_J, sharex=True, sharey=True, dpi = 160, figsize=(12, 8.5))

for i in range(models_I) :
    for j in range(models_J) :
        fpr, tpr, thresholds = roc_curve(y_true[i, j], y_pred[i, j]) # ROC curve
        ax[i, j].plot(fpr, tpr)
        
        # Formatting
        ax[i, j].plot(np.linspace(0,1, 100), np.linspace(0,1,100),'--r')
        ax[i, j].set_xlabel(('FPR' if i == 2 else ''), fontsize = 12)
        ax[i, j].set_ylabel(('TPR' if j == 0 else ''), fontsize = 12)
        ax[i, j].set(ylim = [0., 1.])
        ax[i, j].set_aspect('equal', 'box')
        ax[i, j].text(0.5, 0.2, f'AUC = {roc_auc_score(y_true[i, j], y_pred[i, j]):.3f}', fontsize = 12) # AUC metric

fig.tight_layout()
print("Done.")

# Save fig.
print("Saving figure...")
fig.savefig("figures/ROC_MLP.png", dpi = 600) 
print("Done.")
print("Done evaluating MLP.")

####################
## XGBoost models ##
####################
## Untuned ##
print("Evaluating XGBoost models (untuned)...")

# Reshape the comprehensive XGBoost model collection 
# and the (untuned) trained XGBoost classifier collection as 3x4 grids
# (rows = sex-specific splits, columns = disease specific splits) 
XY_xgb_r = np.reshape(XY_xgb, (models_I, models_J, 4))
xgb_classifiers_r = np.reshape(xgb_classifiers, (models_I, models_J))

# Plot the ROC curve and report the AUC metric for all 12 untuned XGBoost models
print("Plotting the ROC curves...")
fig, ax = plt.subplots(models_I, models_J, sharex=True, sharey=True, dpi = 160, figsize=(12, 8.5))
for i in range(models_I) :
    for j in range(models_J) :
        # Make XGBoost predictions on the test set
        xgb_cl = xgb_classifiers_r[i, j]
        x_test_xgb = XY_xgb_r[i, j][1]
        y_true_xgb = XY_xgb_r[i, j][3]
        y_pred_xgb = xgb_cl.predict_proba(x_test_xgb)
        
        # ROC curve
        fpr, tpr, thresholds = roc_curve(y_true_xgb, y_pred_xgb[:,1])
        ax[i, j].plot(fpr, tpr)
        
        # Formatting
        ax[i, j].plot(np.linspace(0,1, 100), np.linspace(0,1,100),'--r')
        ax[i, j].set_xlabel(('FPR' if i == 2 else ''), fontsize = 12)
        ax[i, j].set_ylabel(('TPR' if j == 0 else ''), fontsize = 12)
        ax[i, j].set(ylim = [0., 1.])
        ax[i, j].set_aspect('equal', 'box')
        ax[i, j].text(0.5, 0.2, f'AUC = {roc_auc_score(y_true_xgb, y_pred_xgb[:,1]):.3f}', fontsize = 12) # AUC metric
        
fig.tight_layout()
print("Done.")

# Save fig.
print("Saving figure...")
fig.savefig("figures/ROC_XGB_Untuned.png", dpi = 600)
print("Done.")
print("Done evaluating XGBoost (untuned).")

## Tntuned ##
print("Evaluating XGBoost models (tuned)...")

# Reshape the comprehensive XGBoost model collection 
# and the (tuned) trained XGBoost classifier collection as 3x4 grids
# (rows = sex-specific splits, columns = disease specific splits) 
XY_xgb_r = np.reshape(XY_xgb, (models_I, models_J, 4))
XY_xgb_rF = np.reshape(XY_xgbF, (models_I, models_J, 4))
xgb_classifiers_tuned_r = np.reshape(xgb_classifiers_tuned, (models_I, models_J))
xgb_classifiers_tuned_rF = np.reshape(xgb_classifiers_tunedF, (models_I, models_J))

# Plot the ROC curve and report the AUC metric for all 24 individually tuned XGBoost models 
# (12 for all features, and 12 for Framingham)
print("Plotting the ROC curves...")
fig, ax = plt.subplots(models_I, models_J, sharex=True, sharey=True, dpi = 160, figsize=(12, 8.5))
for i in range(models_I) :
    for j in range(models_J) :
        # Make XGBoost predictions on the test set
        x_test_xgb = XY_xgb_r[i, j][1]
        y_true_xgb = XY_xgb_r[i, j][3]
        xgb_cl_tuned = xgb_classifiers_tuned_r[i, j]
        y_pred_xgb_tuned = xgb_cl_tuned.predict_proba(x_test_xgb)
        # framingham only
        x_test_xgbF = XY_xgb_rF[i, j][1]
        y_true_xgbF = XY_xgb_rF[i, j][3]
        xgb_cl_tunedF = xgb_classifiers_tuned_rF[i, j]
        y_pred_xgb_tunedF = xgb_cl_tunedF.predict_proba(x_test_xgbF)
        
        # ROC curve
        fpr, tpr, thresholds = roc_curve(y_true_xgb, y_pred_xgb_tuned[:,1])
        ax[i, j].plot(fpr, tpr)
        fprF, tprF, thresholds = roc_curve(y_true_xgbF, y_pred_xgb_tunedF[:,1])
        ax[i, j].plot(fprF, tprF)
        
        # Formatting
        ax[i, j].plot(np.linspace(0, 1, 100), np.linspace(0,1,100),'--r')
        ax[i, j].set_xlabel(('FPR' if i == 2 else ''), fontsize = 12)
        ax[i, j].set_ylabel(('TPR' if j == 0 else ''), fontsize = 12)
        ax[i, j].set(ylim = [0., 1.])
        ax[i, j].set_aspect('equal', 'box')
        ax[i, j].text(0.3, 0.2, f'AUC = {roc_auc_score(y_true_xgb, y_pred_xgb_tuned[:,1]):.3f}', fontsize = 12)
        ax[i, j].text(0.3, 0.1, f'AUC Fram = {roc_auc_score(y_true_xgbF, y_pred_xgb_tunedF[:,1]):.3f}', fontsize = 12) # AUC metric
        
fig.tight_layout()
print("Done.")

# Save fig.
print("Saving figure...")
fig.savefig("figures/ROC_XGB_Tuned.png", dpi = 600)
print("Done.")
print("Done evaluating XGBoost (tuned).")

##################
## SAINT models ##
##################


##############################
## All models in one figure ##
##############################
print("Plotting the ROC curves for all 60 models in one figure...")
fig, ax = plt.subplots(models_I, models_J, sharex=True, sharey=True, dpi = 160, figsize=(12, 8.5))
fig.tight_layout()
for i in range(models_I) :
    for j in range(models_J) :
        # Baseline MLPs
        fpr, tpr, thresholds = roc_curve(y_true[i, j], y_pred[i, j]) # ROC curve
        ax[i, j].plot(fpr, tpr)
        
        # XGBoost (Untuned)
        xgb_cl = xgb_classifiers_r[i, j]
        x_test_xgb = XY_xgb_r[i, j][1]
        y_true_xgb = XY_xgb_r[i, j][3]
        y_pred_xgb = xgb_cl.predict_proba(x_test_xgb) # XGB test set predictions
        fpr, tpr, thresholds = roc_curve(y_true_xgb, y_pred_xgb[:,1]) # ROC curve
        ax[i, j].plot(fpr, tpr)
        
        # XGBoost (Tuned)
        xgb_cl_tuned = xgb_classifiers_tuned_r[i, j]
        y_pred_xgb_tuned = xgb_cl_tuned.predict_proba(x_test_xgb) # XGB test set predictions
        fpr, tpr, thresholds = roc_curve(y_true_xgb, y_pred_xgb_tuned[:,1]) # ROC curve
        ax[i, j].plot(fpr, tpr)
        
        # XGBoost (Tuned, Framingham Only)
        xgb_cl_tunedF = xgb_classifiers_tuned_rF[i, j]
        x_test_xgbF = XY_xgb_rF[i, j][1]
        y_true_xgbF = XY_xgb_rF[i, j][3]
        y_pred_xgb_tunedF = xgb_cl_tunedF.predict_proba(x_test_xgbF) # XGB test set predictions
        fpr, tpr, thresholds = roc_curve(y_true_xgbF, y_pred_xgb_tunedF[:,1]) # ROC curve
        ax[i, j].plot(fpr, tpr)
        
        # Formatting
        ax[i, j].plot(np.linspace(0, 1, 100), np.linspace(0,1,100),'--r')
        ax[i, j].set_xlabel(('FPR' if i == 2 else ''), fontsize = 12)
        ax[i, j].set_ylabel(('TPR' if j == 0 else ''), fontsize = 12)
        ax[i, j].set(ylim = [0., 1.])
        ax[i, j].set_aspect('equal', 'box')
        c = plt.rcParams['axes.prop_cycle'].by_key()['color'][0:3]
        ax[i, j].text(0.35, 0.31, f'AUC MLP = {roc_auc_score(y_true[i, j], y_pred[i, j]):.3f}', fontsize = 12, color = c[0]) # AUC metric
        ax[i, j].text(0.35, 0.24, f'AUC XGBu = {roc_auc_score(y_true_xgb, y_pred_xgb[:,1]):.3f}', fontsize = 12, color = c[1]) # AUC metric
        ax[i, j].text(0.35, 0.17, f'AUC XGBt = {roc_auc_score(y_true_xgb, y_pred_xgb_tuned[:,1]):.3f}', fontsize = 12, color = c[2]) # AUC metric
        ax[i, j].text(0.35, 0.1, f'AUC XGB Fram = {roc_auc_score(y_true_xgbF, y_pred_xgb_tunedF[:,1]):.3f}', fontsize = 12, color = c[2]) # AUC metric
        ax[i, j].set(xlabel = ('FPR' if i == 2 else ''), ylabel = ('TPR' if j == 0 else ''), ylim = [0., 1.])

fig.tight_layout()
print("Done.")

# Save fig.
print("Saving figure...")
fig.savefig("figures/ROC_Comparison.png", dpi = 600)
print("Done.")