# randomized-ifs

This repository contains the implementation and evaluation of various Isolation Forest-based methods for anomaly detection, with a focus on detecting clustered anomalies. The code accompanies the survey paper attached in the *docs* folder.


### 🌴 Isolation Forests

- **Isolation Forest (IF)**: the standard isolation forest algorithm
- **Extended Isolation Forest (EIF)**: extension of isolation forest that uses hyperplanes for splitting data
- **SCiForest**: isolation forest with split-selection criterion for detecting clustered anomalies, which takes into account the distribution of the data before the partition is made
- **Fair Cut Forest (FCF)**: an improved version of SCiForest that also considers the size of each partition

Each implementation follows the algorithms described in their respective papers, with appropriate hyperparameter settings for optimal performance.


### Datasets

The experiments use three datasets known to contain clustered anomalies:

- **SpamBase**: Email classification dataset (spam as inliers, non-spam as outliers)
- **Satellite**: Satellite imagery classification transformed into a multi-modal dataset
- **Arrhythmia**: Heart disease classification dataset transformed into a multi-modal dataset


### Experimental Setup

Includes scripts to reproduce the experiments from the paper. Each algorithm uses 10 random seeds to ensure robustness, with results aggregated by calculating the mean. The configurations and the evaluation metrics used are:

- **Configuration C**: 100 trees, sample size 256, height limit 8
- **Configuration U**: 200 trees, sample size 256, no height limit
- **Evaluation metrics**: *ROC-AUC* and *PR-AUC*

### Results

The empirical evaluation shows that isolation-based approaches with guided splitting criteria (particularly FCF and SCiForest) demonstrate superior performance in identifying clustered anomalies compared to density and distance-based methods like LOF and OC-SVM.
