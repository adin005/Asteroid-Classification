# Explainable Deep Learning-Based Potentially Hazardous Asteroid Classification

## Project Overview

This project focuses on classifying potentially hazardous asteroids using machine learning and deep learning techniques.

The project uses orbital and physical asteroid features to predict whether an asteroid belongs to the potentially hazardous class.

The main Python implementation is contained in:

`asteroid_code.py`

## Dataset

The project uses a dataset stored locally as:

`dataset.csv`

The dataset contains approximately 200,000 samples and includes the following 9 features:

* e
* a
* q
* i
* tp
* H
* diameter
* albedo
* moid

The dataset file is not included in this GitHub repository because it is approximately 436 MB in size.

## Models

The project experiments with several classification approaches:

* GNN with SMOTE
* GNN with tuned classification threshold
* Original GNN
* MLP
* GAT
* Isolation Forest
* Random Forest

SMOTE is used in some experiments to address class imbalance.

## Results

The project stores model evaluation results in:

`results/metrics_summary.csv`

The results include:

* Precision
* Recall
* F1-score
* Accuracy
* ROC-AUC
* PR-AUC
* True Positives
* False Positives
* False Negatives
* True Negatives
* Classification threshold

The project also generates visual results including:

* Feature distributions
* Class distribution after SMOTE
* Confusion matrix
* ROC and Precision-Recall curves
* Training curves
* Feature importance
* Comparison with previous work

These figures are stored in:

`results/figures/`

## Repository Structure

```text
Asteroid_Dataset/
│
├── asteroid_code.py
├── README.md
├── requirements.txt
├── .gitignore
│
└── results/
    ├── metrics_summary.csv
    └── figures/
        ├── fig1_feature_distributions.png
        ├── fig2_class_distribution_smote.png
        ├── fig3_confusion_matrix.png
        ├── fig4_roc_and_pr_curves.png
        ├── fig5_training_curves.png
        ├── fig6_feature_importance.png
        └── fig7_comparison_with_paper.png
```

The local dataset, virtual environment, and saved model weights are excluded from Git tracking.

## Purpose of This Repository

This repository is being used for a DevOps laboratory focused on collaborative machine learning development using GitHub.

The lab demonstrates:

* Git version control
* GitHub repositories
* Branching
* Pull Requests
* Collaborative development
* Model performance tracking

## Technologies

* Python
* Graph Neural Networks
* Scikit-learn
* GitHub
* GitHub Actions
* Git
* Deep Learning
* Machine Learning
* GitHub Issues
