# Breast Cancer Diagnosis Pipeline

This project provides a comprehensive machine learning pipeline for breast cancer diagnosis using various classification and regression algorithms. It includes automated data preprocessing, hyperparameter optimization, and model evaluation.

## Project Structure

```text
.
├── config/                 # Configuration files
│   └── settings.py         # Model parameters and hyperparameter grids
├── src/                    # Source code
│   ├── data_loader.py      # Data loading and preprocessing logic
│   ├── trainer.py          # Model training and tuning classes
│   └── utils.py            # Utility functions for metrics and visualization
├── main.py                 # Main entry point
├── Breast_cancer_data.csv  # Dataset
├── requirements.txt        # Project dependencies
└── README.md               # Project documentation
```

## Features

- **Automated Preprocessing**: Min-Max scaling for all features.
- **Multi-Model Support**: Supports Logistic Regression, SVC, Random Forest, Decision Trees, and Gaussian Naive Bayes.
- **Hyperparameter Tuning**: Integrated `RandomizedSearchCV` for finding optimal model parameters.
- **Visualization**: Built-in support for correlation heatmaps and confusion matrices.

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd breast-cancer-diagnosis
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the main pipeline:
```bash
python main.py
```

## Dataset

The model uses the `Breast_cancer_data.csv` dataset, which contains diagnostic features for breast cancer classification.

## Author

**Sagnik Mukherjee**  
[GitHub Profile](https://github.com/sagnik0712mukherjee)
