# Telco Customer Churn Prediction

Machine learning seminar project focused on predicting customer churn using the Telco Customer Churn dataset.

## Project Structure

- `data/` - raw dataset files
- `notebooks/` - main exploratory analysis and modeling notebooks
- `reports/` - written seminar report drafts and exported artifacts
- `src/telco_churn/` - reusable project code, if the notebook grows too large

## Setup

This project uses `mise` for tool versions and `uv` for Python dependency management.

```bash
mise trust
mise install
uv sync
uv run python -m ipykernel install --user --name telco-churn --display-name "Python (telco-churn)"
uv run jupyter lab
```

The notebook kernel should be selected as `Python (telco-churn)`.

## Dataset

The current dataset is `data/telco-customer-churn.csv`, the standard flat [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) dataset with customer demographics, service subscriptions, billing information, and the `Churn` target column.
