# Store Sales Analytics and ML

This project analyzes Superstore sales data and adds machine learning for pattern detection.

## What this project does

- Exploratory Data Analysis (EDA) on customer segments, shipping, geography, and product categories
- Customer segmentation with KMeans clustering
- Profitability prediction model (Logistic Regression)
- Visual reporting with Matplotlib and Plotly

## Dataset

- File: `data/train.csv`
- Source used: public Superstore CSV source
- The notebook is set to use a real file and does not generate synthetic rows.

## Project structure

- `notebooks/store_sales_analysis.ipynb`: main analysis notebook
- `data/train.csv`: dataset file

## Run locally

1. Create and activate virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Open the notebook in Jupyter and run cells interactively:

```bash
jupyter notebook notebooks/store_sales_analysis.ipynb
```

## Key outputs (latest run)

- Rows: 10,800
- Columns: 21
- Segment distribution:
  - Consumer: 5,191
  - Corporate: 3,020
  - Home Office: 1,783
- KMeans silhouette score: 0.265
- Profitability model accuracy: 0.7463
- Profitability model ROC-AUC: 0.6473

## Upcoming Features

- **Power BI Dashboard Version**: A Power BI version of this analysis dashboard is coming soon with interactive visualizations and real-time insights.

## Notes

- The current VS Code notebook kernel on this machine has an intermittent NumPy C-extension issue, so validation was done via terminal Python run in `.venv`.
- Notebook content and dataset are ready for GitHub.
