# Freight Rate Prediction Challenge

See `Freight_Rate_ML_Assessment.pdf` for the assessment instructions.

## What to do

1. Train and validate your model using `data/train_test.csv`.
2. Predict every load in `data/validation.csv`. Each load has a unique `load_id`.
3. Fill the matching `predicted_rate` values in `data/validation_predictions_template.csv` and save it as `validation_predictions.csv`.
4. Predict every row in `data/december_chart_inputs.csv` by filling its `predicted_rate` column.
5. Install the scorer requirements and run:

```bash
python -m pip install -r requirements.txt
python score.py --predictions data/validation_predictions.csv --december-predictions data/december_chart_inputs.csv
```

The scorer validates both files and creates `scorer_results/candidate_december.png`.

## Submit

- GitHub repository containing your code, dependencies, and run instructions
- `validation_predictions.csv`
- PDF or DOCX report containing your validation, data split approach and `candidate_december.png`
- 2-3 minute Loom link

## Run Instructions

# Freight Rate Prediction & Production Inference Pipeline

## Project Structure
- `data/`: Contains dataset files (`train-test.csv`, `validation.csv`, `december-chart-inputs.csv`, etc.).
- `src/train.py`: Extracts imputation statistics, trains the LightGBM regressor, and exports `model_bundle.pkl`.
- `src/predict.py`: Loads the model bundle, executes hierarchical imputation, and generates outputs for validation and December chart inputs.
- `requirements.txt`: Python package dependencies with exact versions.

---

## Setup & Execution Instructions

### 1. Environment Setup
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
### 2. Run Train
```bash
python src/train.py
```

output: `model_bundle.pkl`

### 3. Run Inference & Predictions

```bash
python src/predict.py
```
Generates `validation_predictions.csv` and populates `data/december-chart-inputs.csv`.
