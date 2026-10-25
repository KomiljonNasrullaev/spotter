import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
from sklearn.preprocessing import OrdinalEncoder

# =============================================================
# 2. INFERENCE FUNCTION (Reusable for Batch or Single Loads)
# =============================================================
def preprocess_and_impute_inference(df, bundle):
    df = df.copy()

    # Ensure all required columns exist in the dataframe (initialize with NaN if missing)
    for col in ['weight', 'market_index', 'quote_signal', 'equipment', 'date', 'pickup', 'delivery', 'distance']:
        if col not in df.columns:
            df[col] = np.nan

    # 1. Weight Imputation using Train Stats
    df['weight'] = pd.to_numeric(df['weight'], errors='coerce').abs()
    df['weight'] = df['weight'].fillna(df['equipment'].map(bundle['eq_weight_medians']))
    df['weight'] = df['weight'].fillna(bundle['global_weight_median'])

    # Parse date / month safely
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df['month'] = df['date'].dt.month.fillna(1).astype(int)
        df['dayofweek'] = df['date'].dt.dayofweek.fillna(0).astype(int)
    else:
        df['month'] = 1
        df['dayofweek'] = 0

    # 2. Market Index Hierarchical Imputation using Train Stats
    df['market_index'] = pd.to_numeric(df['market_index'], errors='coerce')

    route_month_keys = list(zip(df['pickup'], df['delivery'], df['month']))
    df['market_index'] = df['market_index'].fillna(
        pd.Series(route_month_keys).map(bundle['route_month_market_medians'])
    )

    route_keys = list(zip(df['pickup'], df['delivery']))
    df['market_index'] = df['market_index'].fillna(
        pd.Series(route_keys).map(bundle['route_market_medians'])
    )

    df['market_index'] = df['market_index'].fillna(bundle['global_market_median'])

    # Fallback fill for other features if missing
    df['quote_signal'] = pd.to_numeric(df['quote_signal'], errors='coerce').fillna(0)
    df['distance'] = pd.to_numeric(df['distance'], errors='coerce').fillna(0)

    # 3. Categorical Encoding & Numeric Enforcement
    cat_cols = bundle['cat_cols']
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype(str)

    df[cat_cols] = bundle['encoder'].transform(df[cat_cols])

    for col in bundle['features']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(-1)

    return df[bundle['features']]

# =============================================================
# 3. EXECUTE INFERENCE ACROSS SUBMISSION FILES
# =============================================================
print("\n=== INFERENCE PHASE: Generating required submission outputs ===")
bundle = joblib.load('artefacts/model_bundle.pkl')
model = bundle['model']

# A. Validation Predictions
val_df = pd.read_csv('data/validation.csv')
template_df = pd.read_csv('data/validation-predictions-template.csv')

X_val_inf = preprocess_and_impute_inference(val_df, bundle)
val_df['predicted_rate'] = model.predict(X_val_inf)

template_df['predicted_rate'] = template_df['load_id'].map(
    val_df.set_index('load_id')['predicted_rate']
)
template_df.to_csv('data/validation_predictions.csv', index=False)
print("Generated 'data/validation_predictions.csv' successfully!")

# B. December Chart Inputs Predictions
dec_df = pd.read_csv('data/december-chart-inputs.csv')
X_dec_inf = preprocess_and_impute_inference(dec_df, bundle)
dec_df['predicted_rate'] = model.predict(X_dec_inf)

dec_df.to_csv('data/december-chart-inputs.csv', index=False)
print("Generated and populated 'data/december-chart-inputs.csv' successfully!")
