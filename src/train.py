import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
from sklearn.preprocessing import OrdinalEncoder

# =============================================================
# 1. TRAINING & STATS EXTRACTION PHASE
# =============================================================
print("=== TRAINING PHASE: Fitting model & extracting imputation stats ===")
train_df = pd.read_csv("data/train-test.csv")
df_train = train_df.copy()

# Clean & compute training stats for weight
df_train['weight'] = df_train['weight'].abs()
global_weight_median = df_train['weight'].median()
eq_weight_medians = df_train.groupby('equipment')['weight'].median().to_dict()

# Clean & compute training stats for market_index
df_train['date'] = pd.to_datetime(df_train['date'])
df_train['month'] = df_train['date'].dt.month
global_market_median = df_train['market_index'].median()

# Hierarchical market index medians
route_market_medians = df_train.groupby(['pickup', 'delivery'])['market_index'].median().to_dict()
route_month_market_medians = df_train.groupby(['pickup', 'delivery', 'month'])['market_index'].median().to_dict()

# Feature engineering & categorical encoding on train
if 'dayofweek' not in df_train.columns:
    df_train['dayofweek'] = df_train['date'].dt.dayofweek

features = ['distance', 'equipment', 'weight', 'market_index', 'quote_signal', 'dayofweek']
target_col = 'posted_rate'
cat_cols = ['equipment', 'dayofweek']

# Ensure categorical columns are strings before encoding
for col in cat_cols:
    df_train[col] = df_train[col].astype(str)

# Fit single OrdinalEncoder on all categorical columns simultaneously
encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
df_train[cat_cols] = encoder.fit_transform(df_train[cat_cols])

for col in features:
    df_train[col] = pd.to_numeric(df_train[col], errors='coerce').fillna(-1)

X_full = df_train[features]
y_full = df_train[target_col]

# Train LightGBM
final_model = lgb.LGBMRegressor(
    n_estimators=1000,
    learning_rate=0.03,
    num_leaves=31,
    random_state=42,
    n_jobs=-1,
    verbose=-1
)
final_model.fit(X_full, y_full)

# Bundle inference artifacts
inference_bundle = {
    'model': final_model,
    'encoder': encoder,
    'features': features,
    'cat_cols': cat_cols,
    'global_weight_median': global_weight_median,
    'eq_weight_medians': eq_weight_medians,
    'global_market_median': global_market_median,
    'route_market_medians': route_market_medians,
    'route_month_market_medians': route_month_market_medians
}

joblib.dump(inference_bundle, 'artefacts/model_bundle.pkl')
print("Model and training statistics saved successfully to 'artefacts/model_bundle.pkl'!")
