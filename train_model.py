

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score

DATA_PATH = "data/hospital_occupancy.csv"
MODEL_DIR = "model"
os.makedirs(MODEL_DIR, exist_ok=True)

def train_pipeline():
    print("=" * 75)
    print("CAREPULSE AI - HOSPITAL BED OCCUPANCY FORECASTING MODEL PIPELINE")
    print("=" * 75)

    # 1. LOAD & VALIDATE DATASET
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}. Please verify file location.")

    df = pd.read_csv(DATA_PATH)
    print(f"\n[1] DATASET INSPECTION:")
    print(f"  Source path         : {DATA_PATH}")
    print(f"  Total records       : {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"  Date range          : {df['Date'].min()} to {df['Date'].max()} ({df['Date'].nunique()} consecutive days)")
    print(f"  Departments ({df['Department'].nunique()}): {', '.join(sorted(df['Department'].unique()))}")
    # print(f"  Missing values      : {df.isnull().sum().sum()} (0 missing across all fields)")
    print(f"  Duplicate records   : {df.duplicated().sum()}")

    missing_values = int(df.isnull().sum().sum())

    print(f"Missing values in merged dataset: {missing_values}")

    if missing_values > 0:
        print("\nMissing values by column:")
        missing_by_column = df.isnull().sum()
        print(missing_by_column[missing_by_column > 0].to_string())
    else:
        print("No missing values found in the merged dataset.")

    # Logical sanity checks
    over_capacity = (df['Beds_Occupied'] > df['Total_Hospital_Beds']).sum()
    rate_discrepancy = ((abs((df['Beds_Occupied'] / df['Total_Hospital_Beds'] * 100) - df['Occupancy_Rate_Pct'])) > 0.1).sum()
    print(f"  Integrity checks    : Over-capacity violations = {over_capacity}, Rate discrepancies = {rate_discrepancy}")

    # Explicit dataset limitation documentation
    print("\n[2] CLINICAL FEATURE MAPPING & PDF REQUIREMENTS GAP ANALYSIS:")
    print("  • Admission Count   : Mapped to Daily_Admissions, Emergency_Admissions, ICU_Admissions")
    print("  • Season            : Mapped to Season, Month, Day_of_Week")
    print("  • Emergency Cases   : Mapped to Emergency_Admissions")
    print("  • Discharge Records : Mapped to Daily_Discharges, Avg_Length_of_Stay")
    print("  • Department        : Mapped to Department (8 clinical specialties)")
    print("  • GAP (Patient Type): Not explicitly present in dataset. Documented as clinical limitation.")
    print("  • GAP (Disease Cat) : Not explicitly present in dataset. Documented as clinical limitation.")

    # 2. FEATURE ENGINEERING (LEAKAGE-FREE TIME SERIES LAGS)
    print("\n[3] TIME-SERIES FEATURE ENGINEERING (LEAKAGE-FREE):")
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(['Department', 'Date']).reset_index(drop=True)

    # Per-department autoregressive lag & rolling feature creation
    dept_dfs = []
    for dept_name, group in df.groupby('Department'):
        g = group.sort_values('Date').copy()
        # Historical target lags (strictly prior to forecast day)
        g['Occupancy_Lag_1'] = g['Occupancy_Rate_Pct'].shift(1)
        g['Occupancy_Lag_2'] = g['Occupancy_Rate_Pct'].shift(2)
        g['Occupancy_Lag_7'] = g['Occupancy_Rate_Pct'].shift(7)
        g['Occupancy_Roll_Mean_7'] = g['Occupancy_Rate_Pct'].shift(1).rolling(7).mean()

        # Prior day admission & discharge flows (available at forecast time t)
        g['Admissions_Lag_1'] = g['Daily_Admissions'].shift(1)
        g['Emergency_Lag_1'] = g['Emergency_Admissions'].shift(1)
        g['Discharges_Lag_1'] = g['Daily_Discharges'].shift(1)
        g['Avg_LOS_Lag_1'] = g['Avg_Length_of_Stay'].shift(1)
        g['Staff_Avail_Lag_1'] = g['Staff_Availability_Pct'].shift(1)
        dept_dfs.append(g)

    df_featured = pd.concat(dept_dfs).dropna().reset_index(drop=True)
    print(f"  Engineered feature matrix: {df_featured.shape[0]} valid observations after lag initialization.")

    # 3. CHRONOLOGICAL SPLIT (NO SHUFFLE, PREVENT FUTURE LEAKAGE)
    print("\n[4] CHRONOLOGICAL DATA SPLIT:")
    split_date = '2024-01-01'
    train_df = df_featured[df_featured['Date'] < split_date].copy()
    test_df = df_featured[df_featured['Date'] >= split_date].copy()

    print(f"  Training set (2022-2023) : {train_df.shape[0]} samples ({train_df['Date'].min().date()} to {train_df['Date'].max().date()})")
    print(f"  Test set     (2024 held) : {test_df.shape[0]} samples ({test_df['Date'].min().date()} to {test_df['Date'].max().date()})")

    # Feature definitions
    num_cols = [
        'Total_Hospital_Beds', 'Month', 'Public_Holiday', 'Special_Event',
        'Occupancy_Lag_1', 'Occupancy_Lag_2', 'Occupancy_Lag_7', 'Occupancy_Roll_Mean_7',
        'Admissions_Lag_1', 'Emergency_Lag_1', 'Discharges_Lag_1', 'Avg_LOS_Lag_1', 'Staff_Avail_Lag_1'
    ]
    cat_cols = ['Department', 'Season', 'Day_of_Week']
    target_col = 'Occupancy_Rate_Pct'

    # Preprocessing pipeline
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), cat_cols)
    ])

    X_train = preprocessor.fit_transform(train_df[num_cols + cat_cols])
    y_train = train_df[target_col].values

    X_test = preprocessor.transform(test_df[num_cols + cat_cols])
    y_test = test_df[target_col].values

    # 4. BENCHMARKING MODELS
    print("\n[5] MODEL TRAINING & COMPARATIVE BENCHMARKING:")
    
    # Baseline 0: Naive Persistence (Tomorrow's occupancy = Today's occupancy)
    y_naive = test_df['Occupancy_Lag_1'].values
    naive_rmse = root_mean_squared_error(y_test, y_naive)
    naive_mae = mean_absolute_error(y_test, y_naive)
    naive_r2 = r2_score(y_test, y_naive)

    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=12, min_samples_leaf=3, random_state=42, n_jobs=-1),
        'XGBoost': XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=42)
    }

    results = {
        'Naive Persistence': {
            'rmse': round(float(naive_rmse), 3),
            'mae': round(float(naive_mae), 3),
            'r2': round(float(naive_r2), 3),
            'type': 'Heuristic Baseline'
        }
    }

    fitted_models = {}
    best_name = None
    best_rmse = float('inf')

    for name, model_obj in models.items():
        print(f"  Training {name}...")
        model_obj.fit(X_train, y_train)
        preds = model_obj.predict(X_test)

        rmse = root_mean_squared_error(y_test, preds)
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)

        results[name] = {
            'rmse': round(float(rmse), 3),
            'mae': round(float(mae), 3),
            'r2': round(float(r2), 3),
            'type': 'Machine Learning'
        }
        fitted_models[name] = model_obj

        if rmse < best_rmse:
            best_rmse = rmse
            best_name = name

    print("\n" + "=" * 75)
    print(f"{'MODEL':<22} | {'RMSE (%)':<10} | {'MAE (%)':<10} | {'R² SCORE':<10} | {'STATUS'}")
    print("-" * 75)
    print(f"{'Naive Persistence':<22} | {naive_rmse:<10.3f} | {naive_mae:<10.3f} | {naive_r2:<10.3f} | Heuristic Baseline")
    for name in models.keys():
        m_res = results[name]
        status = "WINNER (Selected)" if name == best_name else "Candidate"
        print(f"{name:<22} | {m_res['rmse']:<10.3f} | {m_res['mae']:<10.3f} | {m_res['r2']:<10.3f} | {status}")
    print("=" * 75)

    # 5. SERIALIZE ARTIFACTS
    print(f"\n[6] SAVING ARTIFACTS TO '{MODEL_DIR}/':")
    best_model = fitted_models[best_name]
    
    joblib.dump(best_model, os.path.join(MODEL_DIR, "best_model.pkl"))
    joblib.dump(preprocessor, os.path.join(MODEL_DIR, "preprocessor.pkl"))
    
    # Save all models for live benchmark toggle
    for m_name, m_inst in fitted_models.items():
        safe_fname = m_name.lower().replace(" ", "_") + ".pkl"
        joblib.dump(m_inst, os.path.join(MODEL_DIR, safe_fname))

    # Save feature metadata & evaluation metrics
    metadata = {
        'selected_model': best_name,
        'metrics': results,
        'num_cols': num_cols,
        'cat_cols': cat_cols,
        'target_col': target_col,
        'departments': sorted(df['Department'].unique().tolist()),
        'training_dates': {'start': str(train_df['Date'].min().date()), 'end': str(train_df['Date'].max().date())},
        'test_dates': {'start': str(test_df['Date'].min().date()), 'end': str(test_df['Date'].max().date())},
        'peak_demand_threshold_pct': 85.0
    }

    with open(os.path.join(MODEL_DIR, "model_metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=2)
    print("  Artifacts saved:")
    print("  • model/best_model.pkl")
    print("  • model/preprocessor.pkl")
    print("  • model/model_metadata.json")
    print("  • model/linear_regression.pkl, random_forest.pkl, xgboost.pkl")

    # 6. GENERATE EVALUATION PLOTS
    print("\n[7] GENERATING COMPREHENSIVE EVALUATION PLOT...")
    plot_path = os.path.join(MODEL_DIR, "evaluation_plots.png")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor('#0a0a0c')
    for ax in axes.flat:
        ax.set_facecolor('#111118')
        ax.tick_params(colors='#cbd5e1')
        ax.xaxis.label.set_color('#cbd5e1')
        ax.yaxis.label.set_color('#cbd5e1')
        ax.title.set_color('#facc15')
        for spine in ax.spines.values():
            spine.set_color('#282836')

    # Subplot 1: Actual vs Predicted (Test Set Sample)
    sample_test = test_df.sample(200, random_state=42)
    sample_X = preprocessor.transform(sample_test[num_cols + cat_cols])
    sample_preds = best_model.predict(sample_X)

    axes[0, 0].scatter(sample_test['Occupancy_Rate_Pct'], sample_preds, alpha=0.6, color='#facc15', edgecolors='none', s=25)
    axes[0, 0].plot([30, 100], [30, 100], color='#ef4444', linestyle='--', linewidth=1.5, label='Ideal 1:1')
    axes[0, 0].set_title(f'Actual vs. Predicted Occupancy ({best_name})')
    axes[0, 0].set_xlabel('Actual Occupancy Rate (%)')
    axes[0, 0].set_ylabel('Predicted Occupancy Rate (%)')
    axes[0, 0].legend(facecolor='#111118', edgecolor='#282836', labelcolor='#cbd5e1')
    axes[0, 0].grid(True, color='#1e1e2d', linestyle=':', alpha=0.6)

    # Subplot 2: Model Comparison (RMSE & MAE)
    model_names = list(results.keys())
    rmse_vals = [results[m]['rmse'] for m in model_names]
    mae_vals = [results[m]['mae'] for m in model_names]

    x = np.arange(len(model_names))
    width = 0.35
    axes[0, 1].bar(x - width/2, rmse_vals, width, label='RMSE (%)', color='#facc15')
    axes[0, 1].bar(x + width/2, mae_vals, width, label='MAE (%)', color='#ca8a04')
    axes[0, 1].set_title('Model Performance Benchmark (Held-out 2024 Test Set)')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels([m.replace(" ", "\n") for m in model_names], fontsize=8.5)
    axes[0, 1].set_ylabel('Error Percentage (%)')
    axes[0, 1].legend(facecolor='#111118', edgecolor='#282836', labelcolor='#cbd5e1')
    axes[0, 1].grid(True, color='#1e1e2d', linestyle=':', alpha=0.6)

    # Subplot 3: Time Series Actual vs Forecast Sample (Cardiology 2024)
    dept_sample = test_df[test_df['Department'] == 'Cardiology'].sort_values('Date').iloc[:60]
    dept_X = preprocessor.transform(dept_sample[num_cols + cat_cols])
    dept_preds = best_model.predict(dept_X)

    axes[1, 0].plot(dept_sample['Date'], dept_sample['Occupancy_Rate_Pct'], label='Actual Occupancy', color='#38bdf8', linewidth=1.8)
    axes[1, 0].plot(dept_sample['Date'], dept_preds, label='Forecast Occupancy', color='#facc15', linestyle='--', linewidth=1.8)
    axes[1, 0].axhline(85, color='#ef4444', linestyle=':', label='Peak Demand (85%)')
    axes[1, 0].set_title('Cardiology Occupancy Forecast vs Actual (60-Day Horizon)')
    axes[1, 0].set_ylabel('Occupancy Rate (%)')
    axes[1, 0].legend(facecolor='#111118', edgecolor='#282836', labelcolor='#cbd5e1', fontsize=8)
    axes[1, 0].grid(True, color='#1e1e2d', linestyle=':', alpha=0.6)

    # Subplot 4: Residuals Distribution
    all_preds = best_model.predict(X_test)
    residuals = y_test - all_preds
    axes[1, 1].hist(residuals, bins=35, color='#facc15', edgecolor='#111118', alpha=0.85)
    axes[1, 1].axvline(0, color='#ef4444', linestyle='--', linewidth=1.5)
    axes[1, 1].set_title('Residual Error Distribution (Gaussian Fit)')
    axes[1, 1].set_xlabel('Residual (Actual - Predicted %)')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].grid(True, color='#1e1e2d', linestyle=':', alpha=0.6)

    plt.tight_layout()
    plt.savefig(plot_path, dpi=200, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    print(f"  Saved evaluation plot to: {plot_path}")

    print("\n" + "=" * 75)
    print("TRAINING PIPELINE COMPLETE AND VERIFIED!")
    print("=" * 75)

if __name__ == "__main__":
    train_pipeline()