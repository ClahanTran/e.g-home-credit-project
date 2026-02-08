# Home Credit Feature Engineering Pipeline

**Author:** Clahan Tran  
**Date:** February 7, 2026  
**Version:** 1.0

## Overview

This feature engineering pipeline provides production-ready, reusable functions for cleaning, transforming, and enriching the Home Credit Default Risk dataset. The pipeline ensures train/test consistency and implements best practices identified during exploratory data analysis (EDA).

## Files

- **`feature_engineering.py`** - Main module with all feature engineering functions
- **`test_feature_engineering.py`** - Test suite demonstrating usage and validation
- **`FEATURE_ENGINEERING_README.md`** - This documentation file

## Key Features

✅ **Data Cleaning**
- Fixes DAYS_EMPLOYED = 365243 anomaly (employment status placeholder)
- Imputes missing values in EXT_SOURCE variables (strongest predictors)
- Handles missing data strategically with median/mode imputation
- Creates missing value indicators

✅ **Feature Engineering**
- Demographic features (age, employment years)
- Financial ratios (credit-to-income, loan-to-value, DTI, etc.)
- Interaction terms (EXT_SOURCE products, age/income bins)
- Missing indicators and document completeness scores

✅ **Supplementary Data Aggregation**
- Bureau credit history (counts, debt ratios, overdue amounts)
- Previous applications (approval rates, refusal history)
- Installment payments (late payment rates, payment trends)
- Credit card balances (utilization, payment patterns)
- POS cash transactions (completion rates, DPD statistics)

✅ **Train/Test Consistency**
- `FeatureConfig` class stores training statistics
- Test data uses same medians/modes as training data
- Ensures identical columns in train and test (except TARGET)

## Quick Start

### Basic Usage

```python
import polars as pl
from feature_engineering import process_pipeline, save_processed_data

# Load data
app_train = pl.read_csv('application_train.csv')
app_test = pl.read_csv('application_test.csv')
bureau = pl.read_csv('bureau.csv')
prev_app = pl.read_csv('previous_application.csv')
installments = pl.read_csv('installments_payments.csv')
credit_card = pl.read_csv('credit_card_balance.csv')
pos_cash = pl.read_csv('POS_CASH_balance.csv')

# Run complete pipeline
train_processed, test_processed = process_pipeline(
    app_train, app_test, bureau, prev_app, 
    installments, credit_card, pos_cash
)

# Save results
save_processed_data(train_processed, test_processed, output_dir="./output")
```

### Application Data Only

If supplementary data is unavailable:

```python
# Process application data without supplementary datasets
train_processed, test_processed = process_pipeline(app_train, app_test)
```

### Running Tests

```bash
python test_feature_engineering.py
```

## Detailed Function Reference

### Data Cleaning Functions

#### `fix_employment_anomaly(df)`
Fixes the DAYS_EMPLOYED = 365243 anomaly by:
- Creating `FLAG_UNEMPLOYED_UNKNOWN` binary indicator
- Setting anomalous values to null for proper imputation

**Input:** DataFrame with DAYS_EMPLOYED column  
**Output:** DataFrame with flag and cleaned employment days

#### `impute_ext_source_variables(df, config)`
Imputes missing values in EXT_SOURCE_1, EXT_SOURCE_2, EXT_SOURCE_3 using training medians.

**Input:** DataFrame and FeatureConfig object  
**Output:** DataFrame with imputed EXT_SOURCE values and missing flags

#### `impute_missing_values(df, config)`
Comprehensive missing value imputation:
- Numeric: median imputation
- Categorical: mode imputation
- Creates `FLAG_MISSING_*` indicators

**Input:** DataFrame and FeatureConfig object  
**Output:** DataFrame with all missing values imputed

#### `remove_high_missing_columns(df, threshold=0.5)`
Removes columns exceeding missing value threshold.

**Input:** DataFrame, threshold (default 50%)  
**Output:** DataFrame with high-missing columns removed

### Feature Engineering Functions

#### `engineer_demographic_features(df)`
Converts days-based features to interpretable years:
- `AGE_YEARS` - Age in years
- `EMPLOYMENT_YEARS` - Employment duration in years
- `REGISTRATION_YEARS` - Years since registration
- `ID_PUBLISH_YEARS` - Years since ID published

#### `engineer_financial_ratios(df)`
Creates 8 financial ratio features:
- `CREDIT_TO_INCOME_RATIO` - Loan size vs. income
- `ANNUITY_TO_INCOME_RATIO` - Debt-to-income (DTI)
- `LOAN_TO_VALUE_RATIO` - Credit vs. goods price
- `DOWN_PAYMENT_RATIO` - Upfront contribution percentage
- `INCOME_PER_CHILD` - Income adjusted for dependents
- `CREDIT_PER_PERSON` - Credit per family member
- `PAYMENT_TO_CREDIT_RATIO` - Total payments vs. credit
- `INCOME_TO_AGE_RATIO` - Earning capacity indicator

#### `engineer_interaction_features(df)`
Creates interaction and binned features:
- EXT_SOURCE products (1x2, 1x3, 2x3)
- EXT_SOURCE mean and sum
- Age groups (Young, Middle, Senior)
- Income groups (Low, Medium, High)
- High-risk occupation flag
- Document completeness score

#### `engineer_missing_indicators(df)`
Aggregates all missing indicators into:
- `TOTAL_MISSING_FEATURES` - Count of missing features per applicant

### Supplementary Data Aggregation

#### `aggregate_bureau_data(bureau)`
Aggregates bureau.csv to SK_ID_CURR level:
- Credit counts (total, active, closed)
- Debt amounts (total, overdue, max overdue)
- Time features (most recent, oldest credit)
- Ratios (debt-to-credit, overdue-to-credit, active ratio)

**Returns:** 15 features per applicant

#### `aggregate_previous_application_data(prev_app)`
Aggregates previous_application.csv:
- Application counts (total, approved, refused, canceled)
- Credit amounts (average, maximum)
- Approval/refusal rates
- Credit-to-application ratio

**Returns:** 14 features per applicant

#### `aggregate_installments_data(installments)`
Aggregates installments_payments.csv:
- Payment counts (total, late payments)
- Days past due statistics
- Payment amounts and trends
- Late payment rate

**Returns:** 8 features per applicant

#### `aggregate_credit_card_data(credit_card)`
Aggregates credit_card_balance.csv:
- Balance statistics (average, max, min)
- Credit limits
- Drawing and payment amounts
- Utilization and payment-to-balance ratios

**Returns:** 13 features per applicant

#### `aggregate_pos_cash_data(pos_cash)`
Aggregates POS_CASH_balance.csv:
- Contract counts (active, completed)
- Days past due (DPD) statistics
- Installment counts
- Completion rate

**Returns:** 9 features per applicant

### Pipeline Orchestration

#### `clean_application_data(df, config, is_train=True)`
Complete cleaning pipeline:
1. Remove high-missing columns
2. Fix employment anomaly
3. Fit config (if training data)
4. Impute EXT_SOURCE variables
5. Impute other missing values

#### `engineer_application_features(df)`
Complete feature engineering pipeline:
1. Demographic features
2. Financial ratios
3. Interaction features
4. Missing indicators

#### `join_supplementary_data(app_df, bureau, prev_app, ...)`
Joins all aggregated supplementary features to application data.

#### `process_pipeline(app_train, app_test, bureau, ...)`
**Master function** - Complete end-to-end pipeline:
1. Processes training data and fits FeatureConfig
2. Processes test data using same config
3. Verifies train/test consistency
4. Returns processed train and test DataFrames

### Utility Functions

#### `save_processed_data(train, test, output_dir)`
Saves processed data to CSV files.

#### `get_feature_summary(df)`
Returns descriptive statistics for all numeric features.

## FeatureConfig Class

The `FeatureConfig` class ensures train/test consistency by storing statistics from training data and reusing them on test data.

### Attributes
- `ext_source_medians` - Medians for EXT_SOURCE_1/2/3
- `numeric_medians` - Medians for all numeric features
- `categorical_modes` - Modes for all categorical features
- `is_fitted` - Boolean indicating if config has been fitted

### Methods

#### `fit(df, numeric_cols, categorical_cols)`
Computes and stores statistics from training data.

```python
config = FeatureConfig()
config.fit(train_df, numeric_cols=['AMT_INCOME_TOTAL', ...], 
           categorical_cols=['OCCUPATION_TYPE', ...])
```

## Feature Categories

### Application Features (~130)
- Original application columns
- Demographic features (age, employment years)
- Financial ratios
- Interaction terms
- Missing indicators

### Bureau Features (~15)
- Credit history with other lenders
- Active/closed credits
- Debt and overdue amounts
- Credit type diversity

### Previous Application Features (~14)
- Home Credit application history
- Approval/refusal rates
- Credit amounts requested vs. received

### Installment Features (~8)
- Payment history
- Late payment rates
- Days past due

### Credit Card Features (~13)
- Balance and limit statistics
- Utilization rates
- Payment patterns

### POS Cash Features (~9)
- Transaction counts
- Contract completion rates
- Days past due

**Total: ~190 features**

## Data Quality Checks

The pipeline includes automatic validation:
- ✅ No missing values in critical EXT_SOURCE features after processing
- ✅ Employment anomaly (365243 days) flagged and cleaned
- ✅ Identical columns in train and test (except TARGET)
- ✅ All supplementary features filled with 0 for applicants with no history
- ✅ No high-missing columns (>50%) in final dataset

## Best Practices Implemented

### 1. Train/Test Consistency
```python
# CORRECT: Fit on train, apply to test
config = FeatureConfig()
train_clean = clean_application_data(app_train, config, is_train=True)  # Fits config
test_clean = clean_application_data(app_test, config, is_train=False)   # Uses fitted config
```

### 2. Missing Value Strategy
- Median for numeric (robust to outliers)
- Mode for categorical
- Create missing indicators (missing values are informative)
- Never drop rows (preserves all 307,511 training samples)

### 3. Domain-Informed Engineering
- Convert days to years (interpretability)
- Create financial ratios used in lending (DTI, LTV)
- Flag high-risk occupations (based on EDA)
- Preserve employment anomaly as feature (not a data error)

### 4. Handling Supplementary Data
- Aggregate to applicant level (SK_ID_CURR)
- Fill nulls with 0 (indicates no credit history)
- Create both counts and ratios
- Include time-based features

## Common Issues and Solutions

### Issue: "Config not fitted" error
**Solution:** Always process training data before test data:
```python
train_processed, test_processed = process_pipeline(app_train, app_test, ...)
```

### Issue: Column mismatch between train and test
**Solution:** Use the same pipeline for both datasets. The `process_pipeline()` function handles this automatically.

### Issue: Memory errors with large datasets
**Solution:** Polars is memory-efficient, but for very large datasets:
```python
# Process in chunks or use lazy API
df = pl.scan_csv('large_file.csv').collect()
```

### Issue: Missing supplementary datasets
**Solution:** Pipeline works without them:
```python
# Application features only
train_processed, test_processed = process_pipeline(app_train, app_test)
```

## Performance

Typical processing times on a standard laptop:
- Application data only: ~30 seconds
- With all supplementary data: ~3-5 minutes

## Output Files

After running the pipeline:
- `application_train_processed.csv` - Processed training data (~190 columns)
- `application_test_processed.csv` - Processed test data (~189 columns, no TARGET)

## Next Steps for Modeling

1. **Load processed data:**
   ```python
   train = pl.read_csv('application_train_processed.csv')
   test = pl.read_csv('application_test_processed.csv')
   ```

2. **Prepare for modeling:**
   ```python
   # Separate features and target
   X_train = train.drop(['SK_ID_CURR', 'TARGET'])
   y_train = train.select('TARGET')
   X_test = test.drop('SK_ID_CURR')
   ```

3. **Handle categorical variables:**
   ```python
   # One-hot encode categorical columns
   categorical_cols = X_train.select(cs.string()).columns
   X_train = X_train.to_dummies(columns=categorical_cols)
   X_test = X_test.to_dummies(columns=categorical_cols)
   ```

4. **Build models:**
   - Logistic Regression (baseline)
   - Random Forest
   - LightGBM / XGBoost
   - Neural Networks

5. **Use appropriate metrics:**
   - ROC-AUC (class imbalance)
   - PR-AUC (minority class focus)
   - F1-score
   - Stratified K-fold CV

## Contributing

To extend the pipeline:
1. Add new functions following the same pattern
2. Update `process_pipeline()` to include new features
3. Test with `test_feature_engineering.py`
4. Document in this README

## Citation

If you use this pipeline, please cite:
```
Clahan Tran (2026). Home Credit Feature Engineering Pipeline. 
Exploratory Data Analysis for Default Risk Prediction.
```

## License

This code is provided for educational and research purposes.

---

**Questions?** Review the test script (`test_feature_engineering.py`) for detailed usage examples.
