# Feature Engineering Pipeline - Quick Start Guide

## 📁 Files Created

1. **`feature_engineering.py`** (850+ lines)
   - Main module with all feature engineering functions
   - Production-ready, reusable code
   - Handles train/test consistency automatically

2. **`test_feature_engineering.py`** (250+ lines)
   - Comprehensive test suite
   - Demonstrates all usage patterns
   - Validates functionality

3. **`FEATURE_ENGINEERING_README.md`** (Comprehensive documentation)
   - Detailed function reference
   - Usage examples
   - Best practices guide

## 🚀 Quick Usage

### Option 1: Run the complete pipeline

```python
import polars as pl
from feature_engineering import process_pipeline, save_processed_data

# Load all data
app_train = pl.read_csv('application_train.csv')
app_test = pl.read_csv('application_test.csv')
bureau = pl.read_csv('bureau.csv')
prev_app = pl.read_csv('previous_application.csv')
installments = pl.read_csv('installments_payments.csv')
credit_card = pl.read_csv('credit_card_balance.csv')
pos_cash = pl.read_csv('POS_CASH_balance.csv')

# Run pipeline (handles everything automatically)
train_processed, test_processed = process_pipeline(
    app_train, app_test, bureau, prev_app, 
    installments, credit_card, pos_cash
)

# Save results
save_processed_data(train_processed, test_processed, output_dir="./output")
```

### Option 2: Run the test script

```bash
cd c:/Users/tranc/Downloads/e.g-home-credit-project
python test_feature_engineering.py
```

This will:
- Load all datasets
- Run the complete pipeline
- Validate results
- Save processed files
- Display feature summary

## ✅ What the Pipeline Does

### 1. Data Cleaning ✓
- ✅ Fixes DAYS_EMPLOYED = 365243 anomaly (creates FLAG_UNEMPLOYED_UNKNOWN)
- ✅ Imputes missing values in EXT_SOURCE_1/2/3 using training medians
- ✅ Removes columns with >50% missing data
- ✅ Imputes all other missing values (median/mode)
- ✅ Creates missing value indicators (FLAG_MISSING_*)

### 2. Feature Engineering ✓
**Demographic Features (4 new):**
- AGE_YEARS - Age in years
- EMPLOYMENT_YEARS - Employment duration in years
- REGISTRATION_YEARS - Years since registration
- ID_PUBLISH_YEARS - Years since ID published

**Financial Ratios (7 new):**
- CREDIT_TO_INCOME_RATIO - Loan size vs. income
- ANNUITY_TO_INCOME_RATIO - Debt-to-income (DTI)
- LOAN_TO_VALUE_RATIO - Credit vs. goods price
- DOWN_PAYMENT_RATIO - Upfront contribution %
- INCOME_PER_CHILD - Income adjusted for dependents
- CREDIT_PER_PERSON - Credit per family member
- INCOME_TO_AGE_RATIO - Earning capacity

**Interaction Features (9 new):**
- EXT_SOURCE_1x2, EXT_SOURCE_1x3, EXT_SOURCE_2x3 - Products
- EXT_SOURCE_MEAN, EXT_SOURCE_SUM - Aggregates
- AGE_GROUP - Binned age (Young/Middle/Senior)
- INCOME_GROUP - Binned income (Low/Medium/High)
- FLAG_HIGH_RISK_OCCUPATION - Based on EDA findings
- DOCUMENT_COMPLETENESS_SCORE - Sum of document flags

### 3. Supplementary Data Aggregation ✓
**Bureau Features (15):**
- Credit counts (total, active, closed)
- Debt amounts (total, overdue)
- Debt ratios
- Time features

**Previous Application Features (14):**
- Application counts
- Approval/refusal rates
- Credit amounts

**Installment Features (8):**
- Payment counts
- Late payment rate
- Days past due stats

**Credit Card Features (13):**
- Balance statistics
- Utilization rates
- Payment patterns

**POS Cash Features (9):**
- Transaction counts
- Completion rates
- DPD statistics

### 4. Train/Test Consistency ✓
- ✅ FeatureConfig class stores training statistics
- ✅ Test data uses same medians/modes as training
- ✅ Identical columns in train and test (except TARGET)
- ✅ Automatic validation and reporting

## 📊 Output

**Final Dataset:**
- Training: 307,511 rows × ~190 columns
- Test: 48,744 rows × ~189 columns (no TARGET)

**Files Created:**
- `application_train_processed.csv`
- `application_test_processed.csv`

## 🔍 Key Features Based on EDA

✅ **Employment Anomaly Handled**
- 365243 days = unemployment/unknown status
- Preserved as FLAG_UNEMPLOYED_UNKNOWN
- Not treated as outlier or error

✅ **EXT_SOURCE Variables Prioritized**
- Strongest predictors (correlation -0.155 to -0.178)
- Missing values carefully imputed
- Interaction terms created

✅ **Missing Data Strategy**
- Only 1 column >50% missing (dropped)
- All other features imputed
- Missing indicators created
- All 307,511 training rows preserved

✅ **Financial Ratios**
- Domain-informed features
- Credit-to-income, DTI, LTV ratios
- Normalized by income/family size

✅ **Supplementary Data Value**
- 59 new features from 5 datasets
- Aggregated to applicant level
- Covers payment history and credit behavior

## ⚙️ Advanced Usage

### Use individual functions:

```python
from feature_engineering import (
    clean_application_data,
    engineer_application_features,
    aggregate_bureau_data,
    FeatureConfig
)

# Create config
config = FeatureConfig()

# Clean data
train_clean = clean_application_data(app_train, config, is_train=True)
test_clean = clean_application_data(app_test, config, is_train=False)

# Engineer features
train_eng = engineer_application_features(train_clean)
test_eng = engineer_application_features(test_clean)

# Aggregate bureau data
bureau_agg = aggregate_bureau_data(bureau)
train_final = train_eng.join(bureau_agg, on='SK_ID_CURR', how='left')
```

### Application data only (no supplementary):

```python
train_processed, test_processed = process_pipeline(app_train, app_test)
```

## 📝 Next Steps for Modeling

1. **Load processed data:**
   ```python
   train = pl.read_csv('application_train_processed.csv')
   test = pl.read_csv('application_test_processed.csv')
   ```

2. **Prepare features:**
   ```python
   X_train = train.drop(['SK_ID_CURR', 'TARGET'])
   y_train = train.select('TARGET')
   X_test = test.drop('SK_ID_CURR')
   ```

3. **Handle categorical variables:**
   ```python
   import polars.selectors as cs
   categorical_cols = X_train.select(cs.string()).columns
   X_train = X_train.to_dummies(columns=categorical_cols)
   X_test = X_test.to_dummies(columns=categorical_cols)
   ```

4. **Build models with appropriate metrics:**
   - ROC-AUC (class imbalance handling)
   - PR-AUC (minority class focus)
   - F1-score
   - Stratified K-fold CV

## 🐛 Troubleshooting

**Import errors:**
```python
# Make sure the module is in your path
import sys
sys.path.insert(0, 'c:/Users/tranc/Downloads/e.g-home-credit-project')
from feature_engineering import process_pipeline
```

**Memory issues:**
- Polars is memory-efficient, but for very large datasets use lazy evaluation
- Process supplementary data in chunks if needed

**Column mismatch errors:**
- Always use `process_pipeline()` for both train and test
- Don't process train and test separately

## 📚 Documentation

See **FEATURE_ENGINEERING_README.md** for:
- Detailed function reference
- Parameter descriptions
- Return value specifications
- Best practices
- Common issues and solutions

## ✨ Summary

This pipeline provides a **complete, production-ready solution** for feature engineering on the Home Credit dataset. It implements all EDA findings, ensures train/test consistency, and creates ~190 features ready for modeling.

**Total Lines of Code:** 1,100+  
**Functions:** 25+  
**Features Created:** ~70 new engineered features  
**Documentation:** Comprehensive

Ready to use! 🚀
