"""
Test Script for Feature Engineering Pipeline
=============================================

This script demonstrates how to use the feature engineering functions
and validates that they work correctly.

Author: Clahan Tran
Date: February 7, 2026
"""

import polars as pl
from feature_engineering import (
    process_pipeline,
    save_processed_data,
    get_feature_summary,
    FeatureConfig,
    clean_application_data,
    engineer_application_features,
)


def test_basic_pipeline():
    """
    Test the complete feature engineering pipeline with all datasets.
    """
    print("="*70)
    print("TEST: COMPLETE FEATURE ENGINEERING PIPELINE")
    print("="*70)
    
    # Load all datasets
    print("\n1. Loading datasets...")
    app_train = pl.read_csv('c:/Users/tranc/Desktop/Home Credit/application_train.csv')
    app_test = pl.read_csv('c:/Users/tranc/Desktop/Home Credit/application_test.csv')
    bureau = pl.read_csv('c:/Users/tranc/Desktop/Home Credit/bureau.csv')
    prev_app = pl.read_csv('c:/Users/tranc/Desktop/Home Credit/previous_application.csv')
    installments = pl.read_csv('c:/Users/tranc/Desktop/Home Credit/installments_payments.csv')
    credit_card = pl.read_csv('c:/Users/tranc/Desktop/Home Credit/credit_card_balance.csv')
    pos_cash = pl.read_csv('c:/Users/tranc/Desktop/Home Credit/POS_CASH_balance.csv')
    
    print(f"   ✓ Training data: {app_train.shape[0]:,} rows")
    print(f"   ✓ Test data: {app_test.shape[0]:,} rows")
    print(f"   ✓ Bureau: {bureau.shape[0]:,} rows")
    print(f"   ✓ Previous applications: {prev_app.shape[0]:,} rows")
    print(f"   ✓ Installments: {installments.shape[0]:,} rows")
    print(f"   ✓ Credit card: {credit_card.shape[0]:,} rows")
    print(f"   ✓ POS cash: {pos_cash.shape[0]:,} rows")
    
    # Run pipeline
    print("\n2. Running feature engineering pipeline...")
    train_processed, test_processed = process_pipeline(
        app_train, app_test, bureau, prev_app, installments, credit_card, pos_cash
    )
    
    # Validate results
    print("\n3. Validating results...")
    
    # Check for no missing values in critical features
    ext_source_missing = train_processed.select([
        pl.col('EXT_SOURCE_1').null_count(),
        pl.col('EXT_SOURCE_2').null_count(),
        pl.col('EXT_SOURCE_3').null_count(),
    ]).to_dicts()[0]
    
    print(f"   ✓ EXT_SOURCE missing values: {ext_source_missing}")
    
    # Check for new engineered features
    new_features = [
        'AGE_YEARS', 'EMPLOYMENT_YEARS', 'CREDIT_TO_INCOME_RATIO',
        'FLAG_UNEMPLOYED_UNKNOWN', 'EXT_SOURCE_MEAN', 'BUREAU_NUM_CREDITS'
    ]
    
    for feat in new_features:
        if feat in train_processed.columns:
            print(f"   ✓ Feature '{feat}' created successfully")
        else:
            print(f"   ✗ Feature '{feat}' NOT FOUND")
    
    # Check train/test consistency
    train_cols = set(train_processed.columns) - {'TARGET'}
    test_cols = set(test_processed.columns)
    
    if train_cols == test_cols:
        print(f"   ✓ Train/test consistency: {len(train_cols)} matching columns")
    else:
        print(f"   ✗ Train/test mismatch!")
        print(f"      Missing in test: {train_cols - test_cols}")
        print(f"      Extra in test: {test_cols - train_cols}")
    
    # Save processed data
    print("\n4. Saving processed data...")
    save_processed_data(
        train_processed, 
        test_processed, 
        output_dir="c:/Users/tranc/Downloads/e.g-home-credit-project"
    )
    
    print("\n✅ Pipeline test completed successfully!")
    
    return train_processed, test_processed


def test_application_only_pipeline():
    """
    Test feature engineering on application data only (no supplementary data).
    Useful for quick testing or when supplementary data is unavailable.
    """
    print("\n" + "="*70)
    print("TEST: APPLICATION DATA ONLY (NO SUPPLEMENTARY)")
    print("="*70)
    
    # Load application data
    print("\n1. Loading application data...")
    app_train = pl.read_csv('c:/Users/tranc/Desktop/Home Credit/application_train.csv')
    app_test = pl.read_csv('c:/Users/tranc/Desktop/Home Credit/application_test.csv')
    
    # Run pipeline without supplementary data
    print("\n2. Running pipeline (application features only)...")
    train_processed, test_processed = process_pipeline(
        app_train, app_test
    )
    
    print("\n✅ Application-only pipeline test completed!")
    
    return train_processed, test_processed


def test_feature_config_consistency():
    """
    Test that FeatureConfig ensures consistent train/test processing.
    """
    print("\n" + "="*70)
    print("TEST: FEATURE CONFIG CONSISTENCY")
    print("="*70)
    
    # Load data
    app_train = pl.read_csv('c:/Users/tranc/Desktop/Home Credit/application_train.csv')
    app_test = pl.read_csv('c:/Users/tranc/Desktop/Home Credit/application_test.csv')
    
    # Create config and fit on train
    config = FeatureConfig()
    
    print("\n1. Processing training data and fitting config...")
    train_clean = clean_application_data(app_train, config, is_train=True)
    
    # Check that config was fitted
    assert config.is_fitted, "Config should be fitted after training"
    print(f"   ✓ Config fitted with {len(config.ext_source_medians)} EXT_SOURCE medians")
    
    # Print stored medians
    for col, median in config.ext_source_medians.items():
        print(f"   ✓ {col} median: {median:.4f}")
    
    print("\n2. Processing test data with same config...")
    test_clean = clean_application_data(app_test, config, is_train=False)
    
    # Verify no missing values in EXT_SOURCE columns
    for i in [1, 2, 3]:
        col = f'EXT_SOURCE_{i}'
        train_missing = train_clean.select(pl.col(col).null_count()).item()
        test_missing = test_clean.select(pl.col(col).null_count()).item()
        
        print(f"   ✓ {col}: train missing = {train_missing}, test missing = {test_missing}")
        assert train_missing == 0, f"Training data should have no missing {col}"
        assert test_missing == 0, f"Test data should have no missing {col}"
    
    print("\n✅ Feature config consistency test passed!")


def display_feature_importance_preview(train: pl.DataFrame):
    """
    Display a preview of engineered features for inspection.
    """
    print("\n" + "="*70)
    print("ENGINEERED FEATURES PREVIEW")
    print("="*70)
    
    # Display key engineered features
    key_features = [
        'SK_ID_CURR', 'TARGET', 'AGE_YEARS', 'EMPLOYMENT_YEARS',
        'CREDIT_TO_INCOME_RATIO', 'ANNUITY_TO_INCOME_RATIO',
        'FLAG_UNEMPLOYED_UNKNOWN', 'EXT_SOURCE_MEAN',
        'BUREAU_NUM_CREDITS', 'PREV_APPROVAL_RATE',
        'INST_LATE_PAYMENT_RATE', 'CC_UTILIZATION_RATE'
    ]
    
    # Filter to columns that exist
    existing_features = [f for f in key_features if f in train.columns]
    
    print("\nSample of engineered features (first 10 rows):")
    print(train.select(existing_features).head(10))
    
    print("\n\nFeature statistics:")
    print(train.select(existing_features).describe())
    
    # Count features by category
    feature_counts = {
        'Application Features': len([c for c in train.columns if not any([
            c.startswith('BUREAU_'), c.startswith('PREV_'),
            c.startswith('INST_'), c.startswith('CC_'), c.startswith('POS_')
        ])]),
        'Bureau Features': len([c for c in train.columns if c.startswith('BUREAU_')]),
        'Previous App Features': len([c for c in train.columns if c.startswith('PREV_')]),
        'Installment Features': len([c for c in train.columns if c.startswith('INST_')]),
        'Credit Card Features': len([c for c in train.columns if c.startswith('CC_')]),
        'POS Cash Features': len([c for c in train.columns if c.startswith('POS_')]),
    }
    
    print("\n\nFeature count by category:")
    for category, count in feature_counts.items():
        print(f"  {category}: {count}")
    
    print(f"\nTotal features: {train.shape[1]}")


def run_all_tests():
    """
    Run all test functions.
    """
    print("\n" + "="*70)
    print("RUNNING ALL TESTS")
    print("="*70)
    
    # Test 1: Complete pipeline
    train, test = test_basic_pipeline()
    
    # Test 2: Feature config consistency
    test_feature_config_consistency()
    
    # Test 3: Application-only pipeline
    test_application_only_pipeline()
    
    # Display feature preview
    display_feature_importance_preview(train)
    
    print("\n" + "="*70)
    print("ALL TESTS COMPLETED SUCCESSFULLY! ✅")
    print("="*70)
    
    return train, test


if __name__ == "__main__":
    """
    Run all tests when script is executed.
    """
    train_processed, test_processed = run_all_tests()
    
    print("\n📊 Processed data is ready for modeling!")
    print(f"   Training: {train_processed.shape}")
    print(f"   Test: {test_processed.shape}")
    print(f"\n💾 Files saved to: c:/Users/tranc/Downloads/e.g-home-credit-project/")
    print("   - application_train_processed.csv")
    print("   - application_test_processed.csv")
