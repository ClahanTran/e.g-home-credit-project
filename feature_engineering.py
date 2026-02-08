"""
Home Credit Default Risk - Feature Engineering Pipeline
========================================================

This module provides reusable functions for cleaning, transforming, and engineering
features from the Home Credit dataset. All functions are designed to work on both
training and test data while maintaining consistency.

Author: Clahan Tran
Date: February 7, 2026
"""

import polars as pl
import polars.selectors as cs
import numpy as np
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# GLOBAL CONFIGURATION
# ============================================================================

class FeatureConfig:
    """
    Stores training statistics for consistent train/test transformation.
    This ensures test data is processed using the same parameters as training data.
    """
    def __init__(self):
        self.ext_source_medians: Dict[str, float] = {}
        self.numeric_medians: Dict[str, float] = {}
        self.categorical_modes: Dict[str, str] = {}
        self.is_fitted: bool = False
    
    def fit(self, df: pl.DataFrame, numeric_cols: List[str], categorical_cols: List[str]):
        """
        Compute and store statistics from training data.
        
        Parameters:
        -----------
        df : pl.DataFrame
            Training data
        numeric_cols : List[str]
            List of numeric columns to compute medians
        categorical_cols : List[str]
            List of categorical columns to compute modes
        """
        # Store medians for numeric columns
        for col in numeric_cols:
            if col in df.columns:
                median_val = df.select(pl.col(col).median()).item()
                self.numeric_medians[col] = median_val if median_val is not None else 0
        
        # Store modes for categorical columns
        for col in categorical_cols:
            if col in df.columns:
                mode_val = df.select(pl.col(col).mode().first()).item()
                self.categorical_modes[col] = mode_val if mode_val is not None else 'Missing'
        
        # Store EXT_SOURCE medians specifically
        for i in [1, 2, 3]:
            col_name = f'EXT_SOURCE_{i}'
            if col_name in df.columns:
                median_val = df.select(pl.col(col_name).median()).item()
                self.ext_source_medians[col_name] = median_val if median_val is not None else 0
        
        self.is_fitted = True
        print(f"✅ FeatureConfig fitted with {len(self.numeric_medians)} numeric and {len(self.categorical_modes)} categorical features")


# Initialize global config object
CONFIG = FeatureConfig()


# ============================================================================
# DATA CLEANING FUNCTIONS
# ============================================================================

def fix_employment_anomaly(df: pl.DataFrame) -> pl.DataFrame:
    """
    Fix the DAYS_EMPLOYED = 365243 anomaly identified in EDA.
    
    This value is a placeholder for unknown/missing employment. We create:
    1. A binary flag indicating this status
    2. Set the employment value to null for proper imputation
    
    Parameters:
    -----------
    df : pl.DataFrame
        Input dataframe with DAYS_EMPLOYED column
    
    Returns:
    --------
    pl.DataFrame
        Dataframe with FLAG_UNEMPLOYED_UNKNOWN and cleaned DAYS_EMPLOYED
    """
    df = df.with_columns([
        # Create flag for employment anomaly (>15000 days = ~41 years)
        (pl.col('DAYS_EMPLOYED') > 15000).cast(pl.Int32).alias('FLAG_UNEMPLOYED_UNKNOWN'),
        
        # Replace anomalous values with null for proper imputation
        pl.when(pl.col('DAYS_EMPLOYED') > 15000)
        .then(None)
        .otherwise(pl.col('DAYS_EMPLOYED'))
        .alias('DAYS_EMPLOYED')
    ])
    
    anomaly_count = df.select(pl.col('FLAG_UNEMPLOYED_UNKNOWN').sum()).item()
    print(f"  ✓ Employment anomaly flagged: {anomaly_count:,} cases ({anomaly_count/df.height*100:.1f}%)")
    
    return df


def impute_ext_source_variables(df: pl.DataFrame, config: FeatureConfig) -> pl.DataFrame:
    """
    Impute missing values in EXT_SOURCE_1, EXT_SOURCE_2, EXT_SOURCE_3.
    These are the strongest predictors identified in EDA.
    
    Strategy: Use median from training data + create missing indicators
    
    Parameters:
    -----------
    df : pl.DataFrame
        Input dataframe
    config : FeatureConfig
        Configuration object with training medians
    
    Returns:
    --------
    pl.DataFrame
        Dataframe with imputed EXT_SOURCE values and missing flags
    """
    for i in [1, 2, 3]:
        col_name = f'EXT_SOURCE_{i}'
        flag_name = f'FLAG_MISSING_{col_name}'
        
        if col_name in df.columns:
            # Get median from config
            median_val = config.ext_source_medians.get(col_name, 0)
            
            df = df.with_columns([
                # Create missing indicator
                pl.col(col_name).is_null().cast(pl.Int32).alias(flag_name),
                
                # Impute missing values with median
                pl.col(col_name).fill_null(median_val).alias(col_name)
            ])
            
            missing_count = df.select(pl.col(flag_name).sum()).item()
            print(f"  ✓ {col_name}: {missing_count:,} missing values imputed with median = {median_val:.4f}")
    
    return df


def impute_missing_values(df: pl.DataFrame, config: FeatureConfig) -> pl.DataFrame:
    """
    Comprehensive missing value imputation for all features.
    
    Strategy:
    - Numeric: median imputation + missing indicator
    - Categorical: mode imputation + missing indicator
    - Skip columns with >50% missing (should be dropped upstream)
    
    Parameters:
    -----------
    df : pl.DataFrame
        Input dataframe
    config : FeatureConfig
        Configuration object with training statistics
    
    Returns:
    --------
    pl.DataFrame
        Dataframe with imputed values and missing indicators
    """
    # Impute numeric columns
    numeric_cols = df.select(cs.numeric()).columns
    numeric_cols = [col for col in numeric_cols if col not in ['SK_ID_CURR', 'TARGET']]
    
    for col in numeric_cols:
        missing_pct = df.select(pl.col(col).null_count() / df.height * 100).item()
        
        # Skip if already handled or too much missing
        if col.startswith('EXT_SOURCE_') or col.startswith('FLAG_') or missing_pct > 50:
            continue
        
        if missing_pct > 0:
            median_val = config.numeric_medians.get(col, 0)
            flag_name = f'FLAG_MISSING_{col}'
            
            df = df.with_columns([
                pl.col(col).is_null().cast(pl.Int32).alias(flag_name),
                pl.col(col).fill_null(median_val).alias(col)
            ])
    
    # Impute categorical columns
    categorical_cols = df.select(cs.string()).columns
    
    for col in categorical_cols:
        missing_pct = df.select(pl.col(col).null_count() / df.height * 100).item()
        
        if missing_pct > 50:
            continue
        
        if missing_pct > 0:
            mode_val = config.categorical_modes.get(col, 'Missing')
            flag_name = f'FLAG_MISSING_{col}'
            
            df = df.with_columns([
                pl.col(col).is_null().cast(pl.Int32).alias(flag_name),
                pl.col(col).fill_null(mode_val).alias(col)
            ])
    
    print(f"  ✓ Missing value imputation complete")
    
    return df


def remove_high_missing_columns(df: pl.DataFrame, threshold: float = 0.5) -> pl.DataFrame:
    """
    Remove columns with missing values exceeding threshold.
    Based on EDA findings: only 1 column had >50% missing.
    
    Parameters:
    -----------
    df : pl.DataFrame
        Input dataframe
    threshold : float
        Proportion of missing values (0.5 = 50%)
    
    Returns:
    --------
    pl.DataFrame
        Dataframe with high-missing columns removed
    """
    cols_to_drop = []
    
    for col in df.columns:
        missing_pct = df.select(pl.col(col).null_count() / df.height).item()
        if missing_pct > threshold:
            cols_to_drop.append(col)
    
    if cols_to_drop:
        df = df.drop(cols_to_drop)
        print(f"  ✓ Dropped {len(cols_to_drop)} columns with >{threshold*100}% missing: {cols_to_drop}")
    else:
        print(f"  ✓ No columns exceed {threshold*100}% missing threshold")
    
    return df


# ============================================================================
# FEATURE ENGINEERING FUNCTIONS
# ============================================================================

def engineer_demographic_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Convert demographic features from days to interpretable years.
    
    Creates:
    - AGE_YEARS: Age in years (from DAYS_BIRTH)
    - EMPLOYMENT_YEARS: Employment duration in years (from DAYS_EMPLOYED)
    - REGISTRATION_YEARS: Years since registration (from DAYS_REGISTRATION)
    - ID_PUBLISH_YEARS: Years since ID published (from DAYS_ID_PUBLISH)
    
    Parameters:
    -----------
    df : pl.DataFrame
        Input dataframe
    
    Returns:
    --------
    pl.DataFrame
        Dataframe with new demographic features
    """
    df = df.with_columns([
        # Age in years (DAYS_BIRTH is negative)
        (pl.col('DAYS_BIRTH') / -365.25).alias('AGE_YEARS'),
        
        # Employment years (DAYS_EMPLOYED is negative, already cleaned for anomaly)
        (pl.col('DAYS_EMPLOYED') / -365.25).alias('EMPLOYMENT_YEARS'),
        
        # Registration years
        (pl.col('DAYS_REGISTRATION') / -365.25).alias('REGISTRATION_YEARS'),
        
        # ID publish years
        (pl.col('DAYS_ID_PUBLISH') / -365.25).alias('ID_PUBLISH_YEARS'),
    ])
    
    print(f"  ✓ Demographic features engineered: AGE_YEARS, EMPLOYMENT_YEARS, etc.")
    
    return df


def engineer_financial_ratios(df: pl.DataFrame) -> pl.DataFrame:
    """
    Create financial ratio features based on domain knowledge and research.
    
    Features created:
    - CREDIT_TO_INCOME_RATIO: Loan size relative to annual income
    - ANNUITY_TO_INCOME_RATIO: Payment burden (similar to DTI)
    - LOAN_TO_VALUE_RATIO: Credit amount vs. goods price
    - DOWN_PAYMENT_RATIO: Upfront payment as fraction of price
    - INCOME_PER_CHILD: Income adjusted for dependents
    - CREDIT_PER_PERSON: Credit amount per family member
    - PAYMENT_TO_CREDIT_RATIO: Total payments vs. credit (if CNT_PAYMENT available)
    - INCOME_TO_AGE_RATIO: Earning capacity indicator
    
    Args:
        df: Input dataframe with financial columns
        
    Returns:
        Dataframe with financial ratio features
    """
    # Base financial ratios (always available)
    df = df.with_columns([
        # Credit-to-Income Ratio (loan size relative to income)
        (pl.col('AMT_CREDIT') / pl.col('AMT_INCOME_TOTAL')).alias('CREDIT_TO_INCOME_RATIO'),
        
        # Annuity-to-Income Ratio (payment burden, similar to DTI)
        (pl.col('AMT_ANNUITY') / pl.col('AMT_INCOME_TOTAL')).alias('ANNUITY_TO_INCOME_RATIO'),
        
        # Loan-to-Value Ratio (credit amount vs. goods price)
        (pl.col('AMT_CREDIT') / pl.col('AMT_GOODS_PRICE')).alias('LOAN_TO_VALUE_RATIO'),
        
        # Down Payment Ratio (how much upfront vs. total price)
        ((pl.col('AMT_GOODS_PRICE') - pl.col('AMT_CREDIT')) / pl.col('AMT_GOODS_PRICE')).alias('DOWN_PAYMENT_RATIO'),
        
        # Income per child (income adjusted for dependents)
        (pl.col('AMT_INCOME_TOTAL') / (pl.col('CNT_CHILDREN') + 1)).alias('INCOME_PER_CHILD'),
        
        # Credit per person (credit amount per family member)
        (pl.col('AMT_CREDIT') / (pl.col('CNT_FAM_MEMBERS') + 1)).alias('CREDIT_PER_PERSON'),
        
        # Income-to-Age Ratio (earning capacity indicator)
        (pl.col('AMT_INCOME_TOTAL') / (pl.col('DAYS_BIRTH') / -365.25)).alias('INCOME_TO_AGE_RATIO'),
    ])
    
    feature_count = 7
    
    # Payment-to-Credit Ratio (only if CNT_PAYMENT exists - from joined data)
    if 'CNT_PAYMENT' in df.columns:
        df = df.with_columns(
            ((pl.col('AMT_ANNUITY') * pl.col('CNT_PAYMENT')) / pl.col('AMT_CREDIT')).alias('PAYMENT_TO_CREDIT_RATIO')
        )
        feature_count += 1

    print(f"  ✓ Financial ratios engineered: {feature_count} new features")

    return df


def engineer_interaction_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Create interaction and binned features to capture non-linear relationships.
    
    Based on EDA findings:
    - EXT_SOURCE interactions (strongest predictors)
    - Age bins (risk varies by age group)
    - Income bins (risk varies by income level)
    - Occupation risk indicators
    
    Parameters:
    -----------
    df : pl.DataFrame
        Input dataframe
    
    Returns:
    --------
    pl.DataFrame
        Dataframe with interaction features
    """
    df = df.with_columns([
        # EXT_SOURCE interactions (multiply strongest predictors)
        (pl.col('EXT_SOURCE_1') * pl.col('EXT_SOURCE_2')).alias('EXT_SOURCE_1x2'),
        (pl.col('EXT_SOURCE_1') * pl.col('EXT_SOURCE_3')).alias('EXT_SOURCE_1x3'),
        (pl.col('EXT_SOURCE_2') * pl.col('EXT_SOURCE_3')).alias('EXT_SOURCE_2x3'),
        
        # Mean and sum of EXT_SOURCE scores
        ((pl.col('EXT_SOURCE_1') + pl.col('EXT_SOURCE_2') + pl.col('EXT_SOURCE_3')) / 3).alias('EXT_SOURCE_MEAN'),
        (pl.col('EXT_SOURCE_1') + pl.col('EXT_SOURCE_2') + pl.col('EXT_SOURCE_3')).alias('EXT_SOURCE_SUM'),
        
        # Age bins (young, middle, senior)
        pl.when(pl.col('DAYS_BIRTH') / -365.25 < 30).then(pl.lit('Young'))
        .when(pl.col('DAYS_BIRTH') / -365.25 < 50).then(pl.lit('Middle'))
        .otherwise(pl.lit('Senior'))
        .alias('AGE_GROUP'),
        
        # Income bins (low, medium, high)
        pl.when(pl.col('AMT_INCOME_TOTAL') < 100000).then(pl.lit('Low'))
        .when(pl.col('AMT_INCOME_TOTAL') < 200000).then(pl.lit('Medium'))
        .otherwise(pl.lit('High'))
        .alias('INCOME_GROUP'),
        
        # High-risk occupation flag (based on EDA: Drivers, Armed Forces)
        pl.when(pl.col('OCCUPATION_TYPE').is_in(['Drivers', 'Laborers', 'Low-skill Laborers', 'Waiters/barmen staff']))
        .then(1)
        .otherwise(0)
        .alias('FLAG_HIGH_RISK_OCCUPATION'),
        
        # Document completeness score (sum of all document flags)
        (pl.col('FLAG_DOCUMENT_2') + pl.col('FLAG_DOCUMENT_3') + pl.col('FLAG_DOCUMENT_4') +
         pl.col('FLAG_DOCUMENT_5') + pl.col('FLAG_DOCUMENT_6') + pl.col('FLAG_DOCUMENT_7') +
         pl.col('FLAG_DOCUMENT_8') + pl.col('FLAG_DOCUMENT_9')).alias('DOCUMENT_COMPLETENESS_SCORE'),
    ])
    
    print(f"  ✓ Interaction features engineered: EXT_SOURCE products, age/income bins, occupation flags")
    
    return df


def engineer_missing_indicators(df: pl.DataFrame) -> pl.DataFrame:
    """
    Create binary missing indicators for key features.
    Missing values themselves can be predictive of default.
    
    Parameters:
    -----------
    df : pl.DataFrame
        Input dataframe
    
    Returns:
    --------
    pl.DataFrame
        Dataframe with missing indicator flags
    """
    # This is already handled in impute_missing_values()
    # But we can create aggregate missing indicators
    
    # Count total missing features per applicant
    missing_cols = [col for col in df.columns if col.startswith('FLAG_MISSING_')]
    
    if missing_cols:
        df = df.with_columns([
            pl.sum_horizontal(missing_cols).alias('TOTAL_MISSING_FEATURES')
        ])
        print(f"  ✓ Aggregate missing indicator created: TOTAL_MISSING_FEATURES")
    
    return df


# ============================================================================
# SUPPLEMENTARY DATA AGGREGATION FUNCTIONS
# ============================================================================

def aggregate_bureau_data(bureau: pl.DataFrame) -> pl.DataFrame:
    """
    Aggregate bureau.csv to applicant level (SK_ID_CURR).
    
    Features created:
    - Count of prior credits
    - Number of active vs. closed credits
    - Total debt and overdue amounts
    - Average days since last credit
    - Debt ratios
    
    Parameters:
    -----------
    bureau : pl.DataFrame
        Bureau credit history data
    
    Returns:
    --------
    pl.DataFrame
        Aggregated bureau features at SK_ID_CURR level
    """
    bureau_agg = bureau.group_by('SK_ID_CURR').agg([
        # Count features
        pl.len().alias('BUREAU_NUM_CREDITS'),
        pl.col('CREDIT_ACTIVE').filter(pl.col('CREDIT_ACTIVE') == 'Active').len().alias('BUREAU_NUM_ACTIVE'),
        pl.col('CREDIT_ACTIVE').filter(pl.col('CREDIT_ACTIVE') == 'Closed').len().alias('BUREAU_NUM_CLOSED'),
        
        # Debt and credit amounts
        pl.col('AMT_CREDIT_SUM').sum().alias('BUREAU_TOTAL_CREDIT'),
        pl.col('AMT_CREDIT_SUM_DEBT').sum().alias('BUREAU_TOTAL_DEBT'),
        pl.col('AMT_CREDIT_SUM_OVERDUE').sum().alias('BUREAU_TOTAL_OVERDUE'),
        pl.col('AMT_CREDIT_MAX_OVERDUE').max().alias('BUREAU_MAX_OVERDUE'),
        
        # Time features
        pl.col('DAYS_CREDIT').max().alias('BUREAU_DAYS_MOST_RECENT_CREDIT'),
        pl.col('DAYS_CREDIT').min().alias('BUREAU_DAYS_OLDEST_CREDIT'),
        pl.col('DAYS_CREDIT').mean().alias('BUREAU_AVG_DAYS_CREDIT'),
        
        # Credit types diversity
        pl.col('CREDIT_TYPE').n_unique().alias('BUREAU_NUM_CREDIT_TYPES'),
    ])
    
    # Calculate debt ratios
    bureau_agg = bureau_agg.with_columns([
        # Debt-to-Credit Ratio
        (pl.col('BUREAU_TOTAL_DEBT') / pl.col('BUREAU_TOTAL_CREDIT')).alias('BUREAU_DEBT_TO_CREDIT_RATIO'),
        
        # Overdue-to-Credit Ratio
        (pl.col('BUREAU_TOTAL_OVERDUE') / pl.col('BUREAU_TOTAL_CREDIT')).alias('BUREAU_OVERDUE_TO_CREDIT_RATIO'),
        
        # Active-to-Total Credits Ratio
        (pl.col('BUREAU_NUM_ACTIVE') / pl.col('BUREAU_NUM_CREDITS')).alias('BUREAU_ACTIVE_RATIO'),
    ])
    
    print(f"  ✓ Bureau data aggregated: {bureau_agg.shape[1]} features for {bureau_agg.shape[0]:,} applicants")
    
    return bureau_agg


def aggregate_previous_application_data(prev_app: pl.DataFrame) -> pl.DataFrame:
    """
    Aggregate previous_application.csv to applicant level.
    
    Features created:
    - Count of previous applications
    - Approval/refusal/cancellation rates
    - Average credit amounts requested vs. received
    - Application trends
    
    Parameters:
    -----------
    prev_app : pl.DataFrame
        Previous application history
    
    Returns:
    --------
    pl.DataFrame
        Aggregated previous application features
    """
    prev_agg = prev_app.group_by('SK_ID_CURR').agg([
        # Count features
        pl.len().alias('PREV_NUM_APPLICATIONS'),
        pl.col('NAME_CONTRACT_STATUS').filter(pl.col('NAME_CONTRACT_STATUS') == 'Approved').len().alias('PREV_NUM_APPROVED'),
        pl.col('NAME_CONTRACT_STATUS').filter(pl.col('NAME_CONTRACT_STATUS') == 'Refused').len().alias('PREV_NUM_REFUSED'),
        pl.col('NAME_CONTRACT_STATUS').filter(pl.col('NAME_CONTRACT_STATUS') == 'Canceled').len().alias('PREV_NUM_CANCELED'),
        
        # Credit amounts
        pl.col('AMT_APPLICATION').mean().alias('PREV_AVG_APPLICATION_AMT'),
        pl.col('AMT_APPLICATION').max().alias('PREV_MAX_APPLICATION_AMT'),
        pl.col('AMT_CREDIT').mean().alias('PREV_AVG_CREDIT_AMT'),
        pl.col('AMT_CREDIT').max().alias('PREV_MAX_CREDIT_AMT'),
        
        # Down payment features
        pl.col('AMT_DOWN_PAYMENT').mean().alias('PREV_AVG_DOWN_PAYMENT'),
        pl.col('AMT_DOWN_PAYMENT').sum().alias('PREV_TOTAL_DOWN_PAYMENT'),
        
        # Time features
        pl.col('DAYS_DECISION').max().alias('PREV_DAYS_MOST_RECENT'),
        pl.col('DAYS_DECISION').mean().alias('PREV_AVG_DAYS_DECISION'),
    ])
    
    # Calculate approval metrics
    prev_agg = prev_agg.with_columns([
        # Approval rate
        (pl.col('PREV_NUM_APPROVED') / pl.col('PREV_NUM_APPLICATIONS')).alias('PREV_APPROVAL_RATE'),
        
        # Refusal rate
        (pl.col('PREV_NUM_REFUSED') / pl.col('PREV_NUM_APPLICATIONS')).alias('PREV_REFUSAL_RATE'),
        
        # Credit-to-Application ratio (how much requested vs. received)
        (pl.col('PREV_AVG_CREDIT_AMT') / pl.col('PREV_AVG_APPLICATION_AMT')).alias('PREV_CREDIT_TO_APPLICATION_RATIO'),
    ])
    
    print(f"  ✓ Previous application data aggregated: {prev_agg.shape[1]} features for {prev_agg.shape[0]:,} applicants")
    
    return prev_agg


def aggregate_installments_data(installments: pl.DataFrame) -> pl.DataFrame:
    """
    Aggregate installments_payments.csv to applicant level.
    
    Features created:
    - Count of installment payments
    - Late payment percentages
    - Payment amount trends
    - Days past due statistics
    
    Parameters:
    -----------
    installments : pl.DataFrame
        Installment payment history
    
    Returns:
    --------
    pl.DataFrame
        Aggregated installment features
    """
    # Calculate payment lateness
    installments = installments.with_columns([
        # Days past due (positive = late, negative = early/on-time)
        (pl.col('DAYS_ENTRY_PAYMENT') - pl.col('DAYS_INSTALMENT')).alias('DAYS_PAST_DUE'),
        
        # Payment amount difference
        (pl.col('AMT_PAYMENT') - pl.col('AMT_INSTALMENT')).alias('PAYMENT_DIFF'),
    ])
    
    inst_agg = installments.group_by('SK_ID_CURR').agg([
        # Count features
        pl.len().alias('INST_NUM_PAYMENTS'),
        pl.col('DAYS_PAST_DUE').filter(pl.col('DAYS_PAST_DUE') > 0).len().alias('INST_NUM_LATE_PAYMENTS'),
        
        # Late payment statistics
        pl.col('DAYS_PAST_DUE').max().alias('INST_MAX_DAYS_PAST_DUE'),
        pl.col('DAYS_PAST_DUE').mean().alias('INST_AVG_DAYS_PAST_DUE'),
        
        # Payment amount statistics
        pl.col('AMT_PAYMENT').mean().alias('INST_AVG_PAYMENT_AMT'),
        pl.col('AMT_PAYMENT').sum().alias('INST_TOTAL_PAYMENT_AMT'),
        pl.col('PAYMENT_DIFF').mean().alias('INST_AVG_PAYMENT_DIFF'),
    ])
    
    # Calculate late payment percentage
    inst_agg = inst_agg.with_columns([
        (pl.col('INST_NUM_LATE_PAYMENTS') / pl.col('INST_NUM_PAYMENTS')).alias('INST_LATE_PAYMENT_RATE'),
    ])
    
    print(f"  ✓ Installments data aggregated: {inst_agg.shape[1]} features for {inst_agg.shape[0]:,} applicants")
    
    return inst_agg


def aggregate_credit_card_data(credit_card: pl.DataFrame) -> pl.DataFrame:
    """
    Aggregate credit_card_balance.csv to applicant level.
    
    Features created:
    - Average credit card balances
    - Credit limit utilization
    - Minimum payment patterns
    - Drawing trends
    
    Parameters:
    -----------
    credit_card : pl.DataFrame
        Credit card balance history
    
    Returns:
    --------
    pl.DataFrame
        Aggregated credit card features
    """
    cc_agg = credit_card.group_by('SK_ID_CURR').agg([
        # Count features
        pl.len().alias('CC_NUM_RECORDS'),
        
        # Balance features
        pl.col('AMT_BALANCE').mean().alias('CC_AVG_BALANCE'),
        pl.col('AMT_BALANCE').max().alias('CC_MAX_BALANCE'),
        pl.col('AMT_BALANCE').min().alias('CC_MIN_BALANCE'),
        
        # Credit limit features
        pl.col('AMT_CREDIT_LIMIT_ACTUAL').mean().alias('CC_AVG_CREDIT_LIMIT'),
        pl.col('AMT_CREDIT_LIMIT_ACTUAL').max().alias('CC_MAX_CREDIT_LIMIT'),
        
        # Drawing features
        pl.col('AMT_DRAWINGS_CURRENT').mean().alias('CC_AVG_DRAWINGS'),
        pl.col('AMT_DRAWINGS_CURRENT').sum().alias('CC_TOTAL_DRAWINGS'),
        
        # Payment features
        pl.col('AMT_PAYMENT_CURRENT').mean().alias('CC_AVG_PAYMENT'),
        pl.col('AMT_PAYMENT_TOTAL_CURRENT').mean().alias('CC_AVG_TOTAL_PAYMENT'),
        
        # Minimum payment features
        pl.col('AMT_INST_MIN_REGULARITY').mean().alias('CC_AVG_MIN_PAYMENT'),
    ])
    
    # Calculate utilization ratios
    cc_agg = cc_agg.with_columns([
        # Credit utilization rate
        (pl.col('CC_AVG_BALANCE') / pl.col('CC_AVG_CREDIT_LIMIT')).alias('CC_UTILIZATION_RATE'),
        
        # Payment-to-balance ratio
        (pl.col('CC_AVG_PAYMENT') / pl.col('CC_AVG_BALANCE')).alias('CC_PAYMENT_TO_BALANCE_RATIO'),
    ])
    
    print(f"  ✓ Credit card data aggregated: {cc_agg.shape[1]} features for {cc_agg.shape[0]:,} applicants")
    
    return cc_agg


def aggregate_pos_cash_data(pos_cash: pl.DataFrame) -> pl.DataFrame:
    """
    Aggregate POS_CASH_balance.csv to applicant level.
    
    Features created:
    - Count of POS/cash transactions
    - Active vs. completed contracts
    - DPD (Days Past Due) statistics
    
    Parameters:
    -----------
    pos_cash : pl.DataFrame
        POS cash balance history
    
    Returns:
    --------
    pl.DataFrame
        Aggregated POS cash features
    """
    pos_agg = pos_cash.group_by('SK_ID_CURR').agg([
        # Count features
        pl.len().alias('POS_NUM_RECORDS'),
        pl.col('NAME_CONTRACT_STATUS').filter(pl.col('NAME_CONTRACT_STATUS') == 'Active').len().alias('POS_NUM_ACTIVE'),
        pl.col('NAME_CONTRACT_STATUS').filter(pl.col('NAME_CONTRACT_STATUS') == 'Completed').len().alias('POS_NUM_COMPLETED'),
        
        # DPD (Days Past Due) features
        pl.col('SK_DPD').max().alias('POS_MAX_DPD'),
        pl.col('SK_DPD').mean().alias('POS_AVG_DPD'),
        pl.col('SK_DPD_DEF').max().alias('POS_MAX_DPD_DEF'),
        
        # Balance features
        pl.col('CNT_INSTALMENT').mean().alias('POS_AVG_INSTALMENTS'),
        pl.col('CNT_INSTALMENT_FUTURE').mean().alias('POS_AVG_FUTURE_INSTALMENTS'),
    ])
    
    # Calculate completion rate
    pos_agg = pos_agg.with_columns([
        (pl.col('POS_NUM_COMPLETED') / (pl.col('POS_NUM_ACTIVE') + pl.col('POS_NUM_COMPLETED'))).alias('POS_COMPLETION_RATE'),
    ])
    
    print(f"  ✓ POS cash data aggregated: {pos_agg.shape[1]} features for {pos_agg.shape[0]:,} applicants")
    
    return pos_agg


# ============================================================================
# PIPELINE ORCHESTRATION FUNCTIONS
# ============================================================================

def clean_application_data(df: pl.DataFrame, config: FeatureConfig, is_train: bool = True) -> pl.DataFrame:
    """
    Complete data cleaning pipeline for application data.
    
    Steps:
    1. Remove high-missing columns
    2. Fix employment anomaly
    3. Impute EXT_SOURCE variables
    4. Impute other missing values
    
    Parameters:
    -----------
    df : pl.DataFrame
        Raw application data
    config : FeatureConfig
        Configuration object (fitted on train, reused on test)
    is_train : bool
        Whether this is training data (for fitting config)
    
    Returns:
    --------
    pl.DataFrame
        Cleaned application data
    """
    print("\n" + "="*70)
    print("STEP 1: DATA CLEANING")
    print("="*70)
    
    # Remove high-missing columns
    df = remove_high_missing_columns(df, threshold=0.5)
    
    # Fix employment anomaly
    df = fix_employment_anomaly(df)
    
    # Fit config on training data
    if is_train and not config.is_fitted:
        numeric_cols = df.select(cs.numeric()).columns
        numeric_cols = [col for col in numeric_cols if col not in ['SK_ID_CURR', 'TARGET']]
        categorical_cols = df.select(cs.string()).columns
        config.fit(df, numeric_cols, categorical_cols)
    
    # Impute EXT_SOURCE variables
    df = impute_ext_source_variables(df, config)
    
    # Impute other missing values
    df = impute_missing_values(df, config)
    
    return df


def engineer_application_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Complete feature engineering pipeline for application data.
    
    Steps:
    1. Engineer demographic features
    2. Engineer financial ratios
    3. Engineer interaction features
    4. Engineer missing indicators
    
    Parameters:
    -----------
    df : pl.DataFrame
        Cleaned application data
    
    Returns:
    --------
    pl.DataFrame
        Application data with engineered features
    """
    print("\n" + "="*70)
    print("STEP 2: FEATURE ENGINEERING")
    print("="*70)
    
    df = engineer_demographic_features(df)
    df = engineer_financial_ratios(df)
    df = engineer_interaction_features(df)
    df = engineer_missing_indicators(df)
    
    return df


def join_supplementary_data(
    app_df: pl.DataFrame,
    bureau: Optional[pl.DataFrame] = None,
    prev_app: Optional[pl.DataFrame] = None,
    installments: Optional[pl.DataFrame] = None,
    credit_card: Optional[pl.DataFrame] = None,
    pos_cash: Optional[pl.DataFrame] = None
) -> pl.DataFrame:
    """
    Join aggregated supplementary data to application data.
    
    Parameters:
    -----------
    app_df : pl.DataFrame
        Application data with engineered features
    bureau, prev_app, installments, credit_card, pos_cash : Optional[pl.DataFrame]
        Supplementary datasets (will be aggregated if provided)
    
    Returns:
    --------
    pl.DataFrame
        Application data with supplementary features joined
    """
    print("\n" + "="*70)
    print("STEP 3: SUPPLEMENTARY DATA INTEGRATION")
    print("="*70)
    
    # Aggregate and join bureau data
    if bureau is not None:
        bureau_agg = aggregate_bureau_data(bureau)
        app_df = app_df.join(bureau_agg, on='SK_ID_CURR', how='left')
    
    # Aggregate and join previous application data
    if prev_app is not None:
        prev_agg = aggregate_previous_application_data(prev_app)
        app_df = app_df.join(prev_agg, on='SK_ID_CURR', how='left')
    
    # Aggregate and join installments data
    if installments is not None:
        inst_agg = aggregate_installments_data(installments)
        app_df = app_df.join(inst_agg, on='SK_ID_CURR', how='left')
    
    # Aggregate and join credit card data
    if credit_card is not None:
        cc_agg = aggregate_credit_card_data(credit_card)
        app_df = app_df.join(cc_agg, on='SK_ID_CURR', how='left')
    
    # Aggregate and join POS cash data
    if pos_cash is not None:
        pos_agg = aggregate_pos_cash_data(pos_cash)
        app_df = app_df.join(pos_agg, on='SK_ID_CURR', how='left')
    
    # Fill nulls in supplementary features with 0 (no history)
    supp_cols = [col for col in app_df.columns if any([
        col.startswith('BUREAU_'),
        col.startswith('PREV_'),
        col.startswith('INST_'),
        col.startswith('CC_'),
        col.startswith('POS_')
    ])]
    
    for col in supp_cols:
        app_df = app_df.with_columns([
            pl.col(col).fill_null(0).alias(col)
        ])
    
    print(f"\n  ✓ Final dataset: {app_df.shape[0]:,} rows × {app_df.shape[1]} columns")
    
    return app_df


def process_pipeline(
    app_train: pl.DataFrame,
    app_test: pl.DataFrame,
    bureau: Optional[pl.DataFrame] = None,
    prev_app: Optional[pl.DataFrame] = None,
    installments: Optional[pl.DataFrame] = None,
    credit_card: Optional[pl.DataFrame] = None,
    pos_cash: Optional[pl.DataFrame] = None
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """
    Complete end-to-end processing pipeline for train and test data.
    Ensures train/test consistency by fitting on train and applying to test.
    
    Parameters:
    -----------
    app_train : pl.DataFrame
        Training application data
    app_test : pl.DataFrame
        Test application data
    bureau, prev_app, installments, credit_card, pos_cash : Optional[pl.DataFrame]
        Supplementary datasets
    
    Returns:
    --------
    Tuple[pl.DataFrame, pl.DataFrame]
        Processed train and test dataframes
    """
    global CONFIG
    CONFIG = FeatureConfig()  # Reset config
    
    print("\n" + "="*70)
    print("HOME CREDIT FEATURE ENGINEERING PIPELINE")
    print("="*70)
    print(f"Training data: {app_train.shape[0]:,} rows × {app_train.shape[1]} columns")
    print(f"Test data: {app_test.shape[0]:,} rows × {app_test.shape[1]} columns")
    
    # Process training data
    print("\n" + "="*70)
    print("PROCESSING TRAINING DATA")
    print("="*70)
    
    train_clean = clean_application_data(app_train, CONFIG, is_train=True)
    train_engineered = engineer_application_features(train_clean)
    train_final = join_supplementary_data(
        train_engineered, bureau, prev_app, installments, credit_card, pos_cash
    )
    
    # Process test data (using same config)
    print("\n" + "="*70)
    print("PROCESSING TEST DATA")
    print("="*70)
    
    test_clean = clean_application_data(app_test, CONFIG, is_train=False)
    test_engineered = engineer_application_features(test_clean)
    test_final = join_supplementary_data(
        test_engineered, bureau, prev_app, installments, credit_card, pos_cash
    )
    
    # Verify column consistency (except TARGET)
    train_cols = set(train_final.columns) - {'TARGET'}
    test_cols = set(test_final.columns)
    
    if train_cols != test_cols:
        print("\n⚠️  WARNING: Train and test columns differ!")
        print(f"Columns in train but not test: {train_cols - test_cols}")
        print(f"Columns in test but not train: {test_cols - train_cols}")
    else:
        print("\n✅ Train/test consistency verified: identical columns (except TARGET)")
    
    print("\n" + "="*70)
    print("PIPELINE COMPLETE")
    print("="*70)
    print(f"Final training data: {train_final.shape[0]:,} rows × {train_final.shape[1]} columns")
    print(f"Final test data: {test_final.shape[0]:,} rows × {test_final.shape[1]} columns")
    
    return train_final, test_final


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def save_processed_data(train: pl.DataFrame, test: pl.DataFrame, output_dir: str = "."):
    """
    Save processed train and test data to CSV files.
    
    Parameters:
    -----------
    train : pl.DataFrame
        Processed training data
    test : pl.DataFrame
        Processed test data
    output_dir : str
        Directory to save files
    """
    train.write_csv(f"{output_dir}/application_train_processed.csv")
    test.write_csv(f"{output_dir}/application_test_processed.csv")
    print(f"\n✅ Processed data saved to {output_dir}/")


def get_feature_summary(df: pl.DataFrame) -> pl.DataFrame:
    """
    Get summary statistics for all features in processed data.
    
    Parameters:
    -----------
    df : pl.DataFrame
        Processed dataframe
    
    Returns:
    --------
    pl.DataFrame
        Summary statistics
    """
    return df.select(cs.numeric()).describe()


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    """
    Example usage of the feature engineering pipeline.
    """
    
    # Load data
    print("Loading data...")
    app_train = pl.read_csv('c:/Users/tranc/Desktop/Home Credit/application_train.csv')
    app_test = pl.read_csv('c:/Users/tranc/Desktop/Home Credit/application_test.csv')
    bureau = pl.read_csv('c:/Users/tranc/Desktop/Home Credit/bureau.csv')
    prev_app = pl.read_csv('c:/Users/tranc/Desktop/Home Credit/previous_application.csv')
    installments = pl.read_csv('c:/Users/tranc/Desktop/Home Credit/installments_payments.csv')
    credit_card = pl.read_csv('c:/Users/tranc/Desktop/Home Credit/credit_card_balance.csv')
    pos_cash = pl.read_csv('c:/Users/tranc/Desktop/Home Credit/POS_CASH_balance.csv')
    
    # Run pipeline
    train_processed, test_processed = process_pipeline(
        app_train, app_test, bureau, prev_app, installments, credit_card, pos_cash
    )
    
    # Save results
    save_processed_data(train_processed, test_processed, output_dir="c:/Users/tranc/Downloads/e.g-home-credit-project")
    
    # Display summary
    print("\n" + "="*70)
    print("FEATURE SUMMARY")
    print("="*70)
    summary = get_feature_summary(train_processed)
    print(summary)
