import pandas as pd
import numpy as np
from src.config import SPECIAL_COLS

def clean_data(df):
    """
    执行数据清洗逻辑 (修复版：确保无 NaN 传给 SMOTE)
    """
    for col in SPECIAL_COLS:
        # 先将 96/98 标记为 NaN
        df.loc[df[col].isin([96, 98]), col] = np.nan
        # 计算该列（剔除异常值后）的中位数
        median_val = df[col].median()
        # 用中位数填补 NaN
        df[col] = df[col].fillna(median_val)

    # 2. 缺失值处理
    # MonthlyIncome: 使用中位数填补
    if df['MonthlyIncome'].isnull().sum() > 0:
        df['MonthlyIncome'] = df['MonthlyIncome'].fillna(df['MonthlyIncome'].median())
    
    # NumberOfDependents: 使用众数填补 (0)
    if df['NumberOfDependents'].isnull().sum() > 0:
        df['NumberOfDependents'] = df['NumberOfDependents'].fillna(0)
    
    # 3. 剔除极端的异常值 (如 Age=0)
    df = df[df['age'] > 0]
    
    # --- 最终安全检查 ---
    # 确保没有遗漏任何 NaN，否则 SMOTE 会再次崩溃
    df = df.fillna(df.median()) 
    
    return df