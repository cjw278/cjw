import pandas as pd
import numpy as np

def calculate_woe_iv(df, feature, target):
    """
    计算单个变量的 WOE 和 IV
    数学公式参考论文公式 (3-1) 和 (3-2) [cite: 169, 173]
    """
    lst = []
    # 这里假设已经进行了分箱 (qcut 或 cut)
    # 实际代码需加入分箱逻辑，此处简化
    for i in range(df[feature].nunique()):
        val = list(df[feature].unique())[i]
        lst.append({
            'Value': val,
            'All': df[df[feature] == val].count()[feature],
            'Good': df[(df[feature] == val) & (df[target] == 0)].count()[feature],
            'Bad': df[(df[feature] == val) & (df[target] == 1)].count()[feature]
        })
    
    dset = pd.DataFrame(lst)
    dset['Distr_Good'] = dset['Good'] / dset['Good'].sum()
    dset['Distr_Bad'] = dset['Bad'] / dset['Bad'].sum()
    
    # 防止除以 0
    dset['WoE'] = np.log(dset['Distr_Good'] / dset['Distr_Bad'])
    dset['WoE'] = dset['WoE'].replace({np.inf: 0, -np.inf: 0})
    dset['IV'] = (dset['Distr_Good'] - dset['Distr_Bad']) * dset['WoE']
    
    iv = dset['IV'].sum()
    return dset, iv

def run_feature_engineering(df, target):
    # 此处应包含分箱逻辑
    # ...
    # 导出 IV 表
    # iv_df.to_csv('output/tables/iv_summary.csv')
    return df