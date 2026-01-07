from imblearn.over_sampling import SMOTE
import pandas as pd

def apply_smote(X_train, y_train):
    """
    应用 SMOTE 算法处理类别不平衡
    引用：论文 2.2.2 节及 3.4.3 节 [cite: 35, 89]
    注意：只在训练集上进行 SMOTE，严禁在测试集使用！
    """
    print(f"Original shape: {y_train.value_counts()}")
    
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    
    print(f"Resampled shape: {y_res.value_counts()}")
    return X_res, y_res