import xgboost as xgb
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.metrics import roc_auc_score
import numpy as np

def hyperopt_objective(params, X_train, y_train, X_val, y_val):
    """
    定义贝叶斯优化的目标函数
    """
    clf = xgb.XGBClassifier(
        n_estimators=int(params['n_estimators']),
        max_depth=int(params['max_depth']),
        learning_rate=params['learning_rate'],
        subsample=params['subsample'],
        colsample_bytree=params['colsample_bytree'],
        scale_pos_weight=params['scale_pos_weight'],
        use_label_encoder=False,
        eval_metric='auc',
        n_jobs=-1,
        random_state=42
    )
    
    # 训练并预测
    clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    y_pred = clf.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, y_pred)
    
    # Hyperopt 寻找最小值，所以返回负 AUC
    return {'loss': -auc, 'status': STATUS_OK}

def train_xgboost_bayesian(X_train, y_train, X_val, y_val):
    """
    执行贝叶斯优化寻找最佳参数
    """
    print("正在启动贝叶斯优化...")
    
    # 定义参数搜索空间
    space = {
        'n_estimators': hp.quniform('n_estimators', 100, 500, 50),
        'max_depth': hp.quniform('max_depth', 3, 8, 1),
        'learning_rate': hp.uniform('learning_rate', 0.01, 0.2),
        'subsample': hp.uniform('subsample', 0.6, 1.0),
        'colsample_bytree': hp.uniform('colsample_bytree', 0.6, 1.0),
        'scale_pos_weight': hp.uniform('scale_pos_weight', 1, 15)
    }
    
    trials = Trials()
    best = fmin(fn=lambda params: hyperopt_objective(params, X_train, y_train, X_val, y_val),
                space=space,
                algo=tpe.suggest,
                max_evals=20,  # 尝试 20 次搜索，可根据电脑性能调整
                trials=trials)
    
    print("最佳参数找到:", best)
    return best

def train_final_model(X_train, y_train, best_params):
    """
    [新增修复] 使用最优参数训练最终模型
    注意：Hyperopt 返回的整数参数通常是浮点型（如 5.0），必须强制转为 int
    """
    print("使用最佳参数训练最终模型...")
    
    model = xgb.XGBClassifier(
        n_estimators=int(best_params['n_estimators']),  # 强制转 int
        max_depth=int(best_params['max_depth']),        # 强制转 int
        learning_rate=best_params['learning_rate'],
        subsample=best_params['subsample'],
        colsample_bytree=best_params['colsample_bytree'],
        scale_pos_weight=best_params['scale_pos_weight'],
        use_label_encoder=False,
        eval_metric='auc',
        n_jobs=-1,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    return model