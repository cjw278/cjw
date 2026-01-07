import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score  # 关键修复：直接导入 roc_auc_score
import pandas as pd
from scipy.stats import ks_2samp

def get_ks(y_true, y_prob):
    """
    计算 KS 值
    公式：KS = max(TPR - FPR)
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    return max(tpr - fpr)

def plot_roc_curve_st(y_test, y_prob):
    """
    [新增修复] 专为 Streamlit 设计的 ROC 绘图函数
    返回：matplotlib figure 对象 (而不是保存文件)
    """
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate (FPR)')
    ax.set_ylabel('True Positive Rate (TPR)')
    ax.set_title('Receiver Operating Characteristic (ROC)')
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    
    return fig

def plot_ks_curve(y_test, y_prob, save_path=None):
    """
    (可选) 传统的 KS 曲线绘制，用于论文插图
    """
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    ks_value = max(tpr - fpr)
    
    fig, ax = plt.subplots()
    ax.plot(1-thresholds, tpr, label='TPR (Cum Bad)')
    ax.plot(1-thresholds, fpr, label='FPR (Cum Good)')
    ax.plot(1-thresholds, tpr - fpr, label=f'KS Diff ({ks_value:.3f})', linestyle='--')
    ax.set_xlabel('Threshold')
    ax.set_title('KS Curve')
    ax.legend()
    
    if save_path:
        plt.savefig(save_path)
    return fig