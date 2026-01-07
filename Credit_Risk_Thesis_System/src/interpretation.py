import shap
import matplotlib.pyplot as plt

def explain_model(model, X_train, save_dir):
    """
    使用 SHAP 解释模型
    引用：论文 5.1 节
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)
    
    # 1. Summary Plot (全局重要性)
    plt.figure()
    shap.summary_plot(shap_values, X_train, show=False)
    plt.savefig(f"{save_dir}/shap_summary.png", bbox_inches='tight')
    
    # 2. Dependence Plot (局部非线性依赖)
    # [cite: 235] 重点分析 'RevolvingUtilizationOfUnsecuredLines'
    plt.figure()
    shap.dependence_plot('RevolvingUtilizationOfUnsecuredLines', shap_values, X_train, show=False)
    plt.savefig(f"{save_dir}/shap_dependence_utilization.png", bbox_inches='tight')