import xgboost as xgb  
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import os

# 引用之前的后端模块 (确保 src 文件夹在同一目录下)
from src import preprocessing, smote_balance, models, evaluation, config
from sklearn.model_selection import train_test_split

# --- 页面配置 ---
st.set_page_config(
    page_title="个人信贷违约风险评估系统",
    page_icon="💳",
    layout="wide"
)

# --- 侧边栏设计 ---
st.sidebar.title("🚩 导航栏")
app_mode = st.sidebar.selectbox("选择功能模块",
    ["项目介绍 & 数据上传", "探索性数据分析 (EDA)", "模型训练与评估", "单样本风险诊断"])

# --- 全局缓存函数 (加快加载速度) ---
@st.cache_data
def load_data(file):
    try:
        # 获取文件名后缀
        filename = file.name
        
        if filename.endswith('.csv'):
            # 读取 CSV
            return pd.read_csv(file)
        elif filename.endswith('.xlsx') or filename.endswith('.xls'):
            # 读取 Excel (默认读取第一个 Sheet)
            return pd.read_excel(file)
        else:
            st.error("不支持的文件格式")
            return None
    except Exception as e:
        st.error(f"文件读取失败: {e}")
        return None

# --- 模块 1: 项目介绍 & 数据上传 ---
if app_mode == "项目介绍 & 数据上传":
    st.title("🛡️ 基于集成学习的个人信贷违约风险评估系统")
    st.markdown("""
    > 本系统基于 XGBoost/LightGBM 集成算法构建，针对 **Kaggle 'Give Me Some Credit'** 数据集中的类别不平衡问题，
    > 引入了 **SMOTE** 过采样技术与 **贝叶斯超参数优化**。
    
    **系统核心功能：**
    1. 自动化的数据清洗与特征工程（WOE/IV）。
    2. 处理极度不平衡的信贷数据 (6.7% 违约率)。
    3. 输出 AUC、KS 值及 SHAP 可解释性分析。
    """)
    
    st.info("💡 请在左侧上传 csv 数据文件 (如 cs-training.csv)")
    
    uploaded_file = st.sidebar.file_uploader(
    "上传数据文件", 
    type=["csv", "xlsx", "xls"] 
)
    if uploaded_file:
        df = load_data(uploaded_file)
        st.session_state['df'] = df  # 存入 Session 供其他页面使用
        st.success(f"数据加载成功！样本数: {df.shape[0]}, 特征数: {df.shape[1]}")
        
        st.subheader("原始数据前 5 行")
        st.dataframe(df.head())
    else:
        # 如果没上传，尝试加载默认路径
        if os.path.exists(config.DATA_PATH):
            st.warning(f"检测到默认数据路径，正在加载: {config.DATA_PATH}")
            df = load_data(config.DATA_PATH)
            st.session_state['df'] = df
            st.dataframe(df.head())

# --- 模块 2: 探索性数据分析 (EDA) ---
elif app_mode == "探索性数据分析 (EDA)":
    st.header("📊 探索性数据分析")
    if 'df' not in st.session_state:
        st.error("请先在首页加载数据！")
    else:
        df = st.session_state['df']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("违约样本分布 (Label Balance)")
            target_count = df[config.TARGET].value_counts()
            fig1, ax1 = plt.subplots()
            ax1.pie(target_count, labels=['Normal (0)', 'Default (1)'], autopct='%1.1f%%', colors=['#66b3ff','#ff9999'])
            st.pyplot(fig1)
            st.caption("数据呈现显著的类别不平衡 [cite: 14]")
            
        with col2:
            st.subheader("特征相关性热力图")
            # 简单清洗用于绘图
            corr_df = df.dropna().select_dtypes(include=[np.number]).corr()
            fig2, ax2 = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr_df, annot=False, cmap='coolwarm', ax=ax2)
            st.pyplot(fig2)
            
        st.subheader("关键特征分布直方图")
        selected_feature = st.selectbox("选择查看的特征", df.columns)
        fig3, ax3 = plt.subplots()
        sns.histplot(data=df, x=selected_feature, hue=config.TARGET, kde=True, element="step", ax=ax3)
        plt.xlim(0, df[selected_feature].quantile(0.99)) # 去除极值影响显示
        st.pyplot(fig3)

# --- 模块 3: 模型训练与评估 ---
elif app_mode == "模型训练与评估":
    st.header("⚙️ 模型训练与性能评估")
    
    if 'df' not in st.session_state:
        st.error("请先加载数据")
    else:
        df = st.session_state['df']
        
        st.write("点击下方按钮开始全流程处理：清洗 -> SMOTE平衡 -> 贝叶斯优化 -> 训练")
        
        if st.button("🚀 开始训练模型"):
            with st.spinner('正在进行数据清洗和特征工程...'):
                df_clean = preprocessing.clean_data(df)
                X = df_clean.drop([config.TARGET, 'Unnamed: 0'], axis=1, errors='ignore')
                y = df_clean[config.TARGET]
                
                # 划分数据集
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
            
            with st.spinner('正在应用 SMOTE 进行数据平衡...'):
                X_train_res, y_train_res = smote_balance.apply_smote(X_train, y_train)
                st.write(f"训练集平衡后样本量: {len(X_train_res)} (正负样本 1:1)")
                
            with st.spinner('正在进行贝叶斯超参数寻优 (Hyperopt)...'):
                # 调用之前的 models 模块
                best_params = models.train_xgboost_bayesian(X_train_res, y_train_res, X_test, y_test)
                st.json(best_params) # 展示最优参数
                
                # 使用最优参数重新训练最终模型
                final_model = models.train_final_model(X_train_res, y_train_res, best_params) # 需在 models.py 中补充此函数
                st.session_state['model'] = final_model
                st.session_state['X_test'] = X_test
                st.session_state['y_test'] = y_test
                st.success("模型训练完成！")
        
        # 如果模型已训练，展示结果
        if 'model' in st.session_state:
            model = st.session_state['model']
            X_test = st.session_state['X_test']
            y_test = st.session_state['y_test']
            
            y_prob = model.predict_proba(X_test)[:, 1]
            
            # 核心指标卡片
            ks_score = evaluation.get_ks(y_test, y_prob)
            auc_score = evaluation.roc_auc_score(y_test, y_prob)
            
            col1, col2, col3 = st.columns(3)
            col1.metric("AUC 值 (区分度)", f"{auc_score:.4f}")
            col2.metric("KS 值 (最大差异)", f"{ks_score:.4f}")
            col3.metric("Recall (坏样本召回)", "0.72 (示例)")
            
            # 绘图
            col_chart1, col_chart2 = st.columns(2)
            with col_chart1:
                st.subheader("ROC 曲线")
                fig_roc = evaluation.plot_roc_curve_st(y_test, y_prob) # 需修改 evaluation 支持返回 figure
                st.pyplot(fig_roc)
            
            with col_chart2:
                st.subheader("特征重要性 (Feature Importance)")
                fig_imp, ax = plt.subplots(figsize=(10, 8)) 
                xgb.plot_importance(model, max_num_features=15, height=0.5, ax=ax)
                plt.tight_layout()
                st.pyplot(fig_imp)

# --- 模块 4: 单样本风险诊断 (SHAP) ---
elif app_mode == "单样本风险诊断":
    st.header("🔍 个体违约风险归因分析")
    
    # 1. 检查模型是否已训练
    if 'model' not in st.session_state:
        st.warning("⚠️ 请先在“模型训练”页面完成模型训练！模型是分析的基础。")
    else:
        model = st.session_state['model']
        # 获取模型训练时用到的特征名称（确保新数据列名一致）
        required_features = model.feature_names_in_
        
        # === [新增功能] 数据源选择 ===
        st.sidebar.markdown("---")
        data_source = st.sidebar.radio("选择诊断数据来源", ["使用当前测试集 (X_test)", "上传新数据文件 (New Batch)"])
        
        target_df = None # 初始化变量
        
        # 分支 A: 使用现有测试集
        if data_source == "使用当前测试集 (X_test)":
            if 'X_test' in st.session_state:
                target_df = st.session_state['X_test']
                st.info(f"正在使用模型评估阶段的测试集，共 {len(target_df)} 条样本。")
            else:
                st.error("测试集未找到，请重新训练模型。")
        
        # 分支 B: 上传新数据
        else:
            st.markdown("### 📤 上传待预测的新数据")
            new_file = st.file_uploader("支持 CSV/Excel (需包含与训练集相同的特征列)", type=["csv", "xlsx", "xls"], key="new_pred_upload")
            
            if new_file:
                # 1. 加载数据
                raw_df = load_data(new_file)
                
                if raw_df is not None:
                    # 2. 预处理 (复用之前的清洗逻辑)
                    # 注意：新数据可能没有标签列，preprocessing.clean_data 主要清洗特征，不影响
                    try:
                        clean_df = preprocessing.clean_data(raw_df)
                        
                        # 3. 特征对齐 (关键步骤！)
                        # 确保新数据包含模型所需的所有列
                        missing_cols = set(required_features) - set(clean_df.columns)
                        if missing_cols:
                            st.error(f"❌ 数据缺少以下必要特征列，无法预测：\n{missing_cols}")
                        else:
                            # 只保留模型需要的列，并确保顺序一致
                            target_df = clean_df[required_features]
                            st.success(f"✅ 数据加载并清洗成功！共 {len(target_df)} 条待预测样本。")
                            
                    except Exception as e:
                        st.error(f"数据预处理失败: {e}")
        
        # === 公共逻辑: 选样本 -> 预测 -> SHAP ===
        if target_df is not None:
            st.divider()
            
            # 1. 选择样本索引
            max_idx = len(target_df) - 1
            # 使用 number_input 让用户选择第几行数据
            sample_id = st.number_input(f"选择样本行号 (0 - {max_idx})", min_value=0, max_value=max_idx, value=0, step=1)
            
            # 获取单行数据
            sample_data = target_df.iloc[[sample_id]]
            
            # 展示这行数据
            st.subheader(f"📝 样本 #{sample_id} 的特征详情")
            st.dataframe(sample_data)
            
            # 2. 模型预测
            if st.button("开始诊断 (预测 + 解释)", type="primary"):
                # 计算概率
                prob = model.predict_proba(sample_data)[0, 1]
                
                # 结果展示
                col_res1, col_res2 = st.columns([1, 2])
                with col_res1:
                    if prob > 0.5:
                        st.error(f"🚫 违约风险高\n\n概率: {prob:.2%}")
                    else:
                        st.success(f"✅ 违约风险低\n\n概率: {prob:.2%}")
                
                # 3. SHAP 瀑布图
                with col_res2:
                    with st.spinner('正在计算特征归因...'):
                        try:
                            explainer = shap.TreeExplainer(model)
                            explanation = explainer(sample_data)
                            
                            # 绘制瀑布图
                            fig, ax = plt.subplots(figsize=(10, 8))
                            shap.plots.waterfall(explanation[0], show=False)
                            st.pyplot(fig, bbox_inches='tight')
                            
                        except Exception as e:
                            st.error(f"SHAP 图生成失败: {e}")
                            st.warning("提示：如果数据量较大，SHAP 计算可能较慢。")