项目简介 (Introduction)
本项目是一个端到端的个人信贷违约风险预测与解释系统。针对金融信贷场景中普遍存在的类别极度不平衡（Class Imbalance）和模型不可解释（Black-box）问题，本系统提出并实现了一套完整的解决方案。

系统前端采用 Streamlit 框架构建，后端集成 XGBoost 算法，引入 SMOTE 技术处理样本不平衡，利用 贝叶斯优化 (TPE) 进行超参数寻优，并最终通过 SHAP (SHapley Additive exPlanations) 理论实现对个体违约风险的精确归因。

✨ 核心功能 (Key Features)
📊 自动化探索性数据分析 (EDA)：

自动识别并处理异常值（如 96/98 异常编码）。

可视化违约样本分布与特征相关性。

⚖️ 数据不平衡处理：

集成 SMOTE (Synthetic Minority Over-sampling Technique) 算法，有效提升少数类（违约样本）的召回率。

🧠 高性能模型训练：

基于 XGBoost 的集成学习模型。

集成 Hyperopt 贝叶斯优化框架，替代传统网格搜索，自动寻找最优超参数。

📈 多维度模型评估：

实时输出 AUC、KS (Kolmogorov-Smirnov) 值。

动态绘制 ROC 曲线与特征重要性排序图。

🔍 可解释性诊断 (Explainability)：

提供单样本风险诊断功能。

支持自定义上传新数据进行预测。

生成 SHAP 瀑布图 (Waterfall Plot)，直观展示每个特征对违约概率的推高或拉低作用。

📂 目录结构 (Directory Structure)
Plaintext

Credit_Risk_Assessment_System/
├── data/
│   ├── raw/                  # 存放原始数据 (cs-training.csv)
│   └── processed/            # (可选) 存放清洗后的中间数据
├── src/                      # 核心算法模块
│   ├── __init__.py
│   ├── config.py             # 路径与全局变量配置
│   ├── preprocessing.py      # 数据清洗与预处理逻辑
│   ├── features.py           # 特征工程 (WOE/IV)
│   ├── smote_balance.py      # SMOTE 过采样算法
│   ├── models.py             # XGBoost 模型定义与贝叶斯优化
│   ├── evaluation.py         # 绘图与评估指标计算
│   └── interpretation.py     # SHAP 解释模块
├── output/                   # 存放输出的图表或模型文件
├── app.py                    # Streamlit 前端主程序入口
├── requirements.txt          # 项目依赖库
└── README.md                 # 项目说明文档
🚀 快速开始 (Quick Start)
1. 环境准备
确保你的本地环境安装了 Python 3.8 或以上版本。

2. 克隆项目
Bash

git clone https://github.com/你的用户名/你的仓库名.git
cd Credit_Risk_Assessment_System
3. 安装依赖
Bash

pip install -r requirements.txt
4. 准备数据
请下载 Kaggle Give Me Some Credit 数据集，并将 cs-training.csv 放入项目文件夹或在网页端直接上传。

5. 启动系统
在终端运行以下命令：

Bash

streamlit run app.py
系统将在浏览器中自动打开，默认地址为 http://localhost:8501。

🖥️ 使用指南 (Usage)
数据上传：

在左侧导航栏选择“项目介绍 & 数据上传”。

拖入 cs-training.csv 文件。

数据分析：

切换至“探索性数据分析”模块，查看数据分布及相关性热力图。

模型训练：

切换至“模型训练与评估”模块。

点击“开始训练模型”，系统将自动执行：清洗 -> SMOTE平衡 -> 贝叶斯寻优 -> 最终训练。

训练完成后，查看 ROC 曲线、AUC 及 KS 指标。

风险诊断：

切换至“单样本风险诊断”模块。

方式 A：使用测试集中的样本进行回测。

方式 B：上传新的数据文件（如 cs-test.csv），系统会自动清洗并对齐特征。

输入样本索引，点击“开始诊断”，查看该用户的违约概率及 SHAP 归因瀑布图。

🛠️ 技术栈 (Tech Stack)
编程语言: Python

Web 框架: Streamlit

数据处理: Pandas, NumPy

机器学习: XGBoost, Scikit-learn

不平衡处理: Imbalanced-learn (SMOTE)

超参数优化: Hyperopt (Bayesian Optimization)

模型解释: SHAP (SHapley Additive exPlanations)

可视化: Matplotlib, Seaborn

📄 许可证 (License)
本资源仅供学术研究与交流使用。