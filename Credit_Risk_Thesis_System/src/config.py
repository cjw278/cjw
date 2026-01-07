import os

# 路径配置
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'raw', 'cs-training.csv')
OUTPUT_FIG = os.path.join(BASE_DIR, 'output', 'figures')
OUTPUT_TBL = os.path.join(BASE_DIR, 'output', 'tables')

# 变量映射
TARGET = 'SeriousDlqin2yrs'
# 96/98 异常值处理的列
SPECIAL_COLS = ['NumberOfTime30-59DaysPastDueNotWorse', 
                'NumberOfTime60-89DaysPastDueNotWorse', 
                'NumberOfTimes90DaysLate']