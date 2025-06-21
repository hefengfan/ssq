# -*- coding: utf-8 -*-
"""
增强版双色球彩票数据分析与推荐系统
================================

在原有系统基础上增加：
1. 多模型集成（随机森林、梯度提升、神经网络）
2. 特征重要性分析
3. 高级超参数优化
4. 神经网络走势分析
5. 斜连组和密度分析
6. 号码形状和聚散分析
7. 自动学习纠正机制

版本: 6.0 (Enhanced & Neural Network Integrated)
"""

import os
import sys
import json
import time
import datetime
import logging
import io
import random
import math
from collections import Counter, defaultdict
from contextlib import redirect_stdout
from typing import Union, Optional, List, Dict, Tuple, Any
from functools import partial
import itertools

# 第三方库导入
import numpy as np
import pandas as pd
import optuna
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import concurrent.futures

# ==============================================================================
# --- 全局常量与配置 ---
# ==============================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_FILE_PATH = os.path.join(SCRIPT_DIR, '../shuangseqiu.csv')
PROCESSED_CSV_PATH = os.path.join(SCRIPT_DIR, '../shuangseqiu_processed.csv')

# 增强功能开关
ENABLE_NEURAL_NETWORK = True
ENABLE_ENSEMBLE_MODELS = True
ENABLE_FEATURE_IMPORTANCE = True
ENABLE_ADVANCED_PATTERNS = True
ENABLE_DIAGONAL_ANALYSIS = True
ENABLE_DENSITY_ANALYSIS = True
ENABLE_AUTO_CORRECTION = True

# 运行模式配置
ENABLE_OPTUNA_OPTIMIZATION = True
ENABLE_FINAL_COMBO_REVERSE = True
ENABLE_REVERSE_REFILL = True

# 彩票规则配置
RED_BALL_RANGE = range(1, 34)
BLUE_BALL_RANGE = range(1, 17)
RED_ZONES = {'Zone1': (1, 11), 'Zone2': (12, 22), 'Zone3': (23, 33)}

# 分析参数配置
ML_LAG_FEATURES = [1, 3, 5, 10, 15]  # 增加更多滞后特征
ML_INTERACTION_PAIRS = [('red_sum', 'red_odd_count'), ('red_span', 'red_consecutive_count')]
ML_INTERACTION_SELF = ['red_span', 'red_sum']
RECENT_FREQ_WINDOW = 20
BACKTEST_PERIODS_COUNT = 100
OPTIMIZATION_BACKTEST_PERIODS = 20
OPTIMIZATION_TRIALS = 150  # 增加优化试验次数
MIN_POSITIVE_SAMPLES_FOR_ML = 20

# 神经网络配置
NEURAL_NETWORK_PARAMS = {
    'hidden_layer_sizes': (100, 50, 25),
    'activation': 'relu',
    'solver': 'adam',
    'alpha': 0.001,
    'learning_rate': 'adaptive',
    'max_iter': 500,
    'random_state': 42
}

# 集成模型配置
ENSEMBLE_MODELS_CONFIG = {
    'random_forest': {
        'n_estimators': 100,
        'max_depth': 10,
        'min_samples_split': 5,
        'random_state': 42
    },
    'gradient_boosting': {
        'n_estimators': 100,
        'learning_rate': 0.1,
        'max_depth': 6,
        'random_state': 42
    }
}

# 默认权重配置（增强版）
DEFAULT_WEIGHTS = {
    # 反向思维
    'FINAL_COMBO_REVERSE_REMOVE_TOP_PERCENT': 0.25,
    
    # 组合生成
    'NUM_COMBINATIONS_TO_GENERATE': 12,
    'TOP_N_RED_FOR_CANDIDATE': 28,
    'TOP_N_BLUE_FOR_CANDIDATE': 8,
    
    # 红球评分权重
    'FREQ_SCORE_WEIGHT': 25.0,
    'OMISSION_SCORE_WEIGHT': 20.0,
    'MAX_OMISSION_RATIO_SCORE_WEIGHT_RED': 15.0,
    'RECENT_FREQ_SCORE_WEIGHT_RED': 15.0,
    'ML_PROB_SCORE_WEIGHT_RED': 20.0,
    
    # 新增：神经网络权重
    'NEURAL_PROB_SCORE_WEIGHT_RED': 18.0,
    'ENSEMBLE_PROB_SCORE_WEIGHT_RED': 22.0,
    
    # 蓝球评分权重
    'BLUE_FREQ_SCORE_WEIGHT': 25.0,
    'BLUE_OMISSION_SCORE_WEIGHT': 20.0,
    'ML_PROB_SCORE_WEIGHT_BLUE': 40.0,
    'NEURAL_PROB_SCORE_WEIGHT_BLUE': 35.0,
    
    # 组合属性匹配奖励
    'COMBINATION_ODD_COUNT_MATCH_BONUS': 12.0,
    'COMBINATION_BLUE_ODD_MATCH_BONUS': 8.0,
    'COMBINATION_ZONE_MATCH_BONUS': 15.0,
    'COMBINATION_BLUE_SIZE_MATCH_BONUS': 10.0,
    
    # 新增：高级模式匹配奖励
    'DIAGONAL_PATTERN_BONUS': 10.0,
    'DENSITY_PATTERN_BONUS': 12.0,
    'SHAPE_PATTERN_BONUS': 8.0,
    
    # 关联规则参数
    'ARM_MIN_SUPPORT': 0.015,
    'ARM_MIN_CONFIDENCE': 0.5,
    'ARM_MIN_LIFT': 1.4,
    'ARM_COMBINATION_BONUS_WEIGHT': 20.0,
    'ARM_BONUS_LIFT_FACTOR': 0.5,
    'ARM_BONUS_CONF_FACTOR': 0.3,
    
    # 组合多样性控制
    'DIVERSITY_MIN_DIFFERENT_REDS': 3,
}

# LightGBM参数
LGBM_PARAMS = {
    'objective': 'binary',
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,
    'n_estimators': 120,
    'num_leaves': 20,
    'min_child_samples': 12,
    'lambda_l1': 0.1,
    'lambda_l2': 0.1,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.85,
    'bagging_freq': 5,
    'seed': 42,
    'n_jobs': 1,
    'verbose': -1,
}

# ==============================================================================
# --- 日志系统配置 ---
# ==============================================================================

console_formatter = logging.Formatter('%(message)s')
detailed_formatter = logging.Formatter('%(asctime)s - %(levelname)s - [%(funcName)s] - %(message)s')

logger = logging.getLogger('enhanced_ssq_analyzer')
logger.setLevel(logging.DEBUG)
logger.propagate = False

progress_logger = logging.getLogger('progress_logger')
progress_logger.setLevel(logging.INFO)
progress_logger.propagate = False

global_console_handler = logging.StreamHandler(sys.stdout)
global_console_handler.setFormatter(console_formatter)

progress_console_handler = logging.StreamHandler(sys.stdout)
progress_console_handler.setFormatter(logging.Formatter('%(message)s'))

logger.addHandler(global_console_handler)
progress_logger.addHandler(progress_console_handler)

def set_console_verbosity(level=logging.INFO, use_simple_formatter=False):
    """动态设置主日志记录器在控制台的输出级别和格式。"""
    global_console_handler.setLevel(level)
    global_console_handler.setFormatter(console_formatter if use_simple_formatter else detailed_formatter)

# ==============================================================================
# --- 核心工具函数 ---
# ==============================================================================

class SuppressOutput:
    """上下文管理器，用于临时抑制标准输出和捕获标准错误。"""
    def __init__(self, suppress_stdout=True, capture_stderr=True):
        self.suppress_stdout, self.capture_stderr = suppress_stdout, capture_stderr
        self.old_stdout, self.old_stderr, self.stdout_io, self.stderr_io = None, None, None, None
    
    def __enter__(self):
        if self.suppress_stdout: 
            self.old_stdout, self.stdout_io = sys.stdout, io.StringIO()
            sys.stdout = self.stdout_io
        if self.capture_stderr: 
            self.old_stderr, self.stderr_io = sys.stderr, io.StringIO()
            sys.stderr = self.stderr_io
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.capture_stderr and self.old_stderr:
            sys.stderr = self.old_stderr
            captured = self.stderr_io.getvalue()
            self.stderr_io.close()
            if captured.strip(): 
                logger.warning(f"捕获到标准错误:\n{captured.strip()}")
        if self.suppress_stdout and self.old_stdout:
            sys.stdout = self.old_stdout
            self.stdout_io.close()
        return False

def get_prize_level(red_hits: int, blue_hit: bool) -> Optional[str]:
    """根据红球和蓝球的命中个数，确定中奖等级。"""
    if blue_hit:
        if red_hits == 6: return "一等奖"
        if red_hits == 5: return "三等奖"
        if red_hits == 4: return "四等奖"
        if red_hits == 3: return "五等奖"
        if red_hits <= 2: return "六等奖"
    else:
        if red_hits == 6: return "二等奖"
        if red_hits == 5: return "四等奖"
        if red_hits == 4: return "五等奖"
    return None

def format_time(seconds: float) -> str:
    """将秒数格式化为易于阅读的 HH:MM:SS 字符串。"""
    if seconds < 0: return "00:00:00"
    hours, remainder = divmod(seconds, 3600)
    minutes, sec = divmod(remainder, 60)
    return f"{int(hours):02d}:{int(minutes):02d}:{int(sec):02d}"

# ==============================================================================
# --- 数据处理模块 ---
# ==============================================================================

def load_data(file_path: str) -> Optional[pd.DataFrame]:
    """从CSV文件加载数据，并能自动尝试多种常用编码格式。"""
    if not os.path.exists(file_path):
        logger.error(f"数据文件未找到: {file_path}")
        return None
    
    for enc in ['utf-8', 'gbk', 'latin-1']:
        try:
            return pd.read_csv(file_path, encoding=enc)
        except UnicodeDecodeError:
            continue
        except Exception as e:
            logger.error(f"使用编码 {enc} 加载 {file_path} 时出错: {e}")
            return None
    
    logger.error(f"无法使用任何支持的编码打开文件 {file_path}。")
    return None

def clean_and_structure(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """清洗和结构化原始DataFrame，确保数据类型正确。"""
    if df is None or df.empty: 
        return None
    
    required_cols = ['期号', '红球', '蓝球']
    if not all(col in df.columns for col in required_cols):
        logger.error(f"输入数据缺少必要列: {required_cols}")
        return None

    df.dropna(subset=required_cols, inplace=True)
    
    try:
        df['期号'] = pd.to_numeric(df['期号'], errors='coerce')
        df.dropna(subset=['期号'], inplace=True)
        df = df.astype({'期号': int})
    except (ValueError, TypeError) as e:
        logger.error(f"转换'期号'为整数时失败: {e}")
        return None

    df.sort_values(by='期号', ascending=True, inplace=True)
    df.reset_index(drop=True, inplace=True)

    parsed_rows = []
    for _, row in df.iterrows():
        try:
            reds = sorted([int(r) for r in str(row['红球']).split(',')])
            blue = int(row['蓝球'])
            
            if len(reds) != 6 or not all(r in RED_BALL_RANGE for r in reds) or blue not in BLUE_BALL_RANGE:
                logger.warning(f"期号 {row['期号']} 的数据无效，已跳过: 红球={reds}, 蓝球={blue}")
                continue
            
            record = {'期号': row['期号'], 'blue': blue}
            for i, r in enumerate(reds):
                record[f'red{i+1}'] = r
            
            if '日期' in row and pd.notna(row['日期']):
                record['日期'] = row['日期']
            
            parsed_rows.append(record)
            
        except (ValueError, TypeError):
            logger.warning(f"解析期号 {row['期号']} 的号码时失败，已跳过。")
            continue
            
    return pd.DataFrame(parsed_rows) if parsed_rows else None

def enhanced_feature_engineer(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """增强版特征工程，包含更多高级特征。"""
    if df is None or df.empty: 
        return None
    
    df_fe = df.copy()
    red_cols = [f'red{i+1}' for i in range(6)]
    
    # 基本统计特征
    df_fe['red_sum'] = df_fe[red_cols].sum(axis=1)
    df_fe['red_span'] = df_fe[red_cols].max(axis=1) - df_fe[red_cols].min(axis=1)
    df_fe['red_odd_count'] = df_fe[red_cols].apply(lambda r: sum(x % 2 != 0 for x in r), axis=1)
    df_fe['red_mean'] = df_fe[red_cols].mean(axis=1)
    df_fe['red_std'] = df_fe[red_cols].std(axis=1)
    
    # 区间特征
    for zone, (start, end) in RED_ZONES.items():
        df_fe[f'red_{zone}_count'] = df_fe[red_cols].apply(
            lambda r: sum(start <= x <= end for x in r), axis=1
        )
    
    # 形态特征
    def count_consecutive(row): 
        return sum(1 for i in range(5) if row.iloc[i+1] - row.iloc[i] == 1)
    
    df_fe['red_consecutive_count'] = df_fe[red_cols].apply(count_consecutive, axis=1)
    
    # 重号特征
    red_sets = df_fe[red_cols].apply(set, axis=1)
    prev_red_sets = red_sets.shift(1)
    df_fe['red_repeat_count'] = [
        len(current.intersection(prev)) if isinstance(prev, set) else 0 
        for current, prev in zip(red_sets, prev_red_sets)
    ]
    
    # 新增：高级特征
    if ENABLE_ADVANCED_PATTERNS:
        # 密度特征
        df_fe['red_density'] = df_fe.apply(lambda row: calculate_density(
            [row[f'red{i+1}'] for i in range(6)]
        ), axis=1)
        
        # 斜连特征
        df_fe['red_diagonal_count'] = df_fe.apply(lambda row: calculate_diagonal_patterns(
            [row[f'red{i+1}'] for i in range(6)]
        ), axis=1)
        
        # 形状特征（聚散度）
        df_fe['red_clustering'] = df_fe.apply(lambda row: calculate_clustering_score(
            [row[f'red{i+1}'] for i in range(6)]
        ), axis=1)
    
    # 蓝球特征
    df_fe['blue_is_odd'] = (df_fe['blue'] % 2 != 0).astype(int)
    df_fe['blue_is_large'] = (df_fe['blue'] > 8).astype(int)
    df_fe['blue_is_prime'] = df_fe['blue'].apply(is_prime).astype(int)
    
    return df_fe

def calculate_density(red_balls: List[int]) -> float:
    """计算红球号码的密度分布。"""
    if len(red_balls) < 2:
        return 0.0
    
    # 计算相邻号码间的平均间距
    sorted_balls = sorted(red_balls)
    gaps = [sorted_balls[i+1] - sorted_balls[i] for i in range(len(sorted_balls)-1)]
    avg_gap = sum(gaps) / len(gaps)
    
    # 密度 = 1 / 平均间距（间距越小，密度越大）
    return 1.0 / (avg_gap + 0.1)

def calculate_diagonal_patterns(red_balls: List[int]) -> int:
    """计算斜连模式的数量。"""
    sorted_balls = sorted(red_balls)
    diagonal_count = 0
    
    # 检查不同间隔的斜连
    for interval in [7, 11, 12]:  # 常见的斜连间隔
        for i in range(len(sorted_balls)):
            for j in range(i+1, len(sorted_balls)):
                if sorted_balls[j] - sorted_balls[i] == interval:
                    diagonal_count += 1
    
    return diagonal_count

def calculate_clustering_score(red_balls: List[int]) -> float:
    """计算号码的聚散程度。"""
    if len(red_balls) < 2:
        return 0.0
    
    sorted_balls = sorted(red_balls)
    total_span = sorted_balls[-1] - sorted_balls[0]
    
    if total_span == 0:
        return 1.0  # 完全聚集
    
    # 聚散度 = 实际跨度 / 理论最大跨度
    max_possible_span = 33 - 1  # 红球最大跨度
    clustering_score = 1.0 - (total_span / max_possible_span)
    
    return clustering_score

def is_prime(n: int) -> bool:
    """判断一个数是否为质数。"""
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    
    for i in range(3, int(n**0.5) + 1, 2):
        if n % i == 0:
            return False
    return True

def create_enhanced_lagged_features(df: pd.DataFrame, lags: List[int]) -> Optional[pd.DataFrame]:
    """创建增强版滞后特征，包含更多交互特征。"""
    if df is None or df.empty or not lags: 
        return None
    
    feature_cols = [col for col in df.columns if 'red_' in col or 'blue_' in col]
    df_features = df[feature_cols].copy()
    
    # 创建交互特征
    for c1, c2 in ML_INTERACTION_PAIRS:
        if c1 in df_features and c2 in df_features: 
            df_features[f'{c1}_x_{c2}'] = df_features[c1] * df_features[c2]
    
    for c in ML_INTERACTION_SELF:
        if c in df_features: 
            df_features[f'{c}_sq'] = df_features[c]**2
            df_features[f'{c}_log'] = np.log1p(df_features[c])  # log(1+x)
    
    # 创建滞后特征
    all_feature_cols = df_features.columns.tolist()
    lagged_dfs = []
    
    for lag in lags:
        lagged_df = df_features[all_feature_cols].shift(lag).add_suffix(f'_lag{lag}')
        lagged_dfs.append(lagged_df)
    
    final_df = pd.concat(lagged_dfs, axis=1)
    final_df.dropna(inplace=True)
    
    return final_df if not final_df.empty else None

# ==============================================================================
# --- 增强分析模块 ---
# ==============================================================================

def analyze_frequency_omission(df: pd.DataFrame) -> dict:
    """分析所有号码的频率、当前遗漏、平均遗漏、最大遗漏和近期频率。"""
    if df is None or df.empty: 
        return {}
    
    red_cols = [f'red{i+1}' for i in range(6)]
    total_periods = len(df)
    most_recent_idx = total_periods - 1
    
    # 频率计算
    all_reds_flat = df[red_cols].values.flatten()
    red_freq = Counter(all_reds_flat)
    blue_freq = Counter(df['blue'])
    
    # 遗漏和近期频率计算
    current_omission = {}
    max_hist_omission = {}
    recent_N_freq = Counter()
    
    for num in RED_BALL_RANGE:
        app_indices = df.index[(df[red_cols] == num).any(axis=1)].tolist()
        if app_indices:
            current_omission[num] = most_recent_idx - app_indices[-1]
            gaps = np.diff([0] + app_indices) - 1
            max_hist_omission[num] = max(gaps.max(), current_omission[num])
        else:
            current_omission[num] = max_hist_omission[num] = total_periods
    
    # 计算近期频率
    if total_periods >= RECENT_FREQ_WINDOW:
        recent_N_freq.update(df.tail(RECENT_FREQ_WINDOW)[red_cols].values.flatten())
    
    for num in BLUE_BALL_RANGE:
        app_indices = df.index[df['blue'] == num].tolist()
        current_omission[f'blue_{num}'] = (
            most_recent_idx - app_indices[-1] if app_indices else total_periods
        )
    
    # 平均间隔
    avg_interval = {}
    for num in RED_BALL_RANGE:
        avg_interval[num] = total_periods / (red_freq.get(num, 0) + 1e-9)
    
    for num in BLUE_BALL_RANGE:
        avg_interval[f'blue_{num}'] = total_periods / (blue_freq.get(num, 0) + 1e-9)
    
    return {
        'red_freq': red_freq,
        'blue_freq': blue_freq,
        'current_omission': current_omission,
        'average_interval': avg_interval,
        'max_historical_omission_red': max_hist_omission,
        'recent_N_freq_red': recent_N_freq
    }

def analyze_enhanced_patterns(df: pd.DataFrame) -> dict:
    """增强版模式分析，包含更多高级模式。"""
    if df is None or df.empty: 
        return {}
    
    res = {}
    
    def safe_mode(s): 
        return s.mode().iloc[0] if not s.empty and not s.mode().empty else None
    
    # 基本模式
    for col, name in [('red_sum', 'sum'), ('red_span', 'span'), ('red_odd_count', 'odd_count')]:
        if col in df.columns: 
            res[f'most_common_{name}'] = safe_mode(df[col])
    
    # 区间分布模式
    zone_cols = [f'red_{zone}_count' for zone in RED_ZONES.keys()]
    if all(c in df.columns for c in zone_cols):
        dist_counts = df[zone_cols].apply(tuple, axis=1).value_counts()
        if not dist_counts.empty: 
            res['most_common_zone_distribution'] = dist_counts.index[0]
    
    # 蓝球模式
    if 'blue_is_odd' in df.columns: 
        res['most_common_blue_is_odd'] = safe_mode(df['blue_is_odd'])
    if 'blue_is_large' in df.columns: 
        res['most_common_blue_is_large'] = safe_mode(df['blue_is_large'])
    
    # 高级模式
    if ENABLE_ADVANCED_PATTERNS:
        if 'red_density' in df.columns:
            res['most_common_density_range'] = categorize_density(safe_mode(df['red_density']))
        if 'red_diagonal_count' in df.columns:
            res['most_common_diagonal_count'] = safe_mode(df['red_diagonal_count'])
        if 'red_clustering' in df.columns:
            res['most_common_clustering_level'] = categorize_clustering(safe_mode(df['red_clustering']))
    
    return res

def categorize_density(density_value: float) -> str:
    """将密度值分类为高、中、低。"""
    if density_value is None:
        return "未知"
    if density_value > 0.3:
        return "高密度"
    elif density_value > 0.15:
        return "中密度"
    else:
        return "低密度"

def categorize_clustering(clustering_value: float) -> str:
    """将聚散度分类为聚集、分散、均匀。"""
    if clustering_value is None:
        return "未知"
    if clustering_value > 0.7:
        return "聚集"
    elif clustering_value > 0.4:
        return "均匀"
    else:
        return "分散"

def analyze_associations(df: pd.DataFrame, weights_config: Dict) -> pd.DataFrame:
    """使用Apriori算法挖掘红球号码之间的关联规则。"""
    min_s = weights_config.get('ARM_MIN_SUPPORT', 0.01)
    min_c = weights_config.get('ARM_MIN_CONFIDENCE', 0.5)
    min_l = weights_config.get('ARM_MIN_LIFT', 1.5)
    
    red_cols = [f'red{i+1}' for i in range(6)]
    if df is None or df.empty: 
        return pd.DataFrame()
    
    try:
        transactions = df[red_cols].astype(str).values.tolist()
        te = TransactionEncoder()
        te_ary = te.fit(transactions).transform(transactions)
        df_oh = pd.DataFrame(te_ary, columns=te.columns_)
        
        frequent_itemsets = apriori(df_oh, min_support=min_s, use_colnames=True)
        if frequent_itemsets.empty: 
            return pd.DataFrame()
        
        rules = association_rules(frequent_itemsets, metric="lift", min_threshold=min_l)
        strong_rules = rules[rules['confidence'] >= min_c].sort_values(by='lift', ascending=False)
        return strong_rules
        
    except Exception as e:
        logger.error(f"关联规则分析失败: {e}")
        return pd.DataFrame()

def calculate_enhanced_scores(freq_data: Dict, probabilities: Dict, weights: Dict) -> Dict[str, Dict[int, float]]:
    """增强版评分计算，包含神经网络和集成模型的预测。"""
    r_scores, b_scores = {}, {}
    
    r_freq = freq_data.get('red_freq', {})
    b_freq = freq_data.get('blue_freq', {})
    omission = freq_data.get('current_omission', {})
    avg_int = freq_data.get('average_interval', {})
    max_hist_o = freq_data.get('max_historical_omission_red', {})
    recent_freq = freq_data.get('recent_N_freq_red', {})
    
    r_pred = probabilities.get('red', {})
    b_pred = probabilities.get('blue', {})
    r_neural = probabilities.get('red_neural', {})
    b_neural = probabilities.get('blue_neural', {})
    r_ensemble = probabilities.get('red_ensemble', {})
    
    # 红球评分
    for num in RED_BALL_RANGE:
        # 基础分数
        freq_s = (r_freq.get(num, 0)) * weights['FREQ_SCORE_WEIGHT']
        omit_s = np.exp(-0.005 * (omission.get(num, 0) - avg_int.get(num, 0))**2) * weights['OMISSION_SCORE_WEIGHT']
        max_o_ratio = (omission.get(num, 0) / max_hist_o.get(num, 1)) if max_hist_o.get(num, 0) > 0 else 0
        max_o_s = max_o_ratio * weights['MAX_OMISSION_RATIO_SCORE_WEIGHT_RED']
        recent_s = recent_freq.get(num, 0) * weights['RECENT_FREQ_SCORE_WEIGHT_RED']
        ml_s = r_pred.get(num, 0.0) * weights['ML_PROB_SCORE_WEIGHT_RED']
        
        # 新增：神经网络和集成模型分数
        neural_s = r_neural.get(num, 0.0) * weights.get('NEURAL_PROB_SCORE_WEIGHT_RED', 0)
        ensemble_s = r_ensemble.get(num, 0.0) * weights.get('ENSEMBLE_PROB_SCORE_WEIGHT_RED', 0)
        
        r_scores[num] = sum([freq_s, omit_s, max_o_s, recent_s, ml_s, neural_s, ensemble_s])
    
    # 蓝球评分
    for num in BLUE_BALL_RANGE:
        freq_s = (b_freq.get(num, 0)) * weights['BLUE_FREQ_SCORE_WEIGHT']
        omit_s = np.exp(-0.01 * (omission.get(f'blue_{num}', 0) - avg_int.get(f'blue_{num}', 0))**2) * weights['BLUE_OMISSION_SCORE_WEIGHT']
        ml_s = b_pred.get(num, 0.0) * weights['ML_PROB_SCORE_WEIGHT_BLUE']
        neural_s = b_neural.get(num, 0.0) * weights.get('NEURAL_PROB_SCORE_WEIGHT_BLUE', 0)
        
        b_scores[num] = sum([freq_s, omit_s, ml_s, neural_s])

    # 归一化分数
    def normalize_scores(scores_dict):
        if not scores_dict: 
            return {}
        vals = list(scores_dict.values())
        min_v, max_v = min(vals), max(vals)
        if max_v == min_v: 
            return {k: 50.0 for k in scores_dict}
        return {k: (v - min_v) / (max_v - min_v) * 100 for k, v in scores_dict.items()}

    return {
        'red_scores': normalize_scores(r_scores), 
        'blue_scores': normalize_scores(b_scores)
    }

# ==============================================================================
# --- 增强机器学习模块 ---
# ==============================================================================

def train_single_model(model_type: str, ball_type_str: str, ball_number: int, 
                      X_train: pd.DataFrame, y_train: pd.Series) -> Tuple[Optional[Any], Optional[str]]:
    """训练单个模型（支持多种模型类型）。"""
    if y_train.sum() < MIN_POSITIVE_SAMPLES_FOR_ML or y_train.nunique() < 2:
        return None, None
    
    model_key = f'{model_type}_{ball_number}'
    
    try:
        if model_type == 'lgbm':
            model_params = LGBM_PARAMS.copy()
            if (pos_count := y_train.sum()) > 0:
                model_params['scale_pos_weight'] = (len(y_train) - pos_count) / pos_count
            model = LGBMClassifier(**model_params)
            
        elif model_type == 'neural' and ENABLE_NEURAL_NETWORK:
            # 标准化特征用于神经网络
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_train)
            model = MLPClassifier(**NEURAL_NETWORK_PARAMS)
            model.fit(X_scaled, y_train)
            # 保存scaler以便预测时使用
            model.scaler = scaler
            return model, model_key
            
        elif model_type == 'rf' and ENABLE_ENSEMBLE_MODELS:
            model = RandomForestClassifier(**ENSEMBLE_MODELS_CONFIG['random_forest'])
            
        elif model_type == 'gb' and ENABLE_ENSEMBLE_MODELS:
            model = GradientBoostingClassifier(**ENSEMBLE_MODELS_CONFIG['gradient_boosting'])
            
        else:
            return None, None
        
        if model_type != 'neural':  # neural已经在上面fit了
            model.fit(X_train, y_train)
        
        return model, model_key
        
    except Exception as e:
        logger.debug(f"训练{model_type} for {ball_type_str} {ball_number} 失败: {e}")
        return None, None

def train_enhanced_prediction_models(df_train_raw: pd.DataFrame, ml_lags_list: List[int]) -> Optional[Dict[str, Any]]:
    """训练增强版预测模型，包含多种模型类型。"""
    X = create_enhanced_lagged_features(df_train_raw.copy(), ml_lags_list)
    if X is None or X.empty:
        logger.warning("创建滞后特征失败或结果为空，跳过模型训练。")
        return None
    
    target_df = df_train_raw.loc[X.index].copy()
    if target_df.empty: 
        return None
    
    red_cols = [f'red{i+1}' for i in range(6)]
    trained_models = {
        'red': {}, 'blue': {}, 
        'red_neural': {}, 'blue_neural': {},
        'red_ensemble': {},
        'feature_cols': X.columns.tolist()
    }
    
    # 定义要训练的模型类型
    model_types = ['lgbm']
    if ENABLE_NEURAL_NETWORK:
        model_types.append('neural')
    if ENABLE_ENSEMBLE_MODELS:
        model_types.extend(['rf', 'gb'])
    
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = {}
        
        # 为每个红球和每种模型类型提交训练任务
        for ball_num in RED_BALL_RANGE:
            y = target_df[red_cols].eq(ball_num).any(axis=1).astype(int)
            for model_type in model_types:
                future = executor.submit(train_single_model, model_type, '红球', ball_num, X, y)
                futures[future] = ('red', ball_num, model_type)
        
        # 为每个蓝球和每种模型类型提交训练任务
        for ball_num in BLUE_BALL_RANGE:
            y = target_df['blue'].eq(ball_num).astype(int)
            for model_type in model_types:
                future = executor.submit(train_single_model, model_type, '蓝球', ball_num, X, y)
                futures[future] = ('blue', ball_num, model_type)
        
        for future in concurrent.futures.as_completed(futures):
            ball_type, ball_num, model_type = futures[future]
            try:
                model, model_key = future.result()
                if model and model_key:
                    if model_type == 'lgbm':
                        trained_models[ball_type][model_key] = model
                    elif model_type == 'neural':
                        trained_models[f'{ball_type}_neural'][model_key] = model
                    elif model_type in ['rf', 'gb']:
                        if f'{ball_type}_ensemble' not in trained_models:
                            trained_models[f'{ball_type}_ensemble'] = {}
                        trained_models[f'{ball_type}_ensemble'][model_key] = model
                        
            except Exception as e:
                logger.error(f"训练球号 {ball_num} ({ball_type}, {model_type}) 的模型时出现异常: {e}")

    return trained_models if any(trained_models[key] for key in trained_models if key != 'feature_cols') else None

def predict_enhanced_probabilities(df_historical: pd.DataFrame, trained_models: Optional[Dict], 
                                 ml_lags_list: List[int]) -> Dict[str, Dict[int, float]]:
    """使用增强版模型预测下一期每个号码的出现概率。"""
    probs = {'red': {}, 'blue': {}, 'red_neural': {}, 'blue_neural': {}, 'red_ensemble': {}}
    
    if not trained_models or not (feat_cols := trained_models.get('feature_cols')):
        return probs
    
    max_lag = max(ml_lags_list) if ml_lags_list else 0
    if len(df_historical) < max_lag + 1:
        return probs
    
    predict_X = create_enhanced_lagged_features(df_historical.tail(max_lag + 1), ml_lags_list)
    if predict_X is None:
        return probs
    
    predict_X = predict_X.reindex(columns=feat_cols, fill_value=0)
    
    # 预测各种模型类型
    model_mappings = [
        ('red', RED_BALL_RANGE, 'lgbm'),
        ('blue', BLUE_BALL_RANGE, 'lgbm'),
        ('red_neural', RED_BALL_RANGE, 'neural'),
        ('blue_neural', BLUE_BALL_RANGE, 'neural'),
        ('red_ensemble', RED_BALL_RANGE, 'ensemble')
    ]
    
    for prob_key, ball_range, model_suffix in model_mappings:
        model_dict = trained_models.get(prob_key, {})
        if not model_dict:
            continue
            
        for ball_num in ball_range:
            model_key = f'{model_suffix}_{ball_num}'
            if model := model_dict.get(model_key):
                try:
                    if hasattr(model, 'scaler'):  # 神经网络模型
                        X_scaled = model.scaler.transform(predict_X)
                        prob = model.predict_proba(X_scaled)[0, 1]
                    else:
                        prob = model.predict_proba(predict_X)[0, 1]
                    probs[prob_key][ball_num] = prob
                except Exception:
                    pass
    
    # 集成多个模型的预测结果
    if ENABLE_ENSEMBLE_MODELS and 'red_ensemble' in probs:
        ensemble_probs = {}
        for ball_num in RED_BALL_RANGE:
            model_probs = []
            for model_type in ['rf', 'gb']:
                model_key = f'{model_type}_{ball_num}'
                if model := trained_models.get('red_ensemble', {}).get(model_key):
                    try:
                        prob = model.predict_proba(predict_X)[0, 1]
                        model_probs.append(prob)
                    except Exception:
                        pass
            if model_probs:
                ensemble_probs[ball_num] = np.mean(model_probs)
        probs['red_ensemble'] = ensemble_probs
    
    return probs

def analyze_feature_importance(trained_models: Dict, feature_names: List[str]) -> Dict[str, Any]:
    """分析特征重要性。"""
    if not ENABLE_FEATURE_IMPORTANCE or not trained_models:
        return {}
    
    importance_analysis = {}
    
    # 分析LightGBM模型的特征重要性
    lgbm_models = trained_models.get('red', {})
    if lgbm_models:
        all_importances = []
        for model_key, model in lgbm_models.items():
            if hasattr(model, 'feature_importances_'):
                all_importances.append(model.feature_importances_)
        
        if all_importances:
            avg_importance = np.mean(all_importances, axis=0)
            feature_importance_dict = dict(zip(feature_names, avg_importance))
            
            # 排序并获取前10个最重要的特征
            sorted_features = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)
            importance_analysis['top_features'] = sorted_features[:10]
            importance_analysis['feature_importance_dict'] = feature_importance_dict
    
    return importance_analysis

# ==============================================================================
# --- 增强组合生成模块 ---
# ==============================================================================

def generate_enhanced_combinations(scores_data: Dict, pattern_data: Dict, arm_rules: pd.DataFrame, 
                                 weights_config: Dict) -> Tuple[List[Dict], List[str]]:
    """生成增强版推荐组合，包含更多策略。"""
    num_to_gen = weights_config['NUM_COMBINATIONS_TO_GENERATE']
    r_scores = scores_data.get('red_scores', {})
    b_scores = scores_data.get('blue_scores', {})
    
    if not r_scores or not b_scores: 
        return [], ["无法生成推荐 (分数数据缺失)。"]

    # 构建候选池
    top_n_red = int(weights_config['TOP_N_RED_FOR_CANDIDATE'])
    top_n_blue = int(weights_config['TOP_N_BLUE_FOR_CANDIDATE'])
    
    r_cand_pool = [n for n, _ in sorted(r_scores.items(), key=lambda i: i[1], reverse=True)[:top_n_red]]
    b_cand_pool = [n for n, _ in sorted(b_scores.items(), key=lambda i: i[1], reverse=True)[:top_n_blue]]
    
    if len(r_cand_pool) < 6 or not b_cand_pool: 
        return [], ["候选池号码不足。"]

    # 生成大量初始组合
    large_pool_size = max(num_to_gen * 60, 600)  # 增加候选池大小
    gen_pool, unique_combos = [], set()
    
    r_weights = np.array([r_scores.get(n, 0) + 1 for n in r_cand_pool])
    r_probs = r_weights / r_weights.sum() if r_weights.sum() > 0 else None
    
    for _ in range(large_pool_size * 25):
        if len(gen_pool) >= large_pool_size: 
            break
            
        if r_probs is not None:
            reds = sorted(np.random.choice(r_cand_pool, size=6, replace=False, p=r_probs).tolist())
        else:
            reds = sorted(random.sample(r_cand_pool, 6))
            
        blue = random.choice(b_cand_pool)
        
        combo_tuple = (tuple(reds), blue)
        if combo_tuple not in unique_combos:
            gen_pool.append({'red': reds, 'blue': blue})
            unique_combos.add(combo_tuple)

    # 增强版评分和筛选
    scored_combos = []
    for c in gen_pool:
        # 基础分
        base_score = sum(r_scores.get(r, 0) for r in c['red']) + b_scores.get(c['blue'], 0)
        
        # 模式匹配奖励
        pattern_bonus = calculate_pattern_bonus(c, pattern_data, weights_config)
        
        # ARM奖励
        arm_bonus = calculate_arm_bonus(c, arm_rules, weights_config)
        
        total_score = base_score + pattern_bonus + arm_bonus
        
        scored_combos.append({
            'combination': c, 
            'score': total_score, 
            'red_tuple': tuple(c['red']),
            'base_score': base_score,
            'pattern_bonus': pattern_bonus,
            'arm_bonus': arm_bonus
        })

    # 多样性筛选和最终选择
    sorted_combos = sorted(scored_combos, key=lambda x: x['score'], reverse=True)
    final_recs = []
    max_common = 6 - int(weights_config.get('DIVERSITY_MIN_DIFFERENT_REDS', 3))
    
    if sorted_combos:
        final_recs.append(sorted_combos.pop(0))
        
        for cand in sorted_combos:
            if len(final_recs) >= num_to_gen: 
                break
                
            # 检查多样性
            if all(len(set(cand['red_tuple']) & set(rec['red_tuple'])) <= max_common 
                   for rec in final_recs):
                final_recs.append(cand)

    # 应用反向思维策略
    applied_msg = ""
    if ENABLE_FINAL_COMBO_REVERSE:
        num_to_remove = int(len(final_recs) * weights_config.get('FINAL_COMBO_REVERSE_REMOVE_TOP_PERCENT', 0))
        if 0 < num_to_remove < len(final_recs):
            removed = final_recs[:num_to_remove]
            final_recs = final_recs[num_to_remove:]
            applied_msg = f" (反向策略: 移除前{num_to_remove}注"
            
            if ENABLE_REVERSE_REFILL:
                refill_candidates = [c for c in sorted_combos 
                                   if c not in final_recs and c not in removed]
                final_recs.extend(refill_candidates[:num_to_remove])
                applied_msg += "并补充)"
            else:
                applied_msg += ")"

    final_recs = sorted(final_recs, key=lambda x: x['score'], reverse=True)[:num_to_gen]

    # 生成输出字符串
    output_strs = [f"增强版推荐组合 (Top {len(final_recs)}{applied_msg}):"]
    for i, c in enumerate(final_recs):
        r_str = ' '.join(f'{n:02d}' for n in c['combination']['red'])
        b_str = f"{c['combination']['blue']:02d}"
        output_strs.append(
            f"  注 {i+1}: 红球 [{r_str}] 蓝球 [{b_str}] "
            f"(总分: {c['score']:.2f} = 基础: {c['base_score']:.2f} + "
            f"模式: {c['pattern_bonus']:.2f} + ARM: {c['arm_bonus']:.2f})"
        )
    
    return final_recs, output_strs

def calculate_pattern_bonus(combination: Dict, pattern_data: Dict, weights_config: Dict) -> float:
    """计算模式匹配奖励分数。"""
    bonus = 0.0
    reds = combination['red']
    blue = combination['blue']
    
    # 基础模式匹配
    red_odd_count = sum(1 for r in reds if r % 2 != 0)
    if red_odd_count == pattern_data.get('most_common_odd_count'):
        bonus += weights_config.get('COMBINATION_ODD_COUNT_MATCH_BONUS', 0)
    
    blue_is_odd = blue % 2 != 0
    if blue_is_odd == pattern_data.get('most_common_blue_is_odd'):
        bonus += weights_config.get('COMBINATION_BLUE_ODD_MATCH_BONUS', 0)
    
    blue_is_large = blue > 8
    if blue_is_large == pattern_data.get('most_common_blue_is_large'):
        bonus += weights_config.get('COMBINATION_BLUE_SIZE_MATCH_BONUS', 0)
    
    # 区间分布匹配
    zone_dist = []
    for zone, (start, end) in RED_ZONES.items():
        count = sum(1 for r in reds if start <= r <= end)
        zone_dist.append(count)
    
    if tuple(zone_dist) == pattern_data.get('most_common_zone_distribution'):
        bonus += weights_config.get('COMBINATION_ZONE_MATCH_BONUS', 0)
    
    # 高级模式匹配
    if ENABLE_ADVANCED_PATTERNS:
        # 密度模式匹配
        combo_density = calculate_density(reds)
        combo_density_cat = categorize_density(combo_density)
        if combo_density_cat == pattern_data.get('most_common_density_range'):
            bonus += weights_config.get('DENSITY_PATTERN_BONUS', 0)
        
        # 斜连模式匹配
        combo_diagonal = calculate_diagonal_patterns(reds)
        if combo_diagonal == pattern_data.get('most_common_diagonal_count'):
            bonus += weights_config.get('DIAGONAL_PATTERN_BONUS', 0)
        
        # 聚散模式匹配
        combo_clustering = calculate_clustering_score(reds)
        combo_clustering_cat = categorize_clustering(combo_clustering)
        if combo_clustering_cat == pattern_data.get('most_common_clustering_level'):
            bonus += weights_config.get('SHAPE_PATTERN_BONUS', 0)
    
    return bonus

def calculate_arm_bonus(combination: Dict, arm_rules: pd.DataFrame, weights_config: Dict) -> float:
    """计算关联规则匹配奖励分数。"""
    if arm_rules.empty:
        return 0.0
    
    bonus = 0.0
    combo_set = set(str(r) for r in combination['red'])
    
    for _, rule in arm_rules.iterrows():
        try:
            antecedent = set(rule['antecedents'])
            consequent = set(rule['consequents'])
            
            # 检查是否命中规则
            if antecedent.issubset(combo_set) and consequent.issubset(combo_set):
                rule_bonus = weights_config.get('ARM_COMBINATION_BONUS_WEIGHT', 0)
                rule_bonus *= (1 + rule['lift'] * weights_config.get('ARM_BONUS_LIFT_FACTOR', 0))
                rule_bonus *= (1 + rule['confidence'] * weights_config.get('ARM_BONUS_CONF_FACTOR', 0))
                bonus += rule_bonus
                
        except Exception:
            continue
    
    return bonus

# ==============================================================================
# --- 增强回测与优化模块 ---
# ==============================================================================

def run_enhanced_analysis_and_recommendation(df_hist: pd.DataFrame, ml_lags: List[int], 
                                           weights_config: Dict, arm_rules: pd.DataFrame) -> Tuple:
    """执行增强版分析和推荐流程。"""
    freq_data = analyze_frequency_omission(df_hist)
    patt_data = analyze_enhanced_patterns(df_hist)
    ml_models = train_enhanced_prediction_models(df_hist, ml_lags)
    
    probabilities = (predict_enhanced_probabilities(df_hist, ml_models, ml_lags) 
                    if ml_models else {'red': {}, 'blue': {}, 'red_neural': {}, 'blue_neural': {}, 'red_ensemble': {}})
    
    scores = calculate_enhanced_scores(freq_data, probabilities, weights_config)
    recs, rec_strings = generate_enhanced_combinations(scores, patt_data, arm_rules, weights_config)
    
    # 特征重要性分析
    feature_importance = {}
    if ml_models and ENABLE_FEATURE_IMPORTANCE:
        feature_importance = analyze_feature_importance(ml_models, ml_models.get('feature_cols', []))
    
    analysis_summary = {
        'frequency_omission': freq_data, 
        'patterns': patt_data,
        'feature_importance': feature_importance
    }
    
    return recs, rec_strings, analysis_summary, ml_models, scores

def run_enhanced_backtest(full_df: pd.DataFrame, ml_lags: List[int], weights_config: Dict, 
                         arm_rules: pd.DataFrame, num_periods: int) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """执行增强版策略回测。"""
    min_data_needed = (max(ml_lags) if ml_lags else 0) + MIN_POSITIVE_SAMPLES_FOR_ML + num_periods
    if len(full_df) < min_data_needed:
        logger.error(f"数据不足以回测{num_periods}期。需要至少{min_data_needed}期，当前有{len(full_df)}期。")
        return pd.DataFrame(), {}

    start_idx = len(full_df) - num_periods
    results = []
    prize_counts = Counter()
    red_cols = [f'red{i+1}' for i in range(6)]
    best_hits_per_period = []
    
    # 自动纠错机制
    correction_history = []
    
    logger.info("增强版策略回测已启动...")
    start_time = time.time()
    
    for i in range(num_periods):
        current_iter = i + 1
        current_idx = start_idx + i
        
        with SuppressOutput(suppress_stdout=True, capture_stderr=True):
            hist_data = full_df.iloc[:current_idx]
            
            # 应用自动纠错
            if ENABLE_AUTO_CORRECTION and correction_history:
                weights_config = apply_auto_correction(weights_config, correction_history)
            
            predicted_combos, _, _, _, _ = run_enhanced_analysis_and_recommendation(
                hist_data, ml_lags, weights_config, arm_rules
            )
        
        actual_outcome = full_df.loc[current_idx]
        actual_red_set = set(actual_outcome[red_cols])
        actual_blue = actual_outcome['blue']
        
        period_max_red_hits = 0
        period_blue_hit_on_max_red = False
        period_results = []
        
        if not predicted_combos:
            best_hits_per_period.append({
                'period': actual_outcome['期号'], 
                'best_red_hits': 0, 
                'blue_hit': False, 
                'prize': None
            })
        else:
            for combo_dict in predicted_combos:
                combo = combo_dict['combination']
                red_hits = len(set(combo['red']) & actual_red_set)
                blue_hit = combo['blue'] == actual_blue
                prize = get_prize_level(red_hits, blue_hit)
                
                if prize: 
                    prize_counts[prize] += 1
                
                result_record = {
                    'period': actual_outcome['期号'], 
                    'red_hits': red_hits, 
                    'blue_hit': blue_hit, 
                    'prize': prize,
                    'predicted_reds': combo['red'],
                    'predicted_blue': combo['blue'],
                    'actual_reds': list(actual_red_set),
                    'actual_blue': actual_blue
                }
                results.append(result_record)
                period_results.append(result_record)
                
                if red_hits > period_max_red_hits: 
                    period_max_red_hits = red_hits
                    period_blue_hit_on_max_red = blue_hit
                elif red_hits == period_max_red_hits and not period_blue_hit_on_max_red and blue_hit: 
                    period_blue_hit_on_max_red = True
            
            best_hits_per_period.append({
                'period': actual_outcome['期号'], 
                'best_red_hits': period_max_red_hits, 
                'blue_hit': period_blue_hit_on_max_red, 
                'prize': get_prize_level(period_max_red_hits, period_blue_hit_on_max_red)
            })
            
            # 记录纠错历史
            if ENABLE_AUTO_CORRECTION:
                correction_history.append({
                    'period': actual_outcome['期号'],
                    'predictions': period_results,
                    'best_red_hits': period_max_red_hits,
                    'blue_hit': period_blue_hit_on_max_red
                })
                
                # 只保留最近的纠错历史
                if len(correction_history) > 20:
                    correction_history.pop(0)

        # 打印进度
        if current_iter == 1 or current_iter % 10 == 0 or current_iter == num_periods:
            elapsed = time.time() - start_time
            avg_time = elapsed / current_iter
            remaining_time = avg_time * (num_periods - current_iter)
            progress_logger.info(
                f"增强回测进度: {current_iter}/{num_periods} | "
                f"平均耗时: {avg_time:.2f}s/期 | 预估剩余: {format_time(remaining_time)}"
            )
    
    return pd.DataFrame(results), {
        'prize_counts': dict(prize_counts), 
        'best_hits_per_period': pd.DataFrame(best_hits_per_period),
        'correction_history': correction_history
    }

def apply_auto_correction(weights_config: Dict, correction_history: List[Dict]) -> Dict:
    """基于历史表现自动调整权重配置。"""
    if not correction_history:
        return weights_config
    
    adjusted_weights = weights_config.copy()
    
    # 分析最近的表现
    recent_performance = correction_history[-10:]  # 最近10期
    avg_red_hits = np.mean([p['best_red_hits'] for p in recent_performance])
    blue_hit_rate = np.mean([p['blue_hit'] for p in recent_performance])
    
    # 根据表现调整权重
    if avg_red_hits < 2.0:  # 红球命中率低
        # 增加频率和遗漏权重，减少ML权重
        adjusted_weights['FREQ_SCORE_WEIGHT'] *= 1.1
        adjusted_weights['OMISSION_SCORE_WEIGHT'] *= 1.1
        adjusted_weights['ML_PROB_SCORE_WEIGHT_RED'] *= 0.9
    elif avg_red_hits > 3.5:  # 红球命中率高
        # 增加ML权重，减少传统权重
        adjusted_weights['ML_PROB_SCORE_WEIGHT_RED'] *= 1.1
        adjusted_weights['FREQ_SCORE_WEIGHT'] *= 0.95
    
    if blue_hit_rate < 0.3:  # 蓝球命中率低
        adjusted_weights['BLUE_FREQ_SCORE_WEIGHT'] *= 1.1
        adjusted_weights['ML_PROB_SCORE_WEIGHT_BLUE'] *= 0.9
    elif blue_hit_rate > 0.6:  # 蓝球命中率高
        adjusted_weights['ML_PROB_SCORE_WEIGHT_BLUE'] *= 1.1
    
    return adjusted_weights

# ==============================================================================
# --- Optuna 增强优化模块 ---
# ==============================================================================

def enhanced_objective(trial: optuna.trial.Trial, df_for_opt: pd.DataFrame, 
                      ml_lags: List[int], arm_rules: pd.DataFrame) -> float:
    """增强版Optuna目标函数。"""
    trial_weights = {}
    
    # 动态构建搜索空间
    for key, value in DEFAULT_WEIGHTS.items():
        if isinstance(value, int):
            if 'NUM_COMBINATIONS' in key: 
                trial_weights[key] = trial.suggest_int(key, 8, 16)
            elif 'TOP_N' in key: 
                trial_weights[key] = trial.suggest_int(key, 20, 35)
            else: 
                trial_weights[key] = trial.suggest_int(key, max(0, value - 3), value + 3)
        elif isinstance(value, float):
            if any(k in key for k in ['PERCENT', 'FACTOR', 'SUPPORT', 'CONFIDENCE']):
                trial_weights[key] = trial.suggest_float(key, value * 0.3, value * 1.8)
            else:
                trial_weights[key] = trial.suggest_float(key, value * 0.3, value * 2.5)

    full_trial_weights = DEFAULT_WEIGHTS.copy()
    full_trial_weights.update(trial_weights)
    
    # 在快速回测中评估
    with SuppressOutput():
        _, backtest_stats = run_enhanced_backtest(
            df_for_opt, ml_lags, full_trial_weights, arm_rules, OPTIMIZATION_BACKTEST_PERIODS
        )
    
    # 增强版评分函数
    prize_weights = {
        '一等奖': 10000, '二等奖': 2000, '三等奖': 500, 
        '四等奖': 50, '五等奖': 5, '六等奖': 1
    }
    
    base_score = sum(
        prize_weights.get(p, 0) * c 
        for p, c in backtest_stats.get('prize_counts', {}).items()
    )
    
    # 添加稳定性奖励
    best_hits_df = backtest_stats.get('best_hits_per_period')
    if best_hits_df is not None and not best_hits_df.empty:
        # 奖励稳定的命中率
        red_hits_std = best_hits_df['best_red_hits'].std()
        stability_bonus = max(0, 100 - red_hits_std * 20)  # 标准差越小奖励越高
        base_score += stability_bonus
    
    return base_score

def enhanced_optuna_progress_callback(study: optuna.study.Study, trial: optuna.trial.FrozenTrial, 
                                    total_trials: int):
    """增强版Optuna进度回调。"""
    global OPTUNA_START_TIME
    current_iter = trial.number + 1
    
    if current_iter == 1 or current_iter % 15 == 0 or current_iter == total_trials:
        elapsed = time.time() - OPTUNA_START_TIME
        avg_time = elapsed / current_iter
        remaining_time = avg_time * (total_trials - current_iter)
        best_value = f"{study.best_value:.2f}" if study.best_trial else "N/A"
        
        progress_logger.info(
            f"增强Optuna进度: {current_iter}/{total_trials} | "
            f"当前最佳得分: {best_value} | 预估剩余: {format_time(remaining_time)}"
        )

# ==============================================================================
# --- 主程序入口 ---
# ==============================================================================

if __name__ == "__main__":
    # 初始化日志系统
    log_filename = os.path.join(
        SCRIPT_DIR, 
        f"enhanced_ssq_analysis_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    )
    file_handler = logging.FileHandler(log_filename, 'w', 'utf-8')
    file_handler.setFormatter(detailed_formatter)
    logger.addHandler(file_handler)
    set_console_verbosity(logging.INFO, use_simple_formatter=True)

    logger.info("=== 增强版双色球数据分析与推荐系统 ===")
    logger.info("启动数据加载和预处理...")

    # 数据加载逻辑
    main_df = None
    if os.path.exists(PROCESSED_CSV_PATH):
        main_df = load_data(PROCESSED_CSV_PATH)
        if main_df is not None:
            logger.info("从缓存文件加载预处理数据成功。")

    if main_df is None or main_df.empty:
        logger.info("未找到或无法加载缓存数据，正在从原始文件生成...")
        raw_df = load_data(CSV_FILE_PATH)
        if raw_df is not None and not raw_df.empty:
            logger.info("原始数据加载成功，开始清洗...")
            cleaned_df = clean_and_structure(raw_df)
            if cleaned_df is not None and not cleaned_df.empty:
                logger.info("数据清洗成功，开始增强特征工程...")
                main_df = enhanced_feature_engineer(cleaned_df)
                if main_df is not None and not main_df.empty:
                    logger.info("增强特征工程成功，保存预处理数据...")
                    try:
                        main_df.to_csv(PROCESSED_CSV_PATH, index=False)
                        logger.info(f"预处理数据已保存到: {PROCESSED_CSV_PATH}")
                    except IOError as e:
                        logger.error(f"保存预处理数据失败: {e}")
                else:
                    logger.error("增强特征工程失败。")
            else:
                logger.error("数据清洗失败。")
        else:
            logger.error("原始数据加载失败。")
    
    if main_df is None or main_df.empty:
        logger.critical("数据准备失败，无法继续。程序终止。")
        sys.exit(1)
    
    logger.info(f"数据加载完成，共 {len(main_df)} 期有效数据。")
    last_period = main_df['期号'].iloc[-1]

    # 执行优化或直接分析
    active_weights = DEFAULT_WEIGHTS.copy()
    optuna_summary = None

    if ENABLE_OPTUNA_OPTIMIZATION:
        logger.info("\n" + "="*30 + " 增强Optuna优化模式 " + "="*30)
        set_console_verbosity(logging.INFO, use_simple_formatter=False)
        
        optuna_arm_rules = analyze_associations(main_df, DEFAULT_WEIGHTS)
        
        study = optuna.create_study(direction="maximize")
        global OPTUNA_START_TIME
        OPTUNA_START_TIME = time.time()
        progress_callback_with_total = partial(
            enhanced_optuna_progress_callback, total_trials=OPTIMIZATION_TRIALS
        )
        
        try:
            study.optimize(
                lambda t: enhanced_objective(t, main_df, ML_LAG_FEATURES, optuna_arm_rules), 
                n_trials=OPTIMIZATION_TRIALS, 
                callbacks=[progress_callback_with_total]
            )
            logger.info("增强Optuna优化完成。")
            active_weights.update(study.best_params)
            optuna_summary = {
                "status": "完成", 
                "best_value": study.best_value, 
                "best_params": study.best_params
            }
        except Exception as e:
            logger.error(f"增强Optuna优化过程中断: {e}", exc_info=True)
            optuna_summary = {"status": "中断", "error": str(e)}
            logger.warning("优化中断，将使用默认权重继续分析。")
    
    # 切换到报告模式
    report_formatter = logging.Formatter('%(message)s')
    file_handler.setFormatter(report_formatter)
    global_console_handler.setFormatter(report_formatter)
    
    logger.info("\n\n" + "="*70)
    logger.info(f"{' ' * 20}增强版双色球策略分析报告")
    logger.info("="*70)
    logger.info(f"报告生成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"分析基于数据: 截至 {last_period} 期 (共 {len(main_df)} 期)")
    logger.info(f"本次预测目标: 第 {last_period + 1} 期")
    logger.info(f"增强功能状态:")
    logger.info(f"  - 神经网络: {'启用' if ENABLE_NEURAL_NETWORK else '禁用'}")
    logger.info(f"  - 集成模型: {'启用' if ENABLE_ENSEMBLE_MODELS else '禁用'}")
    logger.info(f"  - 特征重要性: {'启用' if ENABLE_FEATURE_IMPORTANCE else '禁用'}")
    logger.info(f"  - 高级模式: {'启用' if ENABLE_ADVANCED_PATTERNS else '禁用'}")
    logger.info(f"  - 自动纠错: {'启用' if ENABLE_AUTO_CORRECTION else '禁用'}")
    logger.info(f"日志文件: {os.path.basename(log_filename)}")

    # 打印优化摘要
    if ENABLE_OPTUNA_OPTIMIZATION and optuna_summary:
        logger.info("\n" + "="*30 + " 增强Optuna优化摘要 " + "="*30)
        logger.info(f"优化状态: {optuna_summary['status']}")
        if optuna_summary['status'] == '完成':
            logger.info(f"最佳性能得分: {optuna_summary['best_value']:.4f}")
            logger.info("--- 本次分析已采用以下优化参数 ---")
            best_params_str = json.dumps(optuna_summary['best_params'], indent=2, ensure_ascii=False)
            logger.info(best_params_str)
        else: 
            logger.info(f"错误信息: {optuna_summary['error']}")
    else:
        logger.info("\n--- 本次分析使用脚本内置的默认权重 ---")

    # 全局分析
    full_history_arm_rules = analyze_associations(main_df, active_weights)
    
    # 增强回测
    logger.info("\n" + "="*30 + " 增强策略回测摘要 " + "="*30)
    backtest_results_df, backtest_stats = run_enhanced_backtest(
        main_df, ML_LAG_FEATURES, active_weights, full_history_arm_rules, BACKTEST_PERIODS_COUNT
    )
    
    if not backtest_results_df.empty:
        num_periods_tested = len(backtest_results_df['period'].unique())
        num_combos_per_period = active_weights.get('NUM_COMBINATIONS_TO_GENERATE', 12)
        total_bets = len(backtest_results_df)
        
        logger.info(f"回测周期: 最近 {num_periods_tested} 期 | 每期注数: {num_combos_per_period} | 总投入注数: {total_bets}")
        
        logger.info("\n--- 1. 奖金与回报分析 ---")
        prize_dist = backtest_stats.get('prize_counts', {})
        prize_values = {
            '一等奖': 5e6, '二等奖': 1.5e5, '三等奖': 3e3, 
            '四等奖': 200, '五等奖': 10, '六等奖': 5
        }
        total_revenue = sum(prize_values.get(p, 0) * c for p, c in prize_dist.items())
        total_cost = total_bets * 2
        roi = (total_revenue - total_cost) * 100 / total_cost if total_cost > 0 else 0
        
        logger.info(f"  - 估算总回报: {total_revenue:,.2f} 元 (总成本: {total_cost:,.2f} 元)")
        logger.info(f"  - 投资回报率 (ROI): {roi:.2f}%")
        logger.info("  - 中奖等级分布 (总计):")
        
        if prize_dist:
            for prize in prize_values.keys():
                if prize in prize_dist: 
                    logger.info(f"    - {prize:<4s}: {prize_dist[prize]:>4d} 次")
        else: 
            logger.info("    - 未命中任何奖级。")
        
        logger.info("\n--- 2. 增强性能指标 ---")
        logger.info(f"  - 平均红球命中 (每注): {backtest_results_df['red_hits'].mean():.3f} / 6")
        logger.info(f"  - 蓝球命中率 (每注): {backtest_results_df['blue_hit'].mean() * 100:.2f}%")
        logger.info(f"  - 红球命中标准差: {backtest_results_df['red_hits'].std():.3f} (越小越稳定)")
        
        # 命中分布分析
        red_hits_dist = backtest_results_df['red_hits'].value_counts().sort_index()
        logger.info("  - 红球命中分布:")
        for hits, count in red_hits_dist.items():
            percentage = count / len(backtest_results_df) * 100
            logger.info(f"    - {hits}个红球: {count} 次 ({percentage:.1f}%)")
        
        logger.info("\n--- 3. 每期最佳命中表现 ---")
        best_hits_df = backtest_stats.get('best_hits_per_period')
        if best_hits_df is not None and not best_hits_df.empty:
            logger.info("  - 在一期内至少命中:")
            
            queries_and_names = [
                ("(`best_red_hits` == 4 and `blue_hit`) or (`best_red_hits` == 5 and not `blue_hit`)", "四等奖(4+1或5+0)"),
                ("`best_red_hits` == 5 and `blue_hit`", "三等奖(5+1)"),
                ("`best_red_hits` == 6", "二等奖/一等奖")
            ]
            
            for query, name in queries_and_names:
                try:
                    count = best_hits_df.query(query).shape[0] if not best_hits_df.empty else 0
                    logger.info(f"    - {name:<18s}: {count} / {num_periods_tested} 期")
                except Exception:
                    logger.info(f"    - {name:<18s}: 0 / {num_periods_tested} 期")
            
            any_blue_hit_periods = best_hits_df['blue_hit'].sum()
            blue_coverage = any_blue_hit_periods / num_periods_tested if num_periods_tested > 0 else 0
            logger.info(f"  - 蓝球覆盖率: 在 {blue_coverage:.2%} 的期数中，推荐组合至少有一注命中蓝球")
        
        # 自动纠错分析
        if ENABLE_AUTO_CORRECTION and 'correction_history' in backtest_stats:
            logger.info("\n--- 4. 自动纠错机制分析 ---")
            correction_history = backtest_stats['correction_history']
            if correction_history:
                recent_corrections = correction_history[-10:]
                avg_performance = np.mean([c['best_red_hits'] for c in recent_corrections])
                logger.info(f"  - 最近10期平均红球命中: {avg_performance:.2f}")
                logger.info("  - 纠错机制已根据历史表现动态调整权重")
    else: 
        logger.warning("增强回测未产生有效结果，可能是数据量不足。")
    
    # 最终推荐
    logger.info("\n" + "="*30 + f" 第 {last_period + 1} 期 增强推荐 " + "="*30)
    
    final_recs, final_rec_strings, analysis_summary, final_models, final_scores = run_enhanced_analysis_and_recommendation(
        main_df, ML_LAG_FEATURES, active_weights, full_history_arm_rules
    )
    
    logger.info("\n--- 增强单式推荐 ---")
    for line in final_rec_strings: 
        logger.info(line)
    
    # 特征重要性报告
    if ENABLE_FEATURE_IMPORTANCE and 'feature_importance' in analysis_summary:
        feature_importance = analysis_summary['feature_importance']
        if 'top_features' in feature_importance:
            logger.info("\n--- 特征重要性分析 (Top 10) ---")
            for i, (feature, importance) in enumerate(feature_importance['top_features'], 1):
                logger.info(f"  {i:2d}. {feature:<30s}: {importance:.4f}")
    
    logger.info("\n--- 增强复式参考 ---")
    if final_scores and final_scores.get('red_scores'):
        top_10_red = sorted([
            n for n, _ in sorted(final_scores['red_scores'].items(), key=lambda x: x[1], reverse=True)[:10]
        ])
        top_10_blue = sorted([
            n for n, _ in sorted(final_scores['blue_scores'].items(), key=lambda x: x[1], reverse=True)[:10]
        ])
        logger.info(f"  红球 (Top 10): {' '.join(f'{n:02d}' for n in top_10_red)}")
        logger.info(f"  蓝球 (Top 10): {' '.join(f'{n:02d}' for n in top_10_blue)}")
    
    # 高级分析摘要
    logger.info("\n--- 高级分析摘要 ---")
    patterns = analysis_summary.get('patterns', {})
    if patterns:
        logger.info("  - 历史模式分析:")
        for key, value in patterns.items():
            if value is not None:
                logger.info(f"    - {key}: {value}")
    
    logger.info("\n" + "="*70)
    logger.info(f"--- 增强版报告结束 (详情请查阅: {os.path.basename(log_filename)}) ---")
    logger.info("="*70)

print("增强版双色球分析系统执行完成！")
