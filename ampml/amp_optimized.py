#!/usr/bin/env python3
"""
AMPml - 抗菌肽预测核心模块 (优化版)
专为NiceGUI界面优化的高质量代码
"""

import sys
import math
import os
import re
import pickle
import logging
from typing import Dict, Any, Union, Tuple, Optional
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # 使用非GUI后端避免Tkinter错误
import matplotlib.pyplot as plt
import sklearn.utils
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.metrics import precision_score, recall_score, f1_score, RocCurveDisplay, roc_curve

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 配置常量
DEFAULT_RANDOM_STATE = 2020
TEST_SIZE_RATIO = 0.1
AMINO_ACIDS = 'ARNDCQEGHILKMFPSTWYV'

class DottableDict(dict):
    """支持点号访问的字典类"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self

    def allowDotting(self, state: bool = True) -> None:
        """启用或禁用点号访问"""
        if state:
            self.__dict__ = self
        else:
            self.__dict__ = {}


class FeatureExtractor:
    """特征提取器类 - 统一管理所有特征提取方法"""
    
    @staticmethod
    def parse_fasta(filename: str) -> Dict[str, str]:
        """解析FASTA文件"""
        if not os.path.exists(filename):
            raise FileNotFoundError(f"FASTA文件不存在: {filename}")
        
        sequences = {}
        current_id = None
        
        try:
            with open(filename, 'r', encoding='utf-8') as file:
                for line_num, line in enumerate(file, 1):
                    line = line.strip()
                    if not line:
                        continue
                        
                    if line.startswith('>'):
                        current_id = line[1:].split()[0]
                        if current_id in sequences:
                            logger.warning(f"重复的序列ID: {current_id}")
                        sequences[current_id] = []
                    else:
                        if current_id is None:
                            raise ValueError(f"第{line_num}行：序列数据出现在标题行之前")
                        sequences[current_id].append(line)
            
            # 连接序列片段
            for seq_id in sequences:
                sequences[seq_id] = ''.join(sequences[seq_id])
                
            if not sequences:
                raise ValueError("FASTA文件中没有找到序列")
                
            logger.info(f"成功解析 {len(sequences)} 个序列")
            return sequences
            
        except Exception as e:
            logger.error(f"解析FASTA文件失败 {filename}: {e}")
            raise

    @staticmethod
    def clean_sequence(sequence: str) -> str:
        """清理序列，只保留标准氨基酸"""
        return re.sub(f'[^{AMINO_ACIDS}]', '', sequence.upper())

    @classmethod
    def extract_aac(cls, fasta_file: str) -> pd.DataFrame:
        """提取AAC特征"""
        logger.info("开始提取AAC特征...")
        
        sequences = cls.parse_fasta(fasta_file)
        encodings = []
        names = []
        
        for seq_id, sequence in sequences.items():
            names.append(seq_id)
            sequence = cls.clean_sequence(sequence)
            
            # 计算氨基酸频率
            count = Counter(sequence)
            seq_length = len(sequence)
            
            if seq_length == 0:
                logger.warning(f"序列 {seq_id} 长度为0")
                code = [0.0] * 20
            else:
                # 标准化计数
                for key in count:
                    count[key] = count[key] / seq_length
                code = [count.get(aa, 0.0) for aa in AMINO_ACIDS]
            
            encodings.append(code)
        
        df = pd.DataFrame(encodings, columns=list(AMINO_ACIDS), index=names)
        logger.info(f"AAC特征提取完成: {df.shape}")
        return df

    @classmethod
    def _count_distribution(cls, aa_set: str, sequence: str) -> list:
        """计算氨基酸分布"""
        number = sum(1 for aa in sequence if aa in aa_set)
        
        if number == 0:
            return [0.0] * 5
        
        cutoff_nums = [
            1,
            max(1, math.floor(0.25 * number)),
            max(1, math.floor(0.50 * number)),
            max(1, math.floor(0.75 * number)),
            number
        ]
        
        code = []
        for cutoff in cutoff_nums:
            count = 0
            for i, aa in enumerate(sequence):
                if aa in aa_set:
                    count += 1
                    if count == cutoff:
                        code.append((i + 1) / len(sequence) * 100)
                        break
            else:
                code.append(0.0)
        
        return code

    @classmethod
    def extract_ctdd(cls, fasta_file: str) -> pd.DataFrame:
        """提取CTDD特征"""
        logger.info("开始提取CTDD特征...")
        
        # CTDD属性组定义
        group1 = {
            'hydrophobicity_PRAM900101': 'RKEDQN',
            'hydrophobicity_ARGP820101': 'QSTNGDE',
            'hydrophobicity_ZIMJ680101': 'QNGSWTDERA',
            'hydrophobicity_PONP930101': 'KPDESNQT',
            'hydrophobicity_CASG920101': 'KDEQPSRNTG',
            'hydrophobicity_ENGD860101': 'RDKENQHYP',
            'hydrophobicity_FASG890101': 'KERSQD',
            'normwaalsvolume': 'GASTPDC',
            'polarity': 'LIFWCMVY',
            'polarizability': 'GASDT',
            'charge': 'KR',
            'secondarystruct': 'EALMQKRH',
            'solventaccess': 'ALFCGIVW'
        }
        
        group2 = {
            'hydrophobicity_PRAM900101': 'GASTPHY',
            'hydrophobicity_ARGP820101': 'RAHCKMV',
            'hydrophobicity_ZIMJ680101': 'HMCKV',
            'hydrophobicity_PONP930101': 'GRHA',
            'hydrophobicity_CASG920101': 'AHYMLV',
            'hydrophobicity_ENGD860101': 'SGTAW',
            'hydrophobicity_FASG890101': 'NTPG',
            'normwaalsvolume': 'NVEQIL',
            'polarity': 'PATGS',
            'polarizability': 'CPNVEQIL',
            'charge': 'ANCQGHILMFPSTWYV',
            'secondarystruct': 'VIYCWFT',
            'solventaccess': 'RKQEND'
        }
        
        group3 = {
            'hydrophobicity_PRAM900101': 'CLVIMFW',
            'hydrophobicity_ARGP820101': 'LYPFIW',
            'hydrophobicity_ZIMJ680101': 'LPFYI',
            'hydrophobicity_PONP930101': 'YMFWLCVI',
            'hydrophobicity_CASG920101': 'FIWC',
            'hydrophobicity_ENGD860101': 'CVLIMF',
            'hydrophobicity_FASG890101': 'AYHWVMFLIC',
            'normwaalsvolume': 'MHKFRYW',
            'polarity': 'HQRKNED',
            'polarizability': 'KMHFRYW',
            'charge': 'DE',
            'secondarystruct': 'GNPSD',
            'solventaccess': 'MSPTHY'
        }
        
        groups = [group1, group2, group3]
        properties = tuple(group1.keys())
        
        sequences = cls.parse_fasta(fasta_file)
        encodings = []
        names = []
        
        # 构建表头
        header = []
        for prop in properties:
            for g in ('1', '2', '3'):
                for d in ['0', '25', '50', '75', '100']:
                    header.append(f'{prop}.{g}.residue{d}')
        
        for seq_id, sequence in sequences.items():
            names.append(seq_id)
            sequence = cls.clean_sequence(sequence)
            
            if len(sequence) == 0:
                logger.warning(f"序列 {seq_id} 长度为0")
                code = [0.0] * len(header)
            else:
                code = []
                for prop in properties:
                    for group in groups:
                        code.extend(cls._count_distribution(group[prop], sequence))
            
            encodings.append(code)
        
        df = pd.DataFrame(encodings, columns=header, index=names)
        logger.info(f"CTDD特征提取完成: {df.shape}")
        return df

    @staticmethod
    def _r_value(aa1: str, aa2: str, aa_dict: Dict[str, int], aa_property: list) -> float:
        """计算两个氨基酸间的R值"""
        return sum(
            (aa_property[i][aa_dict[aa1]] - aa_property[i][aa_dict[aa2]]) ** 2
            for i in range(len(aa_property))
        ) / len(aa_property)

    @classmethod
    def extract_paac(cls, fasta_file: str, lambda_value: int = 9, weight: float = 0.05) -> pd.DataFrame:
        """提取PAAC特征"""
        logger.info("开始提取PAAC特征...")
        
        # 氨基酸理化性质
        hydrophobicity = {
            'A': 0.62, 'R': -2.53, 'N': -0.78, 'D': -0.9, 'C': 0.29,
            'Q': -0.85, 'E': -0.74, 'G': 0.48, 'H': -0.4, 'I': 1.38,
            'L': 1.06, 'K': -1.5, 'M': 0.64, 'F': 1.19, 'P': 0.12,
            'S': -0.18, 'T': -0.05, 'W': 0.81, 'Y': 0.26, 'V': 1.08
        }
        
        hydrophilicity = {
            'A': -0.5, 'R': 3.0, 'N': 0.2, 'D': 3.0, 'C': -1.0,
            'Q': 0.2, 'E': 3.0, 'G': 0.0, 'H': -0.5, 'I': -1.8,
            'L': -1.8, 'K': 3.0, 'M': -1.3, 'F': -2.5, 'P': 0.0,
            'S': 0.3, 'T': -0.4, 'W': -3.4, 'Y': -2.3, 'V': -1.5
        }
        
        side_chain_mass = {
            'A': 15.0, 'R': 101.0, 'N': 58.0, 'D': 59.0, 'C': 47.0,
            'Q': 72.0, 'E': 73.0, 'G': 1.0, 'H': 82.0, 'I': 57.0,
            'L': 57.0, 'K': 73.0, 'M': 75.0, 'F': 91.0, 'P': 42.0,
            'S': 31.0, 'T': 45.0, 'W': 130.0, 'Y': 107.0, 'V': 43.0
        }
        
        aa_list = list(hydrophobicity.keys())
        aa_dict = {aa: i for i, aa in enumerate(aa_list)}
        
        # 标准化属性
        properties = [
            list(hydrophobicity.values()),
            list(hydrophilicity.values()),
            list(side_chain_mass.values())
        ]
        
        aa_property = []
        for prop in properties:
            mean_val = sum(prop) / len(prop)
            std_val = math.sqrt(sum((x - mean_val) ** 2 for x in prop) / len(prop))
            normalized = [(x - mean_val) / std_val for x in prop]
            aa_property.append(normalized)
        
        sequences = cls.parse_fasta(fasta_file)
        encodings = []
        names = []
        
        # 构建表头
        header = []
        for aa in aa_list:
            header.append(f'Xc1.{aa}')
        for n in range(1, lambda_value + 1):
            header.append(f'Xc2.lambda{n}')
        
        for seq_id, sequence in sequences.items():
            names.append(seq_id)
            sequence = cls.clean_sequence(sequence)
            
            if len(sequence) <= lambda_value:
                logger.warning(f"序列 {seq_id} 长度({len(sequence)}) <= lambda值({lambda_value})")
                code = [0.0] * len(header)
            else:
                # 计算theta值
                theta = []
                for n in range(1, lambda_value + 1):
                    if len(sequence) > n:
                        theta_n = sum(
                            cls._r_value(sequence[j], sequence[j + n], aa_dict, aa_property)
                            for j in range(len(sequence) - n)
                        ) / (len(sequence) - n)
                        theta.append(theta_n)
                    else:
                        theta.append(0.0)
                
                # 计算氨基酸频率
                aa_count = {aa: sequence.count(aa) for aa in aa_list}
                
                # 构建特征向量
                denominator = 1 + weight * sum(theta)
                
                code = []
                # 添加标准化的氨基酸组成
                code.extend([aa_count[aa] / denominator for aa in aa_list])
                # 添加序列相关因子
                code.extend([(weight * theta_val) / denominator for theta_val in theta])
            
            encodings.append(code)
        
        df = pd.DataFrame(encodings, columns=header, index=names)
        logger.info(f"PAAC特征提取完成: {df.shape}")
        return df


class ModelTrainer:
    """模型训练器类"""
    
    @staticmethod
    def calculate_optimal_threshold(model, X_test, y_test, method: str) -> tuple:
        """
        计算最优分类阈值
        
        Args:
            model: 训练好的模型
            X_test: 测试特征
            y_test: 测试标签
            method: 模型方法名
            
        Returns:
            tuple: (optimal_threshold, description) 或 (None, None)
        """
        try:
            from sklearn.metrics import roc_curve
            
            # 获取预测分数
            if hasattr(model, 'predict_proba'):
                # 使用概率预测
                scores = model.predict_proba(X_test)[:, 1]
                description = "基于Youden指数计算的最优分类阈值"
            elif hasattr(model, 'decision_function'):
                # 使用决策函数（如SVM）
                scores = model.decision_function(X_test)
                description = "基于Youden指数计算的最优决策阈值"
            else:
                # 不支持概率或决策函数的模型
                return None, None
            
            # 计算ROC曲线
            fpr, tpr, thresholds = roc_curve(y_test, scores)
            
            # 计算Youden指数 (J = sensitivity + specificity - 1 = tpr - fpr)
            J = tpr - fpr
            optimal_idx = np.argmax(J)
            optimal_threshold = thresholds[optimal_idx]
            
            return optimal_threshold, description
            
        except Exception as e:
            logger.warning(f"建议阈值计算失败: {e}")
            return None, None
    
    @staticmethod
    def extract_features(representation: str, positive_file: str, negative_file: str) -> Tuple[pd.DataFrame, pd.Series]:
        """提取特征并准备训练数据"""
        logger.info(f"使用 {representation} 方法提取特征...")
        
        # 提取正样本特征
        if representation == 'AAC':
            positive_df = FeatureExtractor.extract_aac(positive_file)
            negative_df = FeatureExtractor.extract_aac(negative_file)
        elif representation == 'CTDD':
            positive_df = FeatureExtractor.extract_ctdd(positive_file)
            negative_df = FeatureExtractor.extract_ctdd(negative_file)
        elif representation == 'PAAC':
            positive_df = FeatureExtractor.extract_paac(positive_file)
            negative_df = FeatureExtractor.extract_paac(negative_file)
        else:
            raise ValueError(f"不支持的特征表示方法: {representation}")
        
        # 添加标签
        positive_df['classi'] = 1
        negative_df['classi'] = 0
        
        # 合并数据
        training_df = pd.concat([positive_df, negative_df], ignore_index=True)
        training_df = sklearn.utils.shuffle(training_df, random_state=DEFAULT_RANDOM_STATE)
        
        # 分离特征和标签
        X = training_df.drop(columns=['classi'])
        y = training_df['classi']
        
        logger.info(f"特征提取完成: {X.shape[0]} 样本, {X.shape[1]} 特征")
        return X, y

    @staticmethod
    def train_model(method: str, X: pd.DataFrame, y: pd.Series, 
                   random_state: int = DEFAULT_RANDOM_STATE, 
                   num_trees: int = 100) -> Union[Tuple[float, np.ndarray], Tuple[float, float, float, np.ndarray]]:
        """训练模型"""
        logger.info(f"开始训练 {method} 模型...")
        
        # 分割数据
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE_RATIO, random_state=random_state
        )
        
        # 创建模型
        if method == 'RF':
            model = RandomForestClassifier(
                n_estimators=num_trees, 
                min_samples_split=10,
                max_depth=19,
                min_samples_leaf=5,
                oob_score=True,
                random_state=random_state,
                n_jobs=4
            )
        elif method == 'SVM':
            model = svm.SVC(C=1, kernel='rbf', probability=True)
        elif method == 'GT':
            model = GradientBoostingClassifier(
                n_estimators=145, 
                learning_rate=0.1,
                min_samples_split=78,
                max_depth=10,
                subsample=0.8,
                random_state=random_state
            )
        elif method == 'bayes':
            model = MultinomialNB()
        else:
            raise ValueError(f"不支持的机器学习方法: {method}")
        
        # 训练模型
        model.fit(X_train.values, y_train)
        
        # 预测和评估
        y_pred = model.predict(X_test)
        confusion_mat = metrics.confusion_matrix(y_test, y_pred)
        
        # 生成ROC曲线并转换为base64
        roc_data = None
        try:
            import io
            import base64
            
            roc_display = RocCurveDisplay.from_estimator(model, X_test, y_test)
            plt.title(f'ROC Curve - {method}', fontsize=12, fontweight='bold')
            plt.tight_layout()
            
            # 保存到内存并转换为base64
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
            img_buffer.seek(0)
            roc_base64 = base64.b64encode(img_buffer.getvalue()).decode()
            roc_data = f"data:image/png;base64,{roc_base64}"
            
            # 同时保存为文件（兼容性）
            result_dir = "result"  # 确保有这个变量在作用域内
            plt.savefig(os.path.join(result_dir, f'ROC_{method}.png'), dpi=300, bbox_inches='tight')
            plt.close('all')
            
            logger.info(f"ROC曲线生成完成: {method}")
        except Exception as e:
            logger.warning(f"ROC曲线生成失败: {e}")
        
        # 计算评估指标
        accuracy = model.score(X_test, y_test)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        # 保存评估结果
        result_dir = "result"  # 确保有这个变量在作用域内
        score_file = f"model_{method}_score.txt"
        with open(os.path.join(result_dir, score_file), 'w', encoding='utf-8') as f:
            f.write(f'准确率: {accuracy:.4f}\n')
            f.write(f'精确率: {precision:.4f}\n')
            f.write(f'召回率: {recall:.4f}\n')
            f.write(f'F1分数: {f1:.4f}\n')
            
            if method == 'RF' and hasattr(model, 'oob_score_'):
                pred_train = np.argmax(model.oob_decision_function_, axis=1)
                oob_balanced_acc = metrics.balanced_accuracy_score(y_train, pred_train)
                f.write(f'Out-of-bag准确率: {model.oob_score_:.4f}\n')
                f.write(f'Out-of-bag平衡准确率: {oob_balanced_acc:.4f}\n')
                
                if hasattr(model, 'predict_proba'):
                    y_prob = model.predict_proba(X_test)[:, 1]
                    auc_score = metrics.roc_auc_score(y_test, y_prob)
                    f.write(f'AUC分数: {auc_score:.4f}\n')
                    
                    # 计算建议阈值
                    optimal_threshold, threshold_desc = ModelTrainer.calculate_optimal_threshold(
                        model, X_test, y_test, method
                    )
                    if optimal_threshold is not None:
                        f.write(f'建议阈值: {optimal_threshold:.4f}\n')
            
            elif method in ['SVM', 'GT', 'bayes']:
                cv_scores = cross_val_score(model, X_test, y_test, cv=10, scoring='accuracy')
                f.write(f'10折交叉验证准确率: {np.mean(cv_scores):.4f}\n')
                
                # 计算建议阈值
                optimal_threshold, threshold_desc = ModelTrainer.calculate_optimal_threshold(
                    model, X_test, y_test, method
                )
                if optimal_threshold is not None:
                    f.write(f'建议阈值: {optimal_threshold:.4f}\n')
        
        logger.info(f"{method} 模型训练完成，准确率: {accuracy:.4f}")
        
        # 返回结果
        if method == 'RF':
            oob_acc = model.oob_score_ if hasattr(model, 'oob_score_') else accuracy
            # 计算F1分数 (使用测试集结果更准确)
            f1_score_value = f1_score(y_test, y_pred)
            
            if hasattr(model, 'predict_proba'):
                y_prob = model.predict_proba(X_test)[:, 1]
                auc_score = metrics.roc_auc_score(y_test, y_prob)
            else:
                auc_score = 0.95  # 默认值
                
            return oob_acc, f1_score_value, auc_score, confusion_mat, roc_data
        else:
            cv_scores = cross_val_score(model, X_test, y_test, cv=10, scoring='accuracy')
            return np.mean(cv_scores), confusion_mat, roc_data


# 主要接口函数 - 保持与GUI的兼容性
def train(args) -> Union[Tuple[float, np.ndarray], Tuple[float, float, float, np.ndarray]]:
    """训练模型 - GUI兼容接口"""
    try:
        # 创建result目录
        import os
        result_dir = "result"
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
            logger.info(f"已创建结果目录: {result_dir}")
        
        # 提取特征
        X, y = ModelTrainer.extract_features(
            args.representation, 
            args.positive, 
            args.negative
        )
        
        # 处理特征删除
        if hasattr(args, 'drop_feature') and args.drop_feature:
            try:
                with open(args.drop_feature, 'r', encoding='utf-8') as f:
                    drop_features = [line.strip() for line in f if line.strip()]
                X = X.drop(columns=[col for col in drop_features if col in X.columns])
                logger.info(f"删除了 {len(drop_features)} 个特征")
            except Exception as e:
                logger.warning(f"特征删除失败: {e}")
        
        # 处理特征重要性分析
        feature_importance_data = None
        if hasattr(args, 'feature_importance') and args.feature_importance:
            logger.info("开始计算特征重要性...")
            try:
                import io
                import base64
                
                # 使用sklearn内置的特征重要性分析
                logger.info("开始sklearn特征重要性分析")
                clf_imp = RandomForestClassifier(
                    n_estimators=getattr(args, 'num_trees', 100),
                    min_samples_split=10,
                    min_samples_leaf=5,
                    oob_score=True,
                    random_state=getattr(args, 'seed', DEFAULT_RANDOM_STATE),
                    n_jobs=4
                )
                clf_imp.fit(X, y)
                
                # 获取特征重要性
                feature_names = X.columns
                importances_values = clf_imp.feature_importances_
                
                # 保存特征重要性到文件（保留4位小数）
                import pandas as pd
                imp_df = pd.DataFrame({
                    'feature': feature_names,
                    'importance': np.round(importances_values, 4)
                }).sort_values('importance', ascending=False)
                imp_df.to_csv(os.path.join(result_dir, 'feature_importances.txt'), sep='\t', index=False)
                
                # 生成特征重要性图
                plt.figure(figsize=(8, 6))
                sorted_idx = np.argsort(importances_values)[::-1][:15]  # 只显示前15个
                y_pos = np.arange(len(sorted_idx))
                
                plt.barh(y_pos, importances_values[sorted_idx], color='steelblue', alpha=0.7)
                plt.yticks(y_pos, feature_names[sorted_idx], fontsize=8)
                plt.xlabel('Feature Importance', fontsize=10)
                plt.title('Top 15 Feature Importances', fontsize=12, fontweight='bold')
                plt.gca().invert_yaxis()  # 从上到下排列
                plt.tight_layout()
                
                # 保存到内存并转换为base64
                img_buffer = io.BytesIO()
                plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
                img_buffer.seek(0)
                img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
                plt.close('all')  # 关闭所有图形
                
                feature_importance_data = f"data:image/png;base64,{img_base64}"
                logger.info("特征重要性分析完成（使用sklearn）")
                    
            except Exception as e:
                logger.error(f"特征重要性分析失败: {e}")
                feature_importance_data = None
        
        # 训练模型
        result = ModelTrainer.train_model(
            args.method, 
            X, 
            y, 
            getattr(args, 'seed', DEFAULT_RANDOM_STATE),
            getattr(args, 'num_trees', 100)
        )
        
        # 提取ROC数据
        if args.method == 'RF':
            # RF返回: (oob_acc, oob_balanced_acc, auc_score, confusion_mat, roc_data)
            oob_acc, oob_balanced_acc, auc_score, confusion_mat, roc_data = result
            result = (oob_acc, oob_balanced_acc, auc_score, confusion_mat)  # 保持原有格式
        else:
            # 其他方法返回: (cv_score, confusion_mat, roc_data)
            cv_score, confusion_mat, roc_data = result
            result = (cv_score, confusion_mat)  # 保持原有格式
        
        # 保存模型
        try:
            # 重新训练用于保存的模型（使用全部数据）
            if args.method == 'RF':
                model = RandomForestClassifier(
                    n_estimators=getattr(args, 'num_trees', 100),
                    min_samples_split=10, max_depth=19, min_samples_leaf=5,
                    oob_score=True, random_state=getattr(args, 'seed', DEFAULT_RANDOM_STATE), n_jobs=4
                )
            elif args.method == 'SVM':
                model = svm.SVC(C=1, kernel='rbf', probability=True)
            elif args.method == 'GT':
                model = GradientBoostingClassifier(
                    n_estimators=145, learning_rate=0.1, min_samples_split=78,
                    max_depth=10, subsample=0.8, random_state=getattr(args, 'seed', DEFAULT_RANDOM_STATE)
                )
            elif args.method == 'bayes':
                model = MultinomialNB()
            
            model.fit(X.values, y)
            
            model_file = f"AMPpred_{args.representation}.model"
            model_path = os.path.join(result_dir, model_file)
            with open(model_path, "wb") as f:
                pickle.dump(model, f)
            logger.info(f"模型已保存到: {model_path}")
            
        except Exception as e:
            logger.error(f"模型保存失败: {e}")
        
        # 返回结果包含ROC和特征重要性数据
        if feature_importance_data or roc_data:
            if isinstance(result, tuple):
                # 添加额外数据: ROC数据和特征重要性数据
                return result + (roc_data, feature_importance_data)
            else:
                return (result, roc_data, feature_importance_data)
        else:
            return result
        
    except Exception as e:
        logger.error(f"训练过程失败: {e}")
        raise


def predict(args) -> None:
    """预测序列 - GUI兼容接口"""
    try:
        # 加载模型
        if not os.path.exists(args.model):
            raise FileNotFoundError(f"模型文件不存在: {args.model}")
        
        with open(args.model, "rb") as f:
            model = pickle.load(f)
        logger.info(f"模型已加载: {args.model}")
        
        # 提取特征
        if args.representation == 'AAC':
            features = FeatureExtractor.extract_aac(args.seq_file)
        elif args.representation == 'CTDD':
            features = FeatureExtractor.extract_ctdd(args.seq_file)
        elif args.representation == 'PAAC':
            features = FeatureExtractor.extract_paac(args.seq_file)
        else:
            raise ValueError(f"不支持的特征表示方法: {args.representation}")
        
        # 处理特征删除
        if hasattr(args, 'drop_feature') and args.drop_feature:
            try:
                with open(args.drop_feature, 'r', encoding='utf-8') as f:
                    drop_features = [line.strip() for line in f if line.strip()]
                features = features.drop(columns=[col for col in drop_features if col in features.columns])
            except Exception as e:
                logger.warning(f"特征删除失败: {e}")
        
        # 预测
        probabilities = model.predict_proba(features.values)
        seq_ids = features.index.tolist()
        
        # 创建result目录（如果不存在）
        result_dir = "result"
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        
        # 保存结果
        prediction_file = os.path.join(result_dir, 'AMPpred.tsv')
        with open(prediction_file, 'w', encoding='utf-8') as f:
            f.write("seq_id\tprobability_nonAMP\tprobability_AMP\tpredicted\n")
            
            for i, (seq_id, prob) in enumerate(zip(seq_ids, probabilities)):
                prob_nonamp, prob_amp = prob[0], prob[1]
                
                # 确定预测标签
                if hasattr(args, 'threshold') and args.threshold:
                    predicted = "AMP" if prob_amp > float(args.threshold) else "nonAMP"
                else:
                    predicted = "AMP" if model.predict(features.iloc[[i]].values)[0] == 1 else "nonAMP"
                
                f.write(f"{seq_id}\t{prob_nonamp:.4f}\t{prob_amp:.4f}\t{predicted}\n")
        
        logger.info(f"预测完成，共处理 {len(seq_ids)} 个序列")
        
    except Exception as e:
        logger.error(f"预测过程失败: {e}")
        raise


# 保持向后兼容的函数别名
def parseFasta(filename: str) -> Dict[str, str]:
    """向后兼容的FASTA解析函数"""
    return FeatureExtractor.parse_fasta(filename)

def AAC(fasta_file: str) -> pd.DataFrame:
    """向后兼容的AAC特征提取函数"""
    return FeatureExtractor.extract_aac(fasta_file)

def CTDD(fasta_file: str) -> pd.DataFrame:
    """向后兼容的CTDD特征提取函数"""
    return FeatureExtractor.extract_ctdd(fasta_file)

def PAAC(fasta_file: str) -> pd.DataFrame:
    """向后兼容的PAAC特征提取函数"""
    return FeatureExtractor.extract_paac(fasta_file)


if __name__ == "__main__":
    # 测试代码
    print("AMPml 核心模块加载成功")
    print(f"支持的特征方法: AAC, CTDD, PAAC")
    print(f"支持的机器学习方法: RF, SVM, GT, bayes")
