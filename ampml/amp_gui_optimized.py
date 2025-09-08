#!/usr/bin/env python3
"""
AMPml - 抗菌肽预测软件 (NiceGUI界面优化版)
使用NiceGUI创建现代化Web界面，具有优化的代码质量和用户体验
"""

import os
import sys
import asyncio
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass

from nicegui import ui, run, app

# 使用优化版本后端
from amp_optimized import train, predict, DottableDict
print("🚀 AMPml优化版本已加载")
BACKEND_VERSION = "优化版"

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """训练配置类"""
    method: str = 'RF'
    representation: str = 'CTDD'
    seed: int = 2020
    num_trees: int = 100
    tree_test: bool = False
    feature_importance: bool = False


@dataclass
class PredictionConfig:
    """预测配置类"""
    representation: str = 'CTDD'
    seed: int = 2020
    threshold: Optional[float] = None


class UIState:
    """UI状态管理类"""
    
    def __init__(self):
        self.is_training = False
        self.is_predicting = False
        self.training_cancelled = False
        self.uploaded_files = {}
        # 消息队列用于异步任务与UI通信
        self.message_queue = []
        # UI更新队列用于异步任务向主线程传递UI更新命令
        self.ui_update_queue = []
    
    def reset_training_state(self):
        """重置训练状态"""
        self.is_training = False
        self.training_cancelled = False
    
    def reset_prediction_state(self):
        """重置预测状态"""
        self.is_predicting = False
    
    def add_message(self, message: str, message_type: str = 'info'):
        """添加消息到队列"""
        self.message_queue.append({'text': message, 'type': message_type})
    
    def get_messages(self):
        """获取并清空消息队列"""
        messages = self.message_queue.copy()
        self.message_queue.clear()
        return messages
    
    def add_ui_update(self, component_name: str, action: str, value: Any = None):
        """添加UI更新命令到队列"""
        self.ui_update_queue.append({
            'component': component_name,
            'action': action,
            'value': value
        })
    
    def get_ui_updates(self):
        """获取并清空UI更新队列"""
        updates = self.ui_update_queue.copy()
        self.ui_update_queue.clear()
        return updates


class FileManager:
    """文件管理类"""
    
    @staticmethod
    def ensure_uploads_dir():
        """确保uploads目录存在"""
        uploads_dir = Path("uploads")
        uploads_dir.mkdir(exist_ok=True)
        return uploads_dir
    
    @staticmethod
    def save_uploaded_file(file_content, filename: str, prefix: str = "") -> str:
        """保存上传的文件"""
        uploads_dir = FileManager.ensure_uploads_dir()
        if prefix:
            safe_filename = f"{prefix}_{filename}"
        else:
            safe_filename = filename
        
        file_path = uploads_dir / safe_filename
        
        with open(file_path, 'wb') as f:
            f.write(file_content.read())
        
        logger.info(f"文件已保存: {file_path}")
        return str(file_path)
    
    @staticmethod
    def validate_fasta_file(file_path: str) -> bool:
        """验证FASTA文件格式"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read(1000)  # 读取前1000字符
                return '>' in content
        except Exception:
            return False


class AMPmlAppOptimized:
    """优化版AMPml应用类"""
    
    def __init__(self):
        self.ui_state = UIState()
        self.file_manager = FileManager()
        self.training_config = TrainingConfig()
        self.prediction_config = PredictionConfig()
        
        # UI组件引用
        self.ui_components = {}
        
        self.setup_ui()
        
        # 设置消息处理定时器
        self._setup_message_timer()
    
    def setup_ui(self):
        """设置用户界面"""
        ui.page_title('AMPml - 抗菌肽预测软件 (优化版)')
        
        # 自定义样式
        self._setup_styles()
        
        # 创建主界面
        with ui.column().classes('w-full max-w-7xl mx-auto p-6'):
            self._create_header()
            self._create_main_tabs()
    
    def _setup_styles(self):
        """设置自定义样式"""
        ui.add_head_html('''
            <style>
                .custom-card {
                    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
                    border-radius: 16px;
                    border: 1px solid #e5e7eb;
                    background: white;
                    transition: all 0.3s ease;
                }
                .custom-card:hover {
                    transform: translateY(-2px);
                    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.15);
                }
                .upload-area {
                    border: 2px dashed #d1d5db;
                    border-radius: 12px;
                    padding: 24px;
                    text-align: center;
                    background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
                    transition: all 0.3s ease;
                    min-height: 100px;
                    display: flex;
                    flex-direction: column;
                    justify-content: center;
                }
                .upload-area:hover {
                    border-color: #3b82f6;
                    background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
                    transform: scale(1.02);
                }
                .result-area {
                    background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
                    border: 1px solid #e5e7eb;
                    border-radius: 12px;
                    padding: 16px;
                    margin-top: 16px;
                    min-height: 140px;
                    font-size: 0.9rem;
                    line-height: 1.5;
                }
                .section-title {
                    color: #1f2937;
                    font-size: 1.25rem;
                    font-weight: 600;
                    margin-bottom: 16px;
                    padding-bottom: 8px;
                    border-bottom: 2px solid #e5e7eb;
                }
                .status-success {
                    color: #059669 !important;
                    font-weight: 500;
                }
                .status-pending {
                    color: #6b7280 !important;
                }
                .status-error {
                    color: #dc2626 !important;
                    font-weight: 500;
                }
                .main-title {
                    background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
                    -webkit-background-clip: text;
                    -webkit-text-fill-color: transparent;
                    background-clip: text;
                    text-align: center;
                    font-size: 2.5rem;
                    font-weight: 700;
                    margin-bottom: 2rem;
                }
                .version-badge {
                    display: inline-block;
                    background: linear-gradient(135deg, #10b981 0%, #059669 100%);
                    color: white;
                    padding: 4px 12px;
                    border-radius: 20px;
                    font-size: 0.875rem;
                    font-weight: 500;
                    margin-left: 12px;
                }
                .progress-bar {
                    width: 100%;
                    height: 8px;
                    background: #e5e7eb;
                    border-radius: 4px;
                    overflow: hidden;
                    margin: 12px 0;
                }
                .progress-fill {
                    height: 100%;
                    background: linear-gradient(90deg, #3b82f6, #1d4ed8);
                    width: 0%;
                    transition: width 0.3s ease;
                    border-radius: 4px;
                }
                @keyframes pulse {
                    0%, 100% { opacity: 1; }
                    50% { opacity: 0.5; }
                }
                .pulse {
                    animation: pulse 2s infinite;
                }
            </style>
        ''')
    
    def _create_header(self):
        """创建页头"""
        ui.label('🧬 AMPml - 抗菌肽预测软件').classes('main-title')
        ui.label(f'{BACKEND_VERSION}').classes('version-badge')
        
        # 工作目录提示
        current_dir = os.getcwd()
        with ui.card().classes('w-full mb-4 p-4').style(
            'background: linear-gradient(135deg, #e0f2fe 0%, #f3e5f5 100%); border: 1px solid #e1bee7;'
        ):
            with ui.row().classes('w-full items-center gap-3'):
                ui.icon('folder').classes('text-purple-600')
                ui.label('当前工作目录:').classes('font-medium text-gray-700')
                ui.label(current_dir).classes('font-mono text-sm bg-white px-2 py-1 rounded border')
                ui.label('(所有生成的文件将保存在此目录)').classes('text-sm text-gray-600')
    
    def _create_main_tabs(self):
        """创建主要功能选项卡"""
        with ui.tabs().classes('w-full') as tabs:
            train_tab = ui.tab('🚀 训练模型', icon='model_training')
            predict_tab = ui.tab('🔮 预测分析', icon='psychology')
            about_tab = ui.tab('ℹ️ 关于软件', icon='info')
        
        with ui.tab_panels(tabs, value=train_tab).classes('w-full'):
            with ui.tab_panel(train_tab):
                self._create_train_panel()
            
            with ui.tab_panel(predict_tab):
                self._create_predict_panel()
            
            with ui.tab_panel(about_tab):
                self._create_about_panel()
    
    def _create_train_panel(self):
        """创建训练面板"""
        with ui.column().classes('w-full gap-6'):
            self._create_file_upload_section()
            self._create_training_config_section()
            self._create_training_execution_section()
    
    def _create_file_upload_section(self):
        """创建文件上传区域"""
        with ui.card().classes('w-full custom-card p-6'):
            ui.label('📁 数据文件上传').classes('section-title')
            
            with ui.grid(columns=3).classes('w-full gap-4'):
                # 正样本文件
                with ui.column().classes('upload-area'):
                    ui.label('正样本文件 (AMP)').classes('font-medium text-gray-700 mb-3')
                    ui.label('FASTA格式，包含抗菌肽序列').classes('text-xs text-gray-500 mb-2')
                    
                    self.ui_components['positive_upload'] = ui.upload(
                        on_upload=lambda e: self._handle_file_upload(e, 'positive'),
                        multiple=False,
                        auto_upload=True
                    ).classes('w-full')
                    
                    self.ui_components['positive_status'] = ui.label('未选择文件').classes('text-sm status-pending mt-2')
                
                # 负样本文件
                with ui.column().classes('upload-area'):
                    ui.label('负样本文件 (非AMP)').classes('font-medium text-gray-700 mb-3')
                    ui.label('FASTA格式，包含非抗菌肽序列').classes('text-xs text-gray-500 mb-2')
                    
                    self.ui_components['negative_upload'] = ui.upload(
                        on_upload=lambda e: self._handle_file_upload(e, 'negative'),
                        multiple=False,
                        auto_upload=True
                    ).classes('w-full')
                    
                    self.ui_components['negative_status'] = ui.label('未选择文件').classes('text-sm status-pending mt-2')
                
                # 特征删除文件（可选）
                with ui.column().classes('upload-area'):
                    ui.label('特征删除文件 (可选)').classes('font-medium text-gray-700 mb-3')
                    ui.label('每行一个要删除的特征名').classes('text-xs text-gray-500 mb-2')
                    
                    self.ui_components['drop_feature_upload'] = ui.upload(
                        on_upload=lambda e: self._handle_file_upload(e, 'drop_feature'),
                        multiple=False,
                        auto_upload=True
                    ).classes('w-full')
                    
                    self.ui_components['drop_feature_status'] = ui.label('未选择文件').classes('text-sm status-pending mt-2')
    
    def _create_training_config_section(self):
        """创建训练配置区域"""
        with ui.card().classes('w-full custom-card p-6'):
            ui.label('⚙️ 训练参数配置').classes('section-title')
            
            with ui.grid(columns=2).classes('w-full gap-6'):
                # 左列：算法参数
                with ui.column().classes('gap-4'):
                    ui.label('🤖 算法设置').classes('text-base font-medium text-gray-700 mb-2')
                    
                    self.ui_components['ml_method'] = ui.select(
                        label='机器学习方法',
                        options={
                            'RF': '🌲 随机森林 (推荐)',
                            'SVM': '🎯 支持向量机',
                            'GT': '🚀 梯度提升',
                            'XGB': '⚡ XGBoost (极速梯度提升)',
                            'bayes': '📊 朴素贝叶斯'
                        },
                        value='RF',
                        on_change=self._on_method_change
                    ).classes('w-full')
                    
                    self.ui_components['representation'] = ui.select(
                        label='特征表示方法',
                        options={
                            'CTDD': '🧬 CTDD (推荐, 195维)',
                            'PAAC': '🔬 PAAC (29维)',
                            'AAC': '⚡ AAC (快速, 20维)'
                        },
                        value='CTDD'
                    ).classes('w-full')
                
                # 右列：参数设置
                with ui.column().classes('gap-4'):
                    ui.label('🎛️ 参数设置').classes('text-base font-medium text-gray-700 mb-2')
                    
                    with ui.row().classes('w-full gap-3'):
                        self.ui_components['seed'] = ui.number(
                            label='随机种子',
                            value=2020,
                            min=1,
                            max=9999,
                            step=1
                        ).classes('flex-1')
                        
                        self.ui_components['num_trees'] = ui.number(
                            label='树数量 (RF专用)',
                            value=100,
                            min=10,
                            max=500,
                            step=10
                        ).classes('flex-1')
                    
                    ui.separator().classes('my-2')
                    ui.label('🔬 高级选项').classes('text-sm font-medium text-gray-600 mb-2')
                    
                    with ui.column().classes('gap-2'):
                        self.ui_components['tree_test'] = ui.checkbox(
                            '树数量优化测试', 
                            value=False
                        ).tooltip('测试不同树数量的性能')
                        
                        self.ui_components['feature_importance'] = ui.checkbox(
                            '特征重要性分析', 
                            value=False
                        ).tooltip('基于sklearn计算和可视化特征重要性')
    
    def _create_training_execution_section(self):
        """创建训练执行区域"""
        with ui.card().classes('w-full custom-card p-6'):
            # 训练控制
            with ui.row().classes('w-full justify-between items-center mb-6'):
                ui.label('🚀 模型训练').classes('text-xl font-semibold text-gray-700')
                
                with ui.row().classes('items-center gap-4'):
                    self.ui_components['training_spinner'] = ui.spinner(size='md', color='primary')
                    self.ui_components['training_spinner'].set_visibility(False)
                    
                    self.ui_components['train_button'] = ui.button(
                        '开始训练', 
                        on_click=self._start_training_async
                    ).props('size=lg color=primary unelevated').classes('px-6 py-2')
                    
                    self.ui_components['stop_button'] = ui.button(
                        '停止训练', 
                        on_click=self._stop_training
                    ).props('size=lg color=negative unelevated').classes('px-6 py-2')
                    self.ui_components['stop_button'].set_visibility(False)
            
            # 进度指示
            self.ui_components['progress_container'] = ui.row().classes('w-full mb-4')
            self.ui_components['progress_container'].set_visibility(False)
            
            # 结果展示
            self.ui_components['result_container'] = ui.column().classes('w-full')
            self.ui_components['result_container'].set_visibility(False)
            
            with self.ui_components['result_container']:
                ui.label('📊 训练结果').classes('text-lg font-medium text-gray-700 mb-3')
                
                with ui.row().classes('w-full gap-6'):
                    # 左侧：精简文字结果
                    with ui.column().classes('w-80'):
                        self.ui_components['training_result'] = ui.markdown('').classes('result-area')
                    
                    # 右侧：扩大的图表展示
                    with ui.column().classes('flex-1'):
                        # ROC图和特征重要性图的容器
                        with ui.row().classes('w-full gap-3'):
                            # ROC图
                            with ui.column().classes('flex-1'):
                                ui.label('ROC曲线').classes('text-sm font-medium text-gray-600 mb-2 text-center')
                                self.ui_components['roc_image'] = ui.image().classes(
                                    'w-full object-contain bg-gray-50 rounded-lg border'
                                ).style('min-height: 250px; max-height: 450px;')
                                self.ui_components['roc_image'].set_visibility(False)
                                
                                self.ui_components['roc_placeholder'] = ui.label(
                                    'ROC曲线将在训练完成后显示'
                                ).classes('text-xs text-gray-500 text-center mt-8 px-2')
                            
                            # 特征重要性图
                            with ui.column().classes('flex-1'):
                                ui.label('特征重要性').classes('text-sm font-medium text-gray-600 mb-2 text-center')
                                self.ui_components['feature_importance_image'] = ui.image().classes(
                                    'w-full object-contain bg-gray-50 rounded-lg border'
                                ).style('min-height: 250px; max-height: 450px;')
                                self.ui_components['feature_importance_image'].set_visibility(False)
                                
                                self.ui_components['feature_importance_placeholder'] = ui.label(
                                    '特征重要性图将在启用分析后显示'
                                ).classes('text-xs text-gray-500 text-center mt-8 px-2')
    
    def _create_predict_panel(self):
        """创建预测面板"""
        with ui.column().classes('w-full gap-6'):
            self._create_prediction_upload_section()
            self._create_prediction_config_section()
            self._create_prediction_execution_section()
    
    def _create_prediction_upload_section(self):
        """创建预测文件上传区域"""
        with ui.card().classes('w-full custom-card p-6'):
            ui.label('📁 预测文件上传').classes('section-title')
            
            with ui.grid(columns=3).classes('w-full gap-4'):
                # 模型文件
                with ui.column().classes('upload-area'):
                    ui.label('训练模型文件').classes('font-medium text-gray-700 mb-3')
                    ui.label('.joblib格式文件 (推荐) 或 .model格式文件').classes('text-xs text-gray-500 mb-2')
                    
                    self.ui_components['model_upload'] = ui.upload(
                        on_upload=lambda e: self._handle_file_upload(e, 'model'),
                        multiple=False,
                        auto_upload=True
                    ).classes('w-full')
                    
                    self.ui_components['model_status'] = ui.label('未选择文件').classes('text-sm status-pending mt-2')
                
                # 序列文件
                with ui.column().classes('upload-area'):
                    ui.label('待预测序列文件').classes('font-medium text-gray-700 mb-3')
                    ui.label('FASTA格式').classes('text-xs text-gray-500 mb-2')
                    
                    self.ui_components['seq_upload'] = ui.upload(
                        on_upload=lambda e: self._handle_file_upload(e, 'sequence'),
                        multiple=False,
                        auto_upload=True
                    ).classes('w-full')
                    
                    self.ui_components['seq_status'] = ui.label('未选择文件').classes('text-sm status-pending mt-2')
                
                # 特征删除文件（可选）
                with ui.column().classes('upload-area'):
                    ui.label('特征删除文件 (可选)').classes('font-medium text-gray-700 mb-3')
                    ui.label('与训练时保持一致').classes('text-xs text-gray-500 mb-2')
                    
                    self.ui_components['pred_drop_feature_upload'] = ui.upload(
                        on_upload=lambda e: self._handle_file_upload(e, 'pred_drop_feature'),
                        multiple=False,
                        auto_upload=True
                    ).classes('w-full')
                    
                    self.ui_components['pred_drop_feature_status'] = ui.label('未选择文件').classes('text-sm status-pending mt-2')
    
    def _create_prediction_config_section(self):
        """创建预测配置区域"""
        with ui.card().classes('w-full custom-card p-6'):
            ui.label('⚙️ 预测参数配置').classes('section-title')
            
            with ui.grid(columns=2).classes('w-full gap-6'):
                # 左列：基础参数
                with ui.column().classes('gap-4'):
                    ui.label('🔧 基础设置').classes('text-base font-medium text-gray-700 mb-2')
                    
                    self.ui_components['pred_representation'] = ui.select(
                        label='特征表示方法 (须与训练一致)',
                        options={
                            'CTDD': '🧬 CTDD (组成-转换-分布)',
                            'PAAC': '🔬 PAAC (伪氨基酸组成)',
                            'AAC': '⚡ AAC (氨基酸组成)'
                        },
                        value='CTDD'
                    ).classes('w-full')
                    
                    self.ui_components['pred_seed'] = ui.number(
                        label='随机种子',
                        value=2020,
                        min=1,
                        max=9999
                    ).classes('w-full')
                
                # 右列：高级参数
                with ui.column().classes('gap-4'):
                    ui.label('🎯 高级设置').classes('text-base font-medium text-gray-700 mb-2')
                    
                    self.ui_components['threshold'] = ui.number(
                        label='预测阈值 (0-1)',
                        value=None,
                        min=0,
                        max=1,
                        step=0.01,
                        placeholder='留空使用模型默认阈值'
                    ).classes('w-full')
                    
                    ui.label('💡 阈值越高，预测越保守；阈值越低，预测越宽松').classes('text-xs text-gray-500 mt-1')
    
    def _create_prediction_execution_section(self):
        """创建预测执行区域"""
        with ui.card().classes('w-full custom-card p-6'):
            # 预测控制
            with ui.row().classes('w-full justify-between items-center mb-6'):
                ui.label('🔮 序列预测').classes('text-xl font-semibold text-gray-700')
                
                with ui.row().classes('items-center gap-4'):
                    self.ui_components['prediction_spinner'] = ui.spinner(size='md', color='positive')
                    self.ui_components['prediction_spinner'].set_visibility(False)
                    
                    ui.button(
                        '开始预测', 
                        on_click=self._start_prediction_async
                    ).props('size=lg color=positive unelevated').classes('px-6 py-2')
            
            # 预测结果
            ui.label('📋 预测结果').classes('text-lg font-medium text-gray-700 mb-3')
            self.ui_components['prediction_result'] = ui.markdown('等待预测完成...').classes('result-area')
            
            # 下载按钮
            self.ui_components['download_button'] = ui.button(
                '💾 下载预测结果',
                on_click=self._download_results
            ).props('color=secondary unelevated').classes('mt-4 px-6 py-2')
            self.ui_components['download_button'].set_visibility(False)
    
    def _setup_message_timer(self):
        """设置消息处理定时器"""
        def process_messages():
            """处理消息队列中的消息，显示通知并记录到日志"""
            messages = self.ui_state.get_messages()
            for msg in messages:
                try:
                    if msg['type'] == 'positive':
                        ui.notify(msg['text'], color='positive', position='top')
                        logger.info(f"成功: {msg['text']}")
                    elif msg['type'] == 'negative':
                        ui.notify(msg['text'], color='negative', position='top')
                        logger.error(f"错误: {msg['text']}")
                    elif msg['type'] == 'warning':
                        ui.notify(msg['text'], color='warning', position='top')
                        logger.warning(f"警告: {msg['text']}")
                    else:
                        ui.notify(msg['text'], color='info', position='top')
                        logger.info(f"信息: {msg['text']}")
                except Exception as e:
                    logger.error(f"消息处理失败: {e}")
            
            # 处理UI更新队列
            ui_updates = self.ui_state.get_ui_updates()
            for update in ui_updates:
                try:
                    self._process_ui_update(update)
                except Exception as e:
                    logger.error(f"UI更新处理失败: {e}")
        
        # 每0.5秒检查一次消息队列和UI更新队列
        ui.timer(0.5, process_messages)
    
    def _process_ui_update(self, update: Dict[str, Any]):
        """处理UI更新命令"""
        component_name = update['component']
        action = update['action']
        value = update['value']
        
        if component_name not in self.ui_components:
            logger.warning(f"未找到UI组件: {component_name}")
            return
        
        component = self.ui_components[component_name]
        
        try:
            if action == 'set_content':
                component.content = value
            elif action == 'set_text':
                component.text = value
            elif action == 'set_visibility':
                component.set_visibility(value)
            elif action == 'set_props':
                component.props(value)
            elif action == 'enable':
                component.enable()
            elif action == 'disable':
                component.disable()
            elif action == 'set_source':
                component.set_source(value)
            elif action == 'set_classes':
                component.classes(value)
            else:
                logger.warning(f"未知的UI更新动作: {action}")
        except Exception as e:
            logger.error(f"UI更新执行失败: {e}")
    
    def _create_about_panel(self):
        """创建关于面板"""
        with ui.column().classes('w-full gap-6'):
            # 软件使用方法
            with ui.card().classes('w-full custom-card p-6'):
                ui.label('📱 AMPml 软件使用方法').classes('section-title')
                
                # 启动软件部分
                ui.label('🚀 启动软件').classes('text-lg font-medium text-gray-700 mb-3')
                ui.markdown('''
                **第1步：打开终端/命令提示符**

                **第2步：进入软件目录**
                ```bash
                cd AMPml-main
                ```

                **第3步：启动程序**
                ```bash
                python run_gui.py
                ```

                **第4步：在浏览器中访问**
                - 通常是 `http://localhost:8080`
                - 浏览器会自动打开界面
                ''')
                
            # 模型训练步骤
            with ui.card().classes('w-full custom-card p-6'):
                ui.label('📊 模型训练步骤').classes('section-title')
                
                with ui.grid(columns=2).classes('w-full gap-6'):
                    # 左列：训练步骤
                    with ui.column().classes('gap-4'):
                        ui.label('📂 第1步：上传训练数据').classes('font-medium text-gray-700')
                        ui.markdown('''
                        - **正样本文件**：包含AMP序列的FASTA文件
                        - **负样本文件**：包含非AMP序列的FASTA文件
                        - **特征删除文件**（可选）：要排除的特征列表
                        ''')
                        
                        ui.label('⚙️ 第2步：选择参数').classes('font-medium text-gray-700 mt-4')
                        ui.markdown('''
                        **特征表示方法**：
                        - `AAC`：氨基酸组成（20个特征）
                        - `CTDD`：组成-转换-分布描述符（40个特征）
                        - `PAAC`：伪氨基酸组成（29个特征）
                        
                        **机器学习方法**：
                        - `RF`：随机森林（推荐）
                        - `SVM`：支持向量机
                        - `GT`：梯度提升
                        - `XGB`：XGBoost极速梯度提升
                        - `Bayes`：朴素贝叶斯
                        ''')
                    
                    # 右列：参数设置和执行
                    with ui.column().classes('gap-4'):
                        ui.label('🎛️ 第3步：设置训练参数').classes('font-medium text-gray-700')
                        ui.markdown('''
                        - **随机种子**：默认2020（保证结果可重现）
                        - **树的数量**：默认100（仅RF方法）
                        - **特征重要性分析**：可选勾选
                        ''')
                        
                        ui.label('▶️ 第4步：开始训练').classes('font-medium text-gray-700 mt-4')
                        ui.markdown('''
                        - 点击"开始训练"按钮
                        - 等待训练完成
                        - 查看ROC曲线和性能指标
                        - 所有结果文件保存在 `result/` 目录
                        ''')
                        
            # 序列预测和结果说明
            with ui.card().classes('w-full custom-card p-6'):
                ui.label('🔮 序列预测步骤').classes('section-title')
                
                with ui.grid(columns=2).classes('w-full gap-6'):
                    # 左列：预测步骤
                    with ui.column().classes('gap-4'):
                        ui.label('📋 预测操作流程').classes('font-medium text-gray-700')
                        ui.markdown('''
                        **第1步**：上传预测文件
                        - 包含待预测序列的FASTA文件
                        
                        **第2步**：选择模型
                        - 选择训练好的模型文件（.joblib格式推荐，.model格式兼容）
                        
                        **第3步**：开始预测
                        - 点击"开始预测"按钮
                        - 查看预测结果
                        ''')
                        
                        ui.label('💡 使用建议').classes('font-medium text-gray-700 mt-4')
                        ui.markdown('''
                        - 确保FASTA文件格式正确
                        - 正负样本数量尽量平衡
                        - 序列长度建议在5-50个氨基酸之间
                        - 初次使用推荐：`AAC` + `RF`
                        ''')
                    
                    # 右列：结果文件和问题解答
                    with ui.column().classes('gap-4'):
                        ui.label('📁 结果文件说明').classes('font-medium text-gray-700')
                        ui.markdown('''
                        **训练结果文件**：
                        - `AMPpred_[方法].joblib`：训练好的模型
                        - `model_[算法]_score.txt`：评估指标
                        - `ROC_[算法].png`：ROC曲线图
                        - `feature_importances.txt`：特征重要性
                        
                        **预测结果文件**：
                        - `AMPpred.tsv`：预测结果表格
                          - `seq_id`：序列ID
                          - `probability_AMP`：AMP概率
                          - `predicted`：预测结果
                        ''')
                        
                        ui.label('❗ 常见问题').classes('font-medium text-gray-700 mt-4')
                        ui.markdown('''
                        - **文件上传失败**：检查FASTA格式
                        - **训练时间过长**：减少树的数量
                        - **预测不准确**：尝试不同特征方法
                        - **结果解读**：AUC > 0.8 表示性能良好
                        ''')
    
    # 事件处理方法
    async def _handle_file_upload(self, event, file_type: str):
        """处理文件上传"""
        try:
            if not event.content:
                return
            
            # 保存文件
            file_path = self.file_manager.save_uploaded_file(
                event.content, event.name, file_type
            )
            
            # 验证FASTA格式
            if file_type in ['positive', 'negative', 'sequence']:
                if not self.file_manager.validate_fasta_file(file_path):
                    # 确定正确的状态键
                    error_status_key = f'{file_type}_status'
                    if file_type == 'sequence':
                        error_status_key = 'seq_status'
                    
                    self.ui_components[error_status_key].text = f'❌ {event.name} (格式错误)'
                    self.ui_components[error_status_key].classes('text-sm status-error')
                    return
            
            # 验证模型文件格式
            elif file_type == 'model':
                if not (event.name.endswith('.joblib') or event.name.endswith('.model')):
                    self.ui_components['model_status'].text = f'❌ {event.name} (请选择.joblib或.model格式)'
                    self.ui_components['model_status'].classes('text-sm status-error')
                    return
            
            # 更新状态
            self.ui_state.uploaded_files[file_type] = file_path
            status_key = f'{file_type}_status'
            if file_type == 'pred_drop_feature':
                status_key = 'pred_drop_feature_status'
            elif file_type == 'sequence':
                status_key = 'seq_status'
            
            self.ui_components[status_key].text = f'✅ {event.name}'
            self.ui_components[status_key].classes('text-sm status-success')
            
        except Exception as e:
            logger.error(f"文件上传失败: {e}")
    
    def _on_method_change(self, event):
        """当机器学习方法改变时的处理"""
        method = event.value
        # 记录方法选择到日志
        logger.info(f"用户选择机器学习方法: {method}")
    
    def _stop_training(self):
        """停止训练"""
        self.ui_state.training_cancelled = True
        logger.info("用户请求停止训练")
    
    def _start_training_async(self):
        """异步启动训练"""
        # 防止重复点击
        if self.ui_state.is_training:
            ui.notify('训练正在进行中，请耐心等待...', color='warning')
            return
        
        # 在主线程中读取UI组件值，避免在异步任务中读取
        try:
            training_params = {
                'method': self.ui_components['ml_method'].value,
                'representation': self.ui_components['representation'].value,
                'seed': int(self.ui_components['seed'].value),
                'num_trees': int(self.ui_components['num_trees'].value),
                'tree_test': self.ui_components['tree_test'].value,
                'feature_importance': self.ui_components['feature_importance'].value
            }
            asyncio.create_task(self._start_training(training_params))
        except Exception as e:
            ui.notify(f'参数读取失败: {str(e)}', color='negative')
            logger.error(f"参数读取失败: {e}")
    
    async def _start_training(self, training_params: dict):
        """开始训练模型"""
        try:
            # 验证必要文件
            if 'positive' not in self.ui_state.uploaded_files or 'negative' not in self.ui_state.uploaded_files:
                error_msg = '❌ 请先上传正样本和负样本文件！'
                # 通过队列更新UI，避免在异步任务中直接操作
                self.ui_state.add_ui_update('training_result', 'set_content', error_msg)
                self.ui_state.add_ui_update('result_container', 'set_visibility', True)
                self.ui_state.add_message(error_msg, 'negative')
                return
            
            # 验证文件是否存在
            positive_file = self.ui_state.uploaded_files['positive']
            negative_file = self.ui_state.uploaded_files['negative']
            
            if not os.path.exists(positive_file):
                error_msg = f'❌ 正样本文件不存在: {positive_file}'
                self.ui_state.add_ui_update('training_result', 'set_content', error_msg)
                self.ui_state.add_ui_update('result_container', 'set_visibility', True)
                self.ui_state.add_message(error_msg, 'negative')
                return
                
            if not os.path.exists(negative_file):
                error_msg = f'❌ 负样本文件不存在: {negative_file}'
                self.ui_state.add_ui_update('training_result', 'set_content', error_msg)
                self.ui_state.add_ui_update('result_container', 'set_visibility', True)
                self.ui_state.add_message(error_msg, 'negative')
                return
            
            # 设置训练状态
            self.ui_state.reset_training_state()
            self.ui_state.is_training = True
            
            # 更新UI状态 - 通过队列避免直接操作
            self.ui_state.add_ui_update('train_button', 'set_text', '训练中...')
            self.ui_state.add_ui_update('train_button', 'set_props', 'color=secondary')
            self.ui_state.add_ui_update('train_button', 'disable', None)
            self.ui_state.add_ui_update('stop_button', 'set_visibility', True)
            self.ui_state.add_ui_update('training_spinner', 'set_visibility', True)
            self.ui_state.add_ui_update('result_container', 'set_visibility', True)
            self.ui_state.add_ui_update('roc_image', 'set_visibility', False)
            self.ui_state.add_ui_update('roc_placeholder', 'set_visibility', True)
            
            # 准备训练参数，使用传入的参数而不是读取UI组件
            args = DottableDict({
                'method': training_params['method'],
                'representation': training_params['representation'],
                'seed': training_params['seed'],
                'num_trees': training_params['num_trees'],
                'tree_test': training_params['tree_test'],
                'feature_importance': training_params['feature_importance'],
                'positive': self.ui_state.uploaded_files['positive'],
                'negative': self.ui_state.uploaded_files['negative'],
                'drop_feature': self.ui_state.uploaded_files.get('drop_feature')
            })
            
            # 执行训练 - 使用队列更新UI而不是直接操作
            self.ui_state.add_ui_update('training_result', 'set_content', '🔄 正在训练模型，请稍候...')
            
            # 使用try-except包装后台任务执行
            try:
                result = await asyncio.get_event_loop().run_in_executor(None, train, args)
            except Exception as train_error:
                # 如果train函数内部出错，捕获并处理
                logger.error(f"训练函数执行失败: {train_error}")
                raise train_error
            
            # 检查是否被取消
            if self.ui_state.training_cancelled:
                self._update_training_ui_cancelled()
                return
            
            # 处理训练结果
            await self._process_training_result(args, result)
            
        except Exception as e:
            logger.error(f"训练失败: {e}")
            self._update_training_ui_error(str(e))
        
        finally:
            self.ui_state.reset_training_state()
    
    
    def _update_training_ui_cancelled(self):
        """更新训练取消时的UI状态"""
        self.ui_state.add_ui_update('training_result', 'set_content', '❌ 训练已被用户取消')
        self.ui_state.add_message('训练已取消', 'warning')
        self._reset_training_ui_via_queue()
    
    def _update_training_ui_error(self, error_msg: str):
        """更新训练错误时的UI状态"""
        self.ui_state.add_ui_update('training_result', 'set_content', f'❌ 训练失败: {error_msg}')
        self.ui_state.add_message(f'训练失败: {error_msg}', 'negative')
        self._reset_training_ui_via_queue()
    
    def _reset_training_ui_via_queue(self):
        """通过队列重置训练UI状态"""
        self.ui_state.add_ui_update('train_button', 'set_text', '开始训练')
        self.ui_state.add_ui_update('train_button', 'set_props', 'color=primary')
        self.ui_state.add_ui_update('train_button', 'enable', None)
        self.ui_state.add_ui_update('stop_button', 'set_visibility', False)
        self.ui_state.add_ui_update('training_spinner', 'set_visibility', False)
    
    async def _process_training_result(self, args, result):
        """处理训练结果"""
        try:
            # 检测ROC和特征重要性数据
            roc_data = None
            feature_importance_data = None
            if isinstance(result, tuple):
                if args.method == 'RF' and len(result) == 6:
                    # RF方法: (oob_score, balanced_accuracy, auc_score, CM, roc_data, feature_importance_data)
                    roc_data = result[4]
                    feature_importance_data = result[5]
                    result = result[:4]  # 保持原有结构
                elif args.method != 'RF' and len(result) == 4:
                    # 其他方法: (cv_score, CM, roc_data, feature_importance_data)
                    roc_data = result[2]
                    feature_importance_data = result[3]
                    result = result[:2]  # 保持原有结构
            
            # 检查生成的文件（在result目录中）
            generated_files = []
            result_dir = "result"
            model_file = f"AMPpred_{args.representation}.joblib"
            score_file = f"model_{args.method}_score.txt"
            feature_importance_file = "feature_importances.txt"
            
            model_path = os.path.join(result_dir, model_file)
            score_path = os.path.join(result_dir, score_file)
            feature_importance_path = os.path.join(result_dir, feature_importance_file)
            
            if os.path.exists(model_path):
                generated_files.append(f"📄 result/{model_file} (训练模型)")
            if os.path.exists(score_path):
                generated_files.append(f"📋 result/{score_file} (评估结果)")
            if os.path.exists(feature_importance_path):
                generated_files.append(f"📊 result/{feature_importance_file} (特征重要性数据)")
            
            files_info = "\n\n".join([f"- {file}" for file in generated_files]) if generated_files else "- 无文件生成"
            
            # 每项单独一行的清晰显示
            if args.method == 'RF':
                result_text = f'''**🎯 {args.method} 训练结果**

**OOB准确率**: {result[0]:.4f}

**F1分数**: {result[1]:.4f}

**AUC分数**: {result[2]:.4f}

**混淆矩阵**: TN={result[3][0][0]}, FP={result[3][0][1]}, FN={result[3][1][0]}, TP={result[3][1][1]}

**📁 生成文件**:

{files_info}'''
            else:
                result_text = f'''**🎯 {args.method} 训练结果**

**交叉验证准确率**: {result[0]:.4f}

**混淆矩阵**: TN={result[1][0][0]}, FP={result[1][0][1]}, FN={result[1][1][0]}, TP={result[1][1][1]}

**📁 生成文件**:

{files_info}'''
            
            # 使用队列更新UI
            self.ui_state.add_ui_update('training_result', 'set_content', result_text)
            
            # 显示ROC曲线
            await self._display_roc_curve(args.method, roc_data)
            
            # 显示特征重要性图（如果有）
            if feature_importance_data:
                await self._display_feature_importance(feature_importance_data)
            else:
                # 隐藏特征重要性图
                self.ui_state.add_ui_update('feature_importance_image', 'set_visibility', False)
                self.ui_state.add_ui_update('feature_importance_placeholder', 'set_visibility', True)
            
            # 更新UI状态
            self.ui_state.add_ui_update('train_button', 'set_text', '重新训练')
            self.ui_state.add_ui_update('train_button', 'set_props', 'color=primary')
            self.ui_state.add_ui_update('train_button', 'enable', None)
            self.ui_state.add_ui_update('stop_button', 'set_visibility', False)
            self.ui_state.add_ui_update('training_spinner', 'set_visibility', False)
            
            file_count = len(generated_files)
            # 使用消息队列显示通知
            self.ui_state.add_message(f'✅ 训练完成！生成了 {file_count} 个文件', 'positive')
            
        except Exception as e:
            logger.error(f"结果处理失败: {e}")
            self._update_training_ui_error(f"结果处理失败: {str(e)}")
    
    async def _display_roc_curve(self, method: str, roc_data: str = None):
        """显示ROC曲线"""
        try:
            if roc_data and roc_data.startswith('data:image/png;base64,'):
                # 优先使用base64数据
                self.ui_state.add_ui_update('roc_image', 'set_source', roc_data)
                self.ui_state.add_ui_update('roc_image', 'set_visibility', True)
                self.ui_state.add_ui_update('roc_placeholder', 'set_visibility', False)
                logger.info("ROC曲线显示成功（base64数据）")
            else:
                # 降级到文件读取
                roc_file = f"ROC_{method}.png"
                if os.path.exists(roc_file):
                    import base64
                    with open(roc_file, 'rb') as f:
                        file_data = f.read()
                        roc_base64 = base64.b64encode(file_data).decode()
                        self.ui_state.add_ui_update('roc_image', 'set_source', f'data:image/png;base64,{roc_base64}')
                        self.ui_state.add_ui_update('roc_image', 'set_visibility', True)
                        self.ui_state.add_ui_update('roc_placeholder', 'set_visibility', False)
                    
                    # 删除临时文件
                    try:
                        os.remove(roc_file)
                    except:
                        pass
                    logger.info("ROC曲线显示成功（文件读取）")
                else:
                    logger.warning(f"ROC数据和文件都不可用: {method}")
                    self.ui_state.add_ui_update('roc_image', 'set_visibility', False)
                    self.ui_state.add_ui_update('roc_placeholder', 'set_visibility', True)
        except Exception as e:
            logger.error(f"ROC曲线显示失败: {e}")
            self.ui_state.add_ui_update('roc_image', 'set_visibility', False)
            self.ui_state.add_ui_update('roc_placeholder', 'set_visibility', True)
    
    async def _display_feature_importance(self, feature_importance_data: str):
        """显示特征重要性图"""
        try:
            if feature_importance_data and feature_importance_data.startswith('data:image/png;base64,'):
                self.ui_state.add_ui_update('feature_importance_image', 'set_source', feature_importance_data)
                self.ui_state.add_ui_update('feature_importance_image', 'set_visibility', True)
                self.ui_state.add_ui_update('feature_importance_placeholder', 'set_visibility', False)
                logger.info("特征重要性图显示成功")
            else:
                logger.warning("特征重要性数据无效")
                self.ui_state.add_ui_update('feature_importance_image', 'set_visibility', False)
                self.ui_state.add_ui_update('feature_importance_placeholder', 'set_visibility', True)
        except Exception as e:
            logger.error(f"显示特征重要性图失败: {e}")
            self.ui_state.add_ui_update('feature_importance_image', 'set_visibility', False)
            self.ui_state.add_ui_update('feature_importance_placeholder', 'set_visibility', True)
    
    def _start_prediction_async(self):
        """异步启动预测"""
        # 防止重复点击
        if self.ui_state.is_predicting:
            ui.notify('预测正在进行中，请耐心等待...', color='warning')
            return
        
        # 在主线程中读取UI组件值，避免在异步任务中读取
        try:
            prediction_params = {
                'representation': self.ui_components['pred_representation'].value,
                'seed': int(self.ui_components['pred_seed'].value),
                'threshold': self.ui_components['threshold'].value if self.ui_components['threshold'].value else None
            }
            asyncio.create_task(self._start_prediction(prediction_params))
        except Exception as e:
            ui.notify(f'参数读取失败: {str(e)}', color='negative')
            logger.error(f"预测参数读取失败: {e}")
    
    async def _start_prediction(self, prediction_params: dict):
        """开始预测"""
        try:
            # 验证必要文件
            if 'model' not in self.ui_state.uploaded_files or 'sequence' not in self.ui_state.uploaded_files:
                error_msg = '❌ 请先上传模型文件和序列文件！'
                self.ui_state.add_ui_update('prediction_result', 'set_content', error_msg)
                self.ui_state.add_message(error_msg, 'negative')
                return
            
            # 验证文件是否存在
            model_file = self.ui_state.uploaded_files['model']
            sequence_file = self.ui_state.uploaded_files['sequence']
            
            if not os.path.exists(model_file):
                error_msg = f'❌ 模型文件不存在: {model_file}'
                self.ui_state.add_ui_update('prediction_result', 'set_content', error_msg)
                self.ui_state.add_message(error_msg, 'negative')
                return
                
            if not os.path.exists(sequence_file):
                error_msg = f'❌ 序列文件不存在: {sequence_file}'
                self.ui_state.add_ui_update('prediction_result', 'set_content', error_msg)
                self.ui_state.add_message(error_msg, 'negative')
                return
            
            # 设置预测状态
            self.ui_state.reset_prediction_state()
            self.ui_state.is_predicting = True
            
            self.ui_state.add_ui_update('prediction_spinner', 'set_visibility', True)
            self.ui_state.add_ui_update('prediction_result', 'set_content', '🔄 正在预测序列，请稍候...')
            
            # 准备预测参数，使用传入的参数而不是读取UI组件
            args = DottableDict({
                'representation': prediction_params['representation'],
                'seed': prediction_params['seed'],
                'threshold': prediction_params['threshold'],
                'model': self.ui_state.uploaded_files['model'],
                'seq_file': self.ui_state.uploaded_files['sequence'],
                'drop_feature': self.ui_state.uploaded_files.get('pred_drop_feature')
            })
            
            # 执行预测
            try:
                await asyncio.get_event_loop().run_in_executor(None, predict, args)
            except Exception as predict_error:
                # 如果predict函数内部出错，捕获并处理
                logger.error(f"预测函数执行失败: {predict_error}")
                raise predict_error
            
            # 处理预测结果
            await self._process_prediction_result()
            
        except Exception as e:
            logger.error(f"预测失败: {e}")
            self.ui_state.add_ui_update('prediction_result', 'set_content', f'❌ 预测失败: {str(e)}')
            self.ui_state.add_message(f'预测失败: {str(e)}', 'negative')
        
        finally:
            self.ui_state.reset_prediction_state()
            self.ui_state.add_ui_update('prediction_spinner', 'set_visibility', False)
    
    async def _process_prediction_result(self):
        """处理预测结果"""
        try:
            result_dir = "result"
            prediction_file = os.path.join(result_dir, 'AMPpred.tsv')
            
            if os.path.exists(prediction_file):
                with open(prediction_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                
                # 统计预测结果
                amp_count = sum(1 for line in lines[1:] if line.strip().endswith('\tAMP'))
                total_count = len(lines) - 1
                amp_percentage = (amp_count / total_count) * 100 if total_count > 0 else 0
                
                result_text = f'''### 🔮 预测完成
                
**📊 统计信息**:
- **总序列数**: {total_count}
- **预测为AMP**: {amp_count} ({amp_percentage:.1f}%)
- **预测为非AMP**: {total_count - amp_count} ({100-amp_percentage:.1f}%)

**📋 预测结果预览** (前10条):
```
{''.join(lines[:11]).strip()}
```

### 📁 生成的文件
- 📄 result/AMPpred.tsv (预测结果)
'''
                
                self.ui_state.add_ui_update('prediction_result', 'set_content', result_text)
                self.ui_state.add_ui_update('download_button', 'set_visibility', True)
                
                self.ui_state.add_message(f'✅ 预测完成！处理了 {total_count} 个序列', 'positive')
            else:
                self.ui_state.add_ui_update('prediction_result', 'set_content', '❌ 预测结果文件未生成')
                self.ui_state.add_message('预测结果文件未生成', 'negative')
                
        except Exception as e:
            logger.error(f"预测结果处理失败: {e}")
            self.ui_state.add_ui_update('prediction_result', 'set_content', f'❌ 结果处理失败: {str(e)}')
    
    def _download_results(self):
        """下载预测结果"""
        result_dir = "result"
        prediction_file = os.path.join(result_dir, 'AMPpred.tsv')
        
        if os.path.exists(prediction_file):
            ui.download(prediction_file)
            logger.info("用户下载预测结果")
        else:
            logger.warning("预测结果文件不存在")


def main():
    """主函数"""
    # 初始化应用
    app = AMPmlAppOptimized()
    
    # 运行服务器
    ui.run(
        title='AMPml - 抗菌肽预测软件 (优化版)',
        port=8081,
        reload=False,
        show=True,
        dark=False,
        favicon='🧬'
    )


if __name__ == '__main__':
    main()
