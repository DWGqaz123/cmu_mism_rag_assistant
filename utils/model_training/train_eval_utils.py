# utils/model_training/train_eval_utils.py

import torch
import numpy as np
import random
import os
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader

import re 

from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# 新增导入：BertForSequenceClassification
from transformers import BertForSequenceClassification, BertTokenizer # 假设也会用到tokenizer

def set_seed(seed_value: int):
    """
    固定所有随机性，以确保实验的可复现性。
    Args:
        seed_value (int): 用于设置随机数的种子值。
    """
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.backends.mps.is_available(): # For Apple Silicon MPS
        torch.mps.manual_seed(seed_value)
    elif torch.cuda.is_available(): # For NVIDIA GPUs
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        # 确保CUDA操作是确定性的，可能会略微降低性能
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"Random seed set to {seed_value}")

# --- 通用训练和评估函数 ---

def train_model(model: nn.Module, 
                train_dataloader: DataLoader, 
                criterion: nn.Module, # 这个criterion实际上在HuggingFace模型中未直接使用，但保留接口
                optimizer: optim.Optimizer, 
                device: torch.device, 
                num_epochs: int, 
                scheduler: _LRScheduler = None,
                clip_grad_norm_value: float = None, 
                log_interval: int = 100): 
    """
    执行模型的训练循环。
    Args:
        model (nn.Module): 要训练的模型。
        train_dataloader (DataLoader): 训练数据加载器。
        criterion (nn.Module): 损失函数。
        optimizer (optim.Optimizer): 优化器。
        device (torch.device): 训练设备 (CPU/MPS/CUDA)。
        num_epochs (int): 训练的轮次。
        scheduler (_LRScheduler, optional): 学习率调度器。默认为 None。
        clip_grad_norm_value (float, optional): 梯度裁剪的L2范数阈值。默认为 None (不裁剪)。
        log_interval (int): 每隔多少个批次打印一次训练日志。
    Returns:
        tuple: (list of float, list of float) 记录每个epoch的平均训练损失和训练准确率。
    """
    train_losses = []
    train_accuracies = []

    print("\n--- Starting Training ---")
    for epoch in range(num_epochs):
        model.train() # 设置模型为训练模式
        
        # 修正1: 在每个Epoch开始时初始化这些累加器变量
        total_train_loss = 0.0
        correct_predictions = 0
        total_train_samples = 0

        for batch_idx, batch_data in enumerate(train_dataloader):
            input_ids = batch_data['input_ids'].to(device)
            attention_mask = batch_data['attention_mask'].to(device)
            labels = batch_data['labels'].to(device)

            optimizer.zero_grad()

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            total_train_loss += loss.item() # 累加的是批次损失项，不是总损失

            loss.backward()
            
            if clip_grad_norm_value is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm_value) 

            optimizer.step()
            
            if scheduler:
                scheduler.step()

            logits = outputs.logits
            _, predicted = torch.max(logits, dim=1)
            correct_predictions += (predicted == labels).sum().item()
            total_train_samples += labels.size(0)

            if batch_idx % log_interval == 0 and batch_idx != 0:
                print(f"  Epoch {epoch+1}, Batch {batch_idx}/{len(train_dataloader)}, Loss: {loss.item():.4f}")

        # 修正2: 计算Epoch的平均损失和准确率
        avg_train_loss = total_train_loss / len(train_dataloader) # 平均每个批次的损失
        train_accuracy = correct_predictions / total_train_samples * 100
        
        current_lr = optimizer.param_groups[0]['lr'] 
        print(f"Epoch {epoch+1} completed. Avg Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, Current LR: {current_lr:.6f}")
        
        train_losses.append(avg_train_loss)
        train_accuracies.append(train_accuracy)
    
    print("--- Training Finished ---")
    return train_losses, train_accuracies


def evaluate_model(model: nn.Module, 
                   test_dataloader: DataLoader, 
                   device: torch.device, 
                   metrics_info: dict = None): 
    """
    评估模型在测试集上的性能，并计算多维度指标。
    Args:
        model (nn.Module): 要评估的模型。
        test_dataloader (DataLoader): 测试数据加载器。
        device (torch.device): 评估设备 (CPU/MPS/CUDA)。
        metrics_info (dict, optional): 字典，包含 'id2label' 映射等信息，用于更详细的报告。
    Returns:
        tuple: (float, float, float, float, dict) Avg Test Loss, Accuracy, Precision, Recall, F1-Score, Confusion Matrix.
    """
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd

    print("\n--- Starting Evaluation ---")
    model.eval() 
    all_labels = []
    all_preds = []
    total_eval_loss = 0

    with torch.no_grad():
        for batch_data in test_dataloader:
            input_ids = batch_data['input_ids'].to(device)
            attention_mask = batch_data['attention_mask'].to(device)
            labels = batch_data['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits

            total_eval_loss += loss.item()

            _, predicted = torch.max(logits, dim=1)
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    avg_eval_loss = total_eval_loss / len(test_dataloader)

    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary', pos_label=1, zero_division=0)

    print(f"Evaluation completed. Avg Test Loss: {avg_eval_loss:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print("--- Evaluation Finished ---")

    cm = confusion_matrix(all_labels, all_preds)
    
    class_names = ['Negative', 'Positive'] 
    if metrics_info and 'id2label' in metrics_info:
        class_names = [metrics_info['id2label'][i] for i in sorted(metrics_info['id2label'].keys())]

    cm_df = pd.DataFrame(cm, index=[f'Actual {name}' for name in class_names], 
                         columns=[f'Predicted {name}' for name in class_names])

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

    return avg_eval_loss, accuracy, precision, recall, f1, cm


# --- 模型保存与加载的通用函数 ---

def save_model(model: nn.Module, save_path: str, tokenizer=None):
    """
    保存模型的 state_dict。
    Args:
        model (nn.Module): 要保存的模型。
        save_path (str): 模型保存的目录路径。
        tokenizer (transformers.PreTrainedTokenizer, optional): 如果是Hugging Face模型，同时保存分词器。
    """
    os.makedirs(save_path, exist_ok=True)
    # 修正：直接判断模型是否具有 save_pretrained 方法
    if hasattr(model, 'save_pretrained'): # 适用于Hugging Face模型
        model.save_pretrained(save_path) 
    else: # PyTorch原生模型
        torch.save(model.state_dict(), os.path.join(save_path, 'model.pth'))
    
    if tokenizer: # 保存分词器
        tokenizer.save_pretrained(save_path)
    print(f"\nModel and tokenizer (if provided) saved to {save_path}")


def load_model(model_class, load_path: str, device: torch.device, tokenizer_class=None):
    """
    加载模型。支持PyTorch原生模型和Hugging Face模型。
    Args:
        model_class: 模型的类定义 (e.g., SimpleCNN, BERTClassifier, or BertForSequenceClassification directly).
                     如果是Hugging Face模型，直接传入 BertForSequenceClassification。
        load_path (str): 模型加载的目录路径。
        device (torch.device): 模型加载到的设备。
        tokenizer_class (transformers.PreTrainedTokenizer, optional): Hugging Face分词器的类定义，如果需要加载分词器。
    Returns:
        nn.Module: 加载后的模型实例。
        transformers.PreTrainedTokenizer: 如果提供了tokenizer_class，返回加载后的分词器实例。
    """
    # 修正：更通用地判断是否是Hugging Face模型，并直接使用其from_pretrained
    # 假设load_path中包含Hugging Face模型保存的结构
    if os.path.exists(os.path.join(load_path, 'config.json')) and \
       os.path.exists(os.path.join(load_path, 'pytorch_model.bin')): # Hugging Face 模型保存的典型文件
        print(f"Loading Hugging Face model from {load_path}")
        model = model_class.from_pretrained(load_path) # 直接使用模型类自带的from_pretrained
        if tokenizer_class:
            tokenizer = tokenizer_class.from_pretrained(load_path)
        else:
            tokenizer = None
    else: # PyTorch原生模型
        print(f"Loading PyTorch native model from {os.path.join(load_path, 'model.pth')}")
        model = model_class() # 实例化模型
        model.load_state_dict(torch.load(os.path.join(load_path, 'model.pth'), map_location=device)) # map_location确保加载到正确设备
        tokenizer = None

    model = model.to(device)
    model.eval() # 加载后通常立即设置为评估模式
    print(f"\nModel loaded successfully from {load_path}")
    
    if tokenizer_class:
        return model, tokenizer
    return model