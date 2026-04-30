import numpy as np
import os
from dataloader import EuroSATDataLoader, augment_dataset
from model import MLPClassifier
from train import train
from eval import run_test_evaluation, load_best_model
from hyperparameter_search import grid_search
import pandas as pd
from visualize import visualize_weights, error_analysis

def get_zscore_stats(X_train):
    """计算训练集的均值和标准差，用于标准化"""
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)

    std[std == 0] = 1.0
    return mean, std

def run_pipeline(data_path, do_grid_search=False):
    # 1. 数据加载
    loader = EuroSATDataLoader(data_path)
    loader.load_data()
    (X_train_raw, y_train_raw), (X_val_raw, y_val_raw), (X_test_raw, y_test_raw) = loader.split_data()
    
    # 2. Z-Score 标准化 (提升准确率的关键)
    mean, std = get_zscore_stats(X_train_raw)
    X_train = (X_train_raw - mean) / std
    X_val = (X_val_raw - mean) / std
    X_test = (X_test_raw - mean) / std

    print(X_train.max(), X_train.min())

    # 3. 超参数查找
    if do_grid_search:
        best_cfg, summary_list = grid_search(X_train, y_train_raw, X_val, y_val_raw)
        df = pd.DataFrame(summary_list)
        df.to_csv("experiment_results.csv", index=False)
    else:
        
        best_cfg = {
            'learning_rate': 0.0005,
            'hidden_dim': 1024, 
            'weight_decay': 0.001,
            'activation': 'relu'
        }
    
    # 4. 正式训练 

    # 4.5. 数据增强 (仅针对训练集)
    X_train_aug, y_train_aug = augment_dataset(X_train, y_train_raw)
    train_data = (X_train_aug, y_train_aug)
    val_data = (X_val, y_val_raw)
    
    input_dim = 64 * 64 * 3
    final_model = MLPClassifier(
        input_dim=input_dim,
        hidden_dim=best_cfg['hidden_dim'],
        output_dim=10,
        activation_type=best_cfg['activation']
    )
    
    # 使用修改后带动量和早停的 train 函数 
    history = train(
        final_model, 
        train_data, 
        val_data, 
        epochs=50,
        learning_rate=best_cfg['learning_rate'],
        weight_decay=best_cfg['weight_decay'],
        momentum=0.9,
        save_path='final_best_model.pkl'
    )
    
    # 6. 测试集评估与混淆矩阵
    load_best_model(final_model, 'final_best_model.pkl')
    
    classes = ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial', 
               'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake']
               
    run_test_evaluation(final_model, (X_test, y_test_raw), classes)


if __name__ == "__main__":
    DATA_DIR = "./EuroSAT_RGB" 
    run_pipeline(DATA_DIR, do_grid_search=False)
