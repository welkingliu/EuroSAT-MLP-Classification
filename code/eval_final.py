import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from model import MLPClassifier
from dataloader import EuroSATDataLoader
from eval import run_test_evaluation, calculate_confusion_matrix, print_confusion_matrix

def load_and_preprocess():
    """复用数据加载和标准化逻辑"""
    loader = EuroSATDataLoader("./EuroSAT_RGB")
    loader.load_data()
    (X_train_raw, y_train_raw), _, (X_test_raw, y_test_raw) = loader.split_data()
    
    # 必须使用与训练时完全一致的标准化参数
    mean, std = np.mean(X_train_raw, axis=0), np.std(X_train_raw, axis=0)
    std[std == 0] = 1.0
    X_test = (X_test_raw - mean) / std
    
    return X_test, y_test_raw, loader.classes

def visualize_first_layer_weights(model, num_neurons=10):
    """
    权重可视化：将 W1 映射回图像空间观察模式
    """
    W1 = model.params['W1']  # (12288, hidden_dim)
    plt.figure(figsize=(15, 6))
    plt.suptitle("First Layer Weights Visualization (Spatial & Color Patterns)")
    
    for i in range(num_neurons):
        # 提取神经元权重并恢复为 (64, 64, 3)
        weight_img = W1[:, i].reshape(64, 64, 3)
        # 归一化到 [0, 1] 以便显示
        weight_img = (weight_img - weight_img.min()) / (weight_img.max() - weight_img.min())
        
        plt.subplot(2, 5, i + 1)
        plt.imshow(weight_img)
        plt.title(f"Neuron {i}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def perform_error_analysis(model, X_test, y_test, classes, num_samples=5):
    """
    错例分析：寻找分类错误的地物并分析原因
    """
    probs = model.forward(X_test)
    y_pred = np.argmax(probs, axis=1)
    errors = np.where(y_pred != y_test)[0]
    
    
    plt.figure(figsize=(15, 5))
    for i in range(min(num_samples, len(errors))):
        idx = errors[i]
        img = X_test[idx].reshape(64, 64, 3)
        # 反向缩放回可视化范围
        img = (img - img.min()) / (img.max() - img.min())
        
        plt.subplot(1, num_samples, i + 1)
        plt.imshow(img)
        plt.title(f"True: {classes[y_test[idx]]}\nPred: {classes[y_pred[idx]]}")
        plt.axis('off')
    plt.show()

def main(weight_path='final_best_model.pkl'):
    # 1. 加载数据
    X_test, y_test, classes = load_and_preprocess()
    
    # 2. 初始化模型（需与训练时的 hidden_dim 和 activation 匹配）
    model = MLPClassifier(input_dim=12288, hidden_dim=1024, output_dim=10, activation_type='relu')
    
    # 3. 加载最优权重
    with open(weight_path, 'rb') as f:
        model.params = pickle.load(f)

    # 4. 测试集整体分析 (准确率 & Confusion Table)
    run_test_evaluation(model, (X_test, y_test), classes)

    # 5. 权重空间模式观察
    visualize_first_layer_weights(model)

    # 6. 错例图像分析
    perform_error_analysis(model, X_test, y_test, classes)

if __name__ == "__main__":
    main('final_best_model.pkl')