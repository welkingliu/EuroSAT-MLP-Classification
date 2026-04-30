import numpy as np
import pickle
import matplotlib.pyplot as plt

def load_best_model(model, weight_path='best_model_weights.pkl'):
    """加载保存的最优模型权重 """
    try:
        with open(weight_path, 'rb') as f:
            model.params = pickle.load(f)
    except FileNotFoundError:
        print("Not Found weight PKL file")

def calculate_confusion_matrix(y_true, y_pred, num_classes=10):
    """
    手动实现混淆矩阵计算
    """
    matrix = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        matrix[t, p] += 1
    return matrix

def print_confusion_matrix(matrix, class_names):
    """打印混淆矩阵"""
    header = "       " + " ".join([f"{name[:3]:>5}" for name in class_names])
    print(header)
    for i, row in enumerate(matrix):
        row_str = f"{class_names[i][:6]:>6} " + " ".join([f"{val:5d}" for val in row])
        print(row_str)

def run_test_evaluation(model, test_data, class_names):
    """
    执行完整的测试评估流程
    """
    X_test, y_test = test_data
    
    # 1. 前向传播获取预测概率
    probs = model.forward(X_test)
    
    # 2. 获取预测索引
    y_pred = np.argmax(probs, axis=1)
    
    print(f"预测类别分布: {np.bincount(y_pred, minlength=10)}")
    print(f"前 5 个样本的概率分布: \n{probs[:5]}")
    
    # 3. 计算准确率
    accuracy = np.mean(y_pred == y_test)
    print(f"\n最终测试集准确率 (Accuracy): {accuracy * 100:.2f}%")
    
    # 4. 生成并打印混淆矩阵
    cm = calculate_confusion_matrix(y_test, y_pred, len(class_names))
    print_confusion_matrix(cm, class_names)
    
    return accuracy, cm

# EuroSAT 官方类别顺序 (用于打印混淆矩阵)
EUROSAT_CLASSES = [
    'AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 
    'Industrial', 'Pasture', 'PermanentCrop', 'Residential', 
    'River', 'SeaLake'
]