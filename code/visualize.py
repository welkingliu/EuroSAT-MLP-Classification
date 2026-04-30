import matplotlib.pyplot as plt
import numpy as np
from model import MLPClassifier
from dataloader import EuroSATDataLoader
from eval import load_best_model
import os

def visualize_weights(model, class_names, num_filters=10):
    """
    改良版权重可视化：增加对比度拉伸与色彩增强
    """
    os.makedirs('plots', exist_ok=True)
    W1 = model.params['W1']  # (12288, hidden_dim)
    img_shape = (64, 64, 3)
    
    # 随机选择神经元以避免只看到训练初期的静态神经元
    indices = np.random.choice(W1.shape[1], num_filters, replace=False)
    
    plt.figure(figsize=(15, 6))
    plt.suptitle("Enhanced First Layer Weight Visualization", fontsize=16)
    
    for i, idx in enumerate(indices):
        # 1. 提取原始权重
        w_raw = W1[:, i].reshape(img_shape)
        
        # 2. 计算当前权重的均值和标准差
        w_mean = np.mean(w_raw)
        w_std = np.std(w_raw)
        
        # 3. 强行拉伸：将权重限制在均值左右 3 个标准差内，然后归一化
        w_clipped = np.clip(w_raw, w_mean - 3*w_std, w_mean + 3*w_std)
        w_norm = (w_clipped - w_clipped.min()) / (w_clipped.max() - w_clipped.min() + 1e-12)
        
        plt.subplot(2, 5, i + 1)
        plt.imshow(w_norm)
        plt.title(f"Neuron {idx}", fontsize=12, pad=8)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("plots/weight_visualization_enhanced.png", dpi=300)

def error_analysis(model, test_data, class_names, num_errors=5):
    os.makedirs('plots', exist_ok=True)

    X_test, y_test = test_data
    probs = model.forward(X_test)
    y_pred = np.argmax(probs, axis=1)
    
    error_indices = np.where(y_pred != y_test)[0]
    
    plt.figure(figsize=(15, 5))
    for i in range(min(num_errors, len(error_indices))):
        idx = error_indices[i]
        
        img = X_test[idx].reshape(64, 64, 3)
        
        img = (img - img.min()) / (img.max() - img.min())
        
        plt.subplot(1, num_errors, i + 1)
        plt.imshow(img)
        plt.title(f"True: {class_names[y_test[idx]]}\nPred: {class_names[y_pred[idx]]}", fontsize=10)
        plt.axis('off')
        
    plt.tight_layout()
    plt.savefig("plots/error_analysis.png")
    plt.show()

if __name__ == "__main__":
    

    # 1. 配置基础参数
    DATA_DIR = "./EuroSAT_RGB" 
    WEIGHT_PATH = "final_best_model.pkl"
    HIDDEN_DIM = 1024 
    CLASSES = [
        'AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 
        'Industrial', 'Pasture', 'PermanentCrop', 'Residential', 
        'River', 'SeaLake'
    ]

    # 2. 环境准备
    if not os.path.exists(WEIGHT_PATH):
        print(f"错误: 找不到权重文件 {WEIGHT_PATH}。")
    else:
        # 3. 加载数据以获取测试集
        loader = EuroSATDataLoader(DATA_DIR)
        loader.load_data()
        (X_train_raw, _), _, (X_test_raw, y_test_raw) = loader.split_data()

        # 计算训练集统计量用于标准化 (模拟 main.py 逻辑)
        mean = np.mean(X_train_raw, axis=0)
        std = np.std(X_train_raw, axis=0)
        std[std == 0] = 1.0
        
        X_test = (X_test_raw - mean) / std

        # 4. 初始化模型并加载权重
        input_dim = 64 * 64 * 3
        model = MLPClassifier(input_dim=input_dim, hidden_dim=HIDDEN_DIM, output_dim=10)
        
        load_best_model(model, WEIGHT_PATH)

        # 5. 执行可视化指令
        visualize_weights(model, CLASSES) 

        error_analysis(model, (X_test, y_test_raw), CLASSES)
