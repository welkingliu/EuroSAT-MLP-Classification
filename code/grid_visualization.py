import os
import pandas as pd
import matplotlib.pyplot as plt
import re

def plot_custom_grid(target_lr, target_wd, curve_dir='logs/detailed_curves'):
    # 1. 定义 3x3 的布局结构
    hidden_dims = [512, 1024, 2048]
    activations = ['relu', 'tanh', 'sigmoid']
    
    fig, axes = plt.subplots(3, 3, figsize=(20, 15))
    plt.subplots_adjust(hspace=0.4, wspace=0.3)
    
    # 2. 遍历 3x3 的网格
    for row, hd in enumerate(hidden_dims):
        for col, act in enumerate(activations):
            ax_acc = axes[row, col]
            ax_loss = ax_acc.twinx()  
            
            # lr{lr}_hd{hd}_wd{wd}_{act}.csv
            file_name = f"curve_lr{target_lr}_hd{hd}_wd{target_wd}_{act}.csv"
            file_path = os.path.join(curve_dir, file_name)
            
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                epochs = df['epoch']
                
                # 绘制曲线
                lns2 = ax_acc.plot(epochs, df['train_acc'], color='red', label='Train Acc', linewidth=1.5)
                lns3 = ax_acc.plot(epochs, df['val_acc'], color='gold', label='Val Acc', linewidth=1.5)
                ax_acc.set_ylim(0.2, 1.0)
                ax_acc.set_ylabel('Accuracy', color='black')
                
                lns1 = ax_loss.plot(epochs, df['loss'], color='blue', label='Loss', linewidth=1.5, alpha=0.7)
                ax_loss.set_ylim(0, 17)
                ax_loss.set_ylabel('Loss', color='blue')
                
                # 设置标题
                ax_acc.set_title(f"HD: {hd} | Act: {act.upper()}", fontweight='bold')
                ax_acc.grid(True, linestyle='--', alpha=0.6)
                
                # 合并图例
                lns = lns1 + lns2 + lns3
                labs = [l.get_label() for l in lns]
                ax_acc.legend(lns, labs, loc='center right', fontsize='small')
            else:
                ax_acc.text(0.5, 0.5, f"File Not Found:\n{file_name}", 
                            ha='center', va='center', color='gray')
                ax_acc.set_title(f"HD: {hd} | Act: {act.upper()} (Missing)")

    # 设置行列外侧标签
    for i, hd in enumerate(hidden_dims):
        axes[i, 0].set_ylabel(f"Hidden Dim: {hd}\n\nAccuracy", fontsize=12, fontweight='bold')
    for j, act in enumerate(activations):
        axes[0, j].set_title(f"Activation: {act.upper()}\n", fontsize=12, fontweight='bold')

    plt.suptitle(f"Training Analysis (LR={target_lr}, WD={target_wd})", fontsize=20, y=0.95)
    
    # 保存结果
    save_name = f"grid_plot_lr{target_lr}_wd{target_wd}.png"
    plt.savefig(save_name, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    # 输入参数
    for in_lr in [0.005, 0.001, 0.0005]:
        for in_wd in [0.001, 0.0005]:
            plot_custom_grid(in_lr, in_wd)