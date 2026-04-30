import os
import pandas as pd
import itertools
from model import MLPClassifier
from train import train

def grid_search(X_train, y_train, X_val, y_val, input_dim=12288, output_dim=10):
    param_grid = {
        'learning_rate': [0.005, 0.001, 0.0005],
        'hidden_dim': [512, 1024, 2048],
        'weight_decay': [1e-3, 5e-4],
        'activation': ['relu', "tanh", "sigmoid"]
    }
    
    keys, values = zip(*param_grid.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    summary_list = []
    
    # 创建文件夹
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('logs/detailed_curves', exist_ok=True)

    for i, config in enumerate(combinations):
        # 1. 生成基于参数的唯一标识符
        config_id = f"lr{config['learning_rate']}_hd{config['hidden_dim']}_wd{config['weight_decay']}_{config['activation']}"
        
        model = MLPClassifier(input_dim, config['hidden_dim'], output_dim, config['activation'])
        
        # 2. 动态指定权重和日志的文件名
        weight_path = f"checkpoints/best_{config_id}.pkl"
        curve_csv_path = f"logs/detailed_curves/curve_{config_id}.csv"
        
        train_params = {
            'learning_rate': config['learning_rate'],
            'weight_decay': config['weight_decay']
        }

        # 3. 运行训练
        history = train(
            model, (X_train, y_train), (X_val, y_val), 
            epochs=50, 
   
            patience=100, 
            save_path=weight_path,
            **train_params
        )
        
        # 4. 将这一组参数的 Loss 过程保存为独立的 CSV
        epoch_data = {
            'epoch': range(1, len(history['train_loss']) + 1),
            'loss': history['train_loss'],
            'train_acc': history['train_acc'],
            'val_acc': history['val_acc']
        }
        df_curve = pd.DataFrame(epoch_data)
        df_curve.to_csv(curve_csv_path, index=False)
        
        # 5. 记录到汇总表
        summary_entry = config.copy()
        summary_entry.update({
            'config_id': config_id,
            'best_val_acc': max(history['val_acc']),
            'weight_file': weight_path,
            'curve_file': curve_csv_path
        })
        summary_list.append(summary_entry)

    # 保存最终的总表
    pd.DataFrame(summary_list).to_csv('logs/grid_search_summary.csv', index=False)
    
    
    # 找到表现最好的配置
    best_config = max(summary_list, key=lambda x: x['best_val_acc'])
    return best_config, summary_list