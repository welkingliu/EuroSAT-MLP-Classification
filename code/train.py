import numpy as np
import os
import pickle

def cross_entropy_loss(y_pred, y_true, model_params=None, weight_decay=0.0):
    """
    计算数值稳定的交叉熵损失 
    """
    m = y_true.shape[0]
    y_pred = np.clip(y_pred, 1e-12, 1.0 - 1e-12)
    ce_loss = -np.sum(y_true * np.log(y_pred)) / m
    
    # L2 正则化 
    if model_params and weight_decay > 0:
        l2_reg = 0.5 * weight_decay * (np.sum(model_params['W1']**2) + np.sum(model_params['W2']**2))
        ce_loss += (l2_reg / m)
    return ce_loss

def to_one_hot(labels, num_classes=10):
    one_hot = np.zeros((labels.size, num_classes))
    one_hot[np.arange(labels.size), labels] = 1
    return one_hot

def evaluate(model, X, y):
    probs = model.forward(X)
    preds = np.argmax(probs, axis=1)
    return np.mean(preds == y)

def train(model, train_data, val_data, epochs=50, batch_size=32, 
          learning_rate=0.01, weight_decay=0.001, momentum=0.9, 
          patience=10, lr_step=20, lr_gamma=0.5, save_path='best_model_weights.pkl'):

    X_train, y_train = train_data
    X_val, y_val = val_data
    y_train_oh = to_one_hot(y_train)
    
    best_val_acc = 0.0
    no_improve_count = 0
    history = {'train_loss': [], 'train_acc': [], 'val_acc': []}
    
    # 初始化动量缓存 
    velocity = {k: np.zeros_like(v) for k, v in model.params.items()}
    curr_lr = learning_rate

    for epoch in range(epochs):
        # 学习率阶梯衰减 
        if epoch > 0 and epoch % lr_step == 0:
            curr_lr *= lr_gamma

        indices = np.random.permutation(X_train.shape[0])
        X_train_sh = X_train[indices]
        y_train_oh_sh = y_train_oh[indices]
        
        epoch_losses = []
        
        for i in range(0, X_train.shape[0], batch_size):
            X_batch = X_train_sh[i:i+batch_size]
            y_batch_oh = y_train_oh_sh[i:i+batch_size]
            
            # 前向传播
            y_pred = model.forward(X_batch)
            loss = cross_entropy_loss(y_pred, y_batch_oh, model.params, weight_decay)
            epoch_losses.append(loss)
            
            # 反向传播
            grads = model.backward(X_batch, y_batch_oh, weight_decay)
            
            # 使用动量进行 SGD 更新 
            for p in model.params:
                grad = grads[f'd{p}']
                grad = np.clip(grad, -1.0, 1.0)
                
                # v = m * v - lr * grad
                velocity[p] = momentum * velocity[p] - curr_lr * grads[f'd{p}']
                # w = w + v
                model.params[p] += velocity[p]
        
        train_acc = evaluate(model, X_train, y_train)
        val_acc = evaluate(model, X_val, y_val)
        avg_loss = np.mean(epoch_losses)
        
        history['train_loss'].append(avg_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            no_improve_count = 0
            with open(save_path, 'wb') as f: # 使用传入的路径
                pickle.dump(model.params, f)
        else:
            no_improve_count += 1
            
        # if no_improve_count >= patience:
        #    
        #     break
            
    return history