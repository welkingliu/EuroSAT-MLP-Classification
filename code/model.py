import numpy as np

class Activation:
    @staticmethod
    def relu(x):
        return np.maximum(0, x)

    @staticmethod
    def relu_derivative(x):
        return (x > 0).astype(float)

    @staticmethod
    def sigmoid(x):
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def sigmoid_derivative(a):
        return a * (1 - a)

    @staticmethod
    def tanh(x):
        return np.tanh(x)

    @staticmethod
    def tanh_derivative(a):
        return 1 - a**2

class MLPClassifier:
    def __init__(self, input_dim, hidden_dim, output_dim, activation_type='relu'):
        """
        三层神经网络初始化
        """
        # 1. 改进的 He 初始化 (Kaiming Init)
        if activation_type == 'relu':
            factor1 = np.sqrt(2.0 / input_dim)
            factor2 = np.sqrt(2.0 / hidden_dim)
        else:
            factor1 = np.sqrt(1.0 / input_dim)
            factor2 = np.sqrt(1.0 / hidden_dim)

        self.params = {
            'W1': np.random.randn(input_dim, hidden_dim) * factor1,
            'b1': np.zeros((1, hidden_dim)),
            'W2': np.random.randn(hidden_dim, output_dim) * factor2,
            'b2': np.zeros((1, output_dim))
        }
        
        self.activation_type = activation_type
        if activation_type == 'relu':
            self.activate = Activation.relu
            self.activate_deriv = Activation.relu_derivative
        elif activation_type == 'sigmoid':
            self.activate = Activation.sigmoid
            self.activate_deriv = Activation.sigmoid_derivative
        elif activation_type == 'tanh':
            self.activate = Activation.tanh
            self.activate_deriv = Activation.tanh_derivative
            
        self.cache = {} 

    def softmax(self, x):
        """数值稳定的 Softmax"""
        # 减去每行的最大值防止 exp(x) 溢出
        shift_x = x - np.max(x, axis=1, keepdims=True)
        exps = np.exp(shift_x)
        return exps / np.sum(exps, axis=1, keepdims=True)

    def forward(self, X):
        """前向传播"""
        # 第一层
        self.cache['Z1'] = np.dot(X, self.params['W1']) + self.params['b1']
        self.cache['A1'] = self.activate(self.cache['Z1'])
        
        # 第二层 (输出层)
        self.cache['Z2'] = np.dot(self.cache['A1'], self.params['W2']) + self.params['b2']
        self.cache['A2'] = self.softmax(self.cache['Z2'])
        
        return self.cache['A2']

    def backward(self, X, y_true, weight_decay=0.0):
        """
        手动实现反向传播计算梯度
        :param weight_decay: L2 正则化强度 (lambda)
        """
        m = X.shape[0] 
        
        # 1. 输出层误差 (Softmax + CrossEntropy)
        dZ2 = self.cache['A2'] - y_true
        
        # 2. 第二层梯度更新 (增加 L2 项梯度: lambda * W)
        dW2 = (np.dot(self.cache['A1'].T, dZ2) + weight_decay * self.params['W2']) / m
        db2 = np.mean(dZ2, axis=0, keepdims=True)
        
        # 3. 误差回传
        dA1 = np.dot(dZ2, self.params['W2'].T)
        
        # 根据不同激活函数计算 Z1 的导数
        if self.activation_type == 'relu':
            dZ1 = dA1 * self.activate_deriv(self.cache['Z1'])
        else:
            # Sigmoid 和 Tanh 的导数通常直接用激活后的 A1 计算更准
            dZ1 = dA1 * self.activate_deriv(self.cache['A1'])
        
        # 4. 第一层梯度更新
        dW1 = (np.dot(X.T, dZ1) + weight_decay * self.params['W1']) / m
        db1 = np.mean(dZ1, axis=0, keepdims=True)
        
        return {'dW1': dW1, 'db1': db1, 'dW2': dW2, 'db2': db2}