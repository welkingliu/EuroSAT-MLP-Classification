import os
import numpy as np
from PIL import Image

class EuroSATDataLoader:
    def __init__(self, data_dir, img_size=(64, 64), split_ratio=(0.8, 0.1, 0.1)):
        """
        初始化数据加载器
        :param data_dir: EuroSAT_RGB 文件夹路径
        :param img_size: 缩放后的图像尺寸，默认为 64x64
        :param split_ratio: 训练集、验证集、测试集的比例
        """
        self.data_dir = data_dir
        self.img_size = img_size
        self.split_ratio = split_ratio
        self.classes = sorted(os.listdir(data_dir))  # 10个类别
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        
        self.all_images = []
        self.all_labels = []

    def load_data(self):
        """遍历目录加载图像并进行初步归一化"""
        for cls_name in self.classes:
            cls_path = os.path.join(self.data_dir, cls_name)
            if not os.path.isdir(cls_path): continue
            
            label = self.class_to_idx[cls_name]
            for img_name in os.listdir(cls_path):
                img_path = os.path.join(cls_path, img_name)
                try:
                    # 读取、缩放、转换为 NumPy 数组
                    img = Image.open(img_path).resize(self.img_size)
                    img_array = np.array(img).astype(np.float32)
                    
                    # 归一化到 [0, 1] 区间，有助于模型收敛 
                    img_array /= 255.0
                    
                    # 展平图像: (64, 64, 3) -> (12288,)，适配 MLP 输入层
                    self.all_images.append(img_array.flatten())
                    self.all_labels.append(label)
                except Exception as e:
                    print(f"Fail to load {img_path}: {e}")

        self.all_images = np.array(self.all_images)
        self.all_labels = np.array(self.all_labels)
        print(f"Sample #: {len(self.all_images)}")

    def split_data(self):
        num_samples = len(self.all_images)
        indices = np.random.permutation(num_samples) # 随机打乱
        
        train_end = int(num_samples * self.split_ratio[0])
        val_end = train_end + int(num_samples * self.split_ratio[1])
        
        train_idx, val_idx, test_idx = indices[:train_end], indices[train_end:val_end], indices[val_end:]
        
        return (self.all_images[train_idx], self.all_labels[train_idx]), \
               (self.all_images[val_idx], self.all_labels[val_idx]), \
               (self.all_images[test_idx], self.all_labels[test_idx])

class DataAugmentor:
    """使用 NumPy 实现基础图像增强"""
    
    @staticmethod
    def horizontal_flip(image_vec, img_size=(64, 64, 3)):
        img = image_vec.reshape(img_size)
        flipped = np.flip(img, axis=1)
        return flipped.flatten()

    @staticmethod
    def vertical_flip(image_vec, img_size=(64, 64, 3)):
        """增加垂直翻转"""
        img = image_vec.reshape(img_size)
        flipped = np.flip(img, axis=0)
        return flipped.flatten()

    @staticmethod
    def rotate(image_vec, k=1, img_size=(64, 64, 3)):
        """旋转 k*90 度"""
        img = image_vec.reshape(img_size)
        rotated = np.rot90(img, k=k, axes=(0, 1))
        return rotated.flatten()

def augment_dataset(X_train, y_train):
    """
    对训练集进行扩充：将原始数据通过旋转和翻转扩充至 6 倍 
    """
    aug_X = []
    aug_y = []
    
    print(">>> 正在进行 6 倍数据增强 (旋转+翻转)...")
    for i in range(len(X_train)):
        img = X_train[i]
        label = y_train[i]
        
        # 1. 原始图像
        aug_X.append(img)
        aug_y.append(label)
        
        # 2. 水平翻转
        aug_X.append(DataAugmentor.horizontal_flip(img))
        aug_y.append(label)

        # 3. 垂直翻转
        aug_X.append(DataAugmentor.vertical_flip(img))
        aug_y.append(label)
        
        # 4. 旋转 90, 180, 270 度
        for k in [1, 2, 3]:
            aug_X.append(DataAugmentor.rotate(img, k=k))
            aug_y.append(label)
        
    return np.array(aug_X), np.array(aug_y)

def get_batches(X, y, batch_size, shuffle=True):
    """
    生成小批量数据的迭代器
    """
    n_samples = X.shape[0]
    if shuffle:
        indices = np.random.permutation(n_samples)
        X, y = X[indices], y[indices]
    
    for i in range(0, n_samples, batch_size):
        yield X[i:i + batch_size], y[i:i + batch_size]