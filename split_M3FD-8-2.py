import os
import shutil
import glob
from sklearn.model_selection import train_test_split

def create_dataset(rgb_images_path, ir_images_path, labels_path, train_ratio=0.8):
    # 获取所有可见光图像的路径
    rgb_images = glob.glob(os.path.join(rgb_images_path, '*.png'))
    
    # 按照8:2的比例随机划分训练集和验证集
    train_rgb, val_rgb = train_test_split(rgb_images, test_size=1-train_ratio, random_state=42)
    
    # 为训练集和验证集创建目录
    train_rgb_path = rgb_images_path.replace('all', 'train_8_2')
    val_rgb_path = rgb_images_path.replace('all', 'val_8_2')
    train_ir_path = ir_images_path.replace('all', 'train_8_2')
    val_ir_path = ir_images_path.replace('all', 'val_8_2')
    train_labels_path = labels_path.replace('all', 'train_8_2')
    val_labels_path = labels_path.replace('all', 'val_8_2')
    
    for path in [train_rgb_path, val_rgb_path, train_ir_path, val_ir_path, train_labels_path, val_labels_path]:
        os.makedirs(path, exist_ok=True)
    
    # 复制图像文件和对应的标签文件到相应的目录中
    for file in train_rgb + val_rgb:
        # 根据文件路径判断属于训练集还是验证集
        dest_path = train_rgb_path if file in train_rgb else val_rgb_path
        shutil.copy(file, dest_path)
        
        # 获取文件名，用于查找对应的红外图像和标签文件
        filename = os.path.basename(file)
        base_filename = os.path.splitext(filename)[0]
        
        # 将对应的红外图像复制到训练集或验证集
        ir_dest_path = train_ir_path if file in train_rgb else val_ir_path
        shutil.copy(os.path.join(ir_images_path, filename), ir_dest_path)
        
        # 将对应的标签文件复制到训练集或验证集的标签目录
        label_dest_path = train_labels_path if file in train_rgb else val_labels_path
        shutil.copy(os.path.join(labels_path, base_filename + '.txt'), label_dest_path)

        print("Process ", file)

# 假设的文件路径
rgb_images_path = "/home/disk0/zyq/datasets/M3FD/RGBimages/all"
ir_images_path = "/home/disk0/zyq/datasets/M3FD/IRimages/all"
labels_path = "/home/disk0/zyq/datasets/M3FD/labels/all"

# 运行函数
create_dataset(rgb_images_path, ir_images_path, labels_path)
