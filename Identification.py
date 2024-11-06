import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
import os
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. 自定义数据集类，裁剪图片中的刀具区域并保存裁剪后的图片
class KnifeDataset(Dataset):
    def __init__(self, img_dir, label_dir, cropped_save_dir, transform=None):
        """
        初始化数据集。
        
        参数:
        - img_dir: 图像文件夹路径。
        - label_dir: 标签文件夹路径。
        - cropped_save_dir: 裁剪后的图像保存路径。
        - transform: 图像增强转换。
        """
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.cropped_save_dir = cropped_save_dir
        # 找到所有的.jpg图像文件
        self.image_files = [f for f in os.listdir(img_dir) if f.endswith('.jpg')]
        self.transform = transform

        # 如果保存裁剪图片的目录不存在，创建它
        if not os.path.exists(cropped_save_dir):
            os.makedirs(cropped_save_dir)

    def __len__(self):
        # 返回数据集的长度
        return len(self.image_files)

    def __getitem__(self, idx):
        # 获取图像文件名和路径
        img_name = self.image_files[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert("RGB")

        # 获取对应的标签文件路径
        label_name = img_name.replace('.jpg', '.txt')
        label_path = os.path.join(self.label_dir, label_name)

        # 读取标签文件
        with open(label_path, 'r') as f:
            label_data = f.readline().strip().split()
            label = int(label_data[0])  # 第一列为类别标签
            bbox = np.array(label_data[1:], dtype='float')  # 剩余部分为边界框（x_center, y_center, width, height）

        # 计算左上角和右下角坐标（将中心点坐标和宽高转换为裁剪用的坐标）
        img_width, img_height = image.size
        x_center, y_center, width, height = bbox
        x1 = int((x_center - width / 2) * img_width)
        y1 = int((y_center - height / 2) * img_height)
        x2 = int((x_center + width / 2) * img_width)
        y2 = int((y_center + height / 2) * img_height)

        # 裁剪刀具区域
        cropped_image = image.crop((x1, y1, x2, y2))

        # 保存裁剪后的图片到指定目录
        cropped_img_path = os.path.join(self.cropped_save_dir, img_name)
        cropped_image.save(cropped_img_path)

        # 应用转换（如数据增强、缩放等）
        if self.transform:
            cropped_image = self.transform(cropped_image)

        # 返回裁剪后的图像及其标签
        return cropped_image, label

# 2. 图像增强与数据加载器设置
transform = transforms.Compose([
    transforms.Resize((64, 64)),  # 将图像大小调整为64x64
    transforms.ToTensor(),        # 转换为Tensor格式
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 标准化处理
])

# 初始化数据集和数据加载器
dataset = KnifeDataset(
    img_dir='/home/calvin/cutter type/images',              # 图像数据文件夹路径
    label_dir='/home/calvin/cutter type/labels',            # 标签数据文件夹路径
    cropped_save_dir='/home/calvin/cutter type/cropped',    # 裁剪后保存的文件夹路径
    transform=transform
)
data_loader = DataLoader(dataset, batch_size=16, shuffle=True)

# 3. CNN模型定义
class KnifeTypeCNN(nn.Module):
    def __init__(self, num_classes):
        """
        初始化简单的卷积神经网络模型，用于刀型分类。
        
        参数:
        - num_classes: 类别数。
        """
        super(KnifeTypeCNN, self).__init__()
        # 第一个卷积层，输入通道3（RGB），输出通道16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        # 最大池化层，缩小图像尺寸
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # 第二个卷积层，输入16通道，输出32通道
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        # 三卷积层
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        # 全连接层，将32 * 16 * 16个特征转换为128个特征
        self.fc1 = nn.Linear(32 * 16 * 16, 128)
        # 输出层，分类为num_classes种类别
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        # 前向传播：卷积 -> 激活 -> 池化
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        # 展平成适合全连接层的形状
        x = x.view(-1, 32 * 16 * 16)
        # 全连接层 -> 激活
        x = self.relu(self.fc1(x))
        # 输出层
        x = self.fc2(x)
        return x

# 4. 模型、损失函数和优化器设置
num_classes = 5  # 假设有3类刀型
model = KnifeTypeCNN(num_classes=num_classes).to(device)

# 使用交叉熵损失函数和Adam优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 5. 模型训练函数
def train_model(model, data_loader, num_epochs=10):
    """
    训练模型。
    
    参数:
    - model: 待训练的模型。
    - data_loader: 数据加载器。
    - num_epochs: 训练轮数。
    """
    model.train()  # 切换到训练模式
    for epoch in range(num_epochs):
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # 反向传播和优化
            optimizer.zero_grad()  # 梯度清零
            loss.backward()        # 反向传播
            optimizer.step()       # 更新参数

        print(f'第 [{epoch+1}/{num_epochs}] 轮, 损失: {loss.item():.4f}')
    torch.save(model.state_dict(), 'knife_type_cnn.pth')
    print("训练完成。")

# 6. 开始训练模型
train_model(model, data_loader, num_epochs=30)
