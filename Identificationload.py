import torch
from PIL import Image
import torchvision.transforms as transforms
import torch.nn as nn

# 1. 定义 CNN 模型类（与训练时一致）
class KnifeTypeCNN(nn.Module):
    def __init__(self, num_classes):
        super(KnifeTypeCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 32 * 16 * 16)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 2. 加载保存的模型
num_classes = 5  # 之前定义的类别数
model = KnifeTypeCNN(num_classes=num_classes)

# 加载模型权重
model.load_state_dict(torch.load('knife_type_cnn.pth'))
model.eval()  # 切换到评估模式

# 3. 图像预处理
def preprocess_image(image_path):
    """
    对输入图片进行预处理，包括调整大小、转换为Tensor和标准化。

    参数:
    - image_path: 图片路径。

    返回:
    - 处理后的Tensor。
    """
    transform = transforms.Compose([
        transforms.Resize((64, 64)),  # 与训练时的尺寸一致
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    image = Image.open(image_path).convert('RGB')
    image = transform(image)
    image = image.unsqueeze(0)  # 增加batch维度
    return image

# 4. 进行预测
def predict(model, image_path):
    """
    对输入的图像进行分类预测。

    参数:
    - model: 训练好的CNN模型。
    - image_path: 待预测的图像路径。

    返回:
    - 预测类别。
    """
    image_tensor = preprocess_image(image_path)
    with torch.no_grad():  # 关闭梯度计算以加速推理
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)
    return predicted.item()

# 5. 输入图片并进行预测
image_path = '/home/calvin/cutter type/8.png'  # 修改为您想要识别的图像路径
predicted_class = predict(model, image_path)

# 输出结果
print(f'预测的类别为: {predicted_class}')
