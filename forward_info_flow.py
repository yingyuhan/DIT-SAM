
import torch
from torch import nn
from models.module_lib import LayerNorm2d
import torch.nn.functional as F

def laplacian_enhancement(x, sigma=0.5, alpha=1.0):

    # 1. 高斯平滑降噪（减少拉普拉斯对噪声的敏感）
    kernel_size = int(4 * sigma) + 1  # 计算合适的核尺寸（如sigma=0.5时核为3x3）
    gaussian_kernel = create_gaussian_kernel(kernel_size, sigma, channels=x.shape[0])
    smoothed = F.conv2d(x.unsqueeze(0), gaussian_kernel, padding=kernel_size//2, groups=x.shape[0]).squeeze(0)
    
    # 2. 定义拉普拉斯核（各向同性版本）
    laplacian_kernel = torch.tensor(
        [[[0, -1, 0], [-1, 4, -1], [0, -1, 0]]],  # 形状为 [1, 1, 3, 3]
        dtype=torch.float32,
        device=x.device
    )
    laplacian_kernel = laplacian_kernel.repeat(x.shape[0], 1, 1, 1)  # 扩展到所有通道
    
    # 3. 应用拉普拉斯算子
    laplacian = F.conv2d(smoothed.unsqueeze(0), laplacian_kernel, padding=1, groups=x.shape[0]).squeeze(0)
    
    # 4. 增强微观结构（ 拉普拉斯响应）
    enhanced = alpha * laplacian  # alpha控制增强强度
    
    # 5. 动态范围裁剪（防止像素值越界）
    enhanced = torch.clamp(enhanced, 0.0, 1.0)  # 假设输入已归一化到[0,1]
    
    return enhanced

def create_gaussian_kernel(kernel_size: int, sigma: float, channels: int) -> torch.Tensor:
    # 生成1D高斯核
    x = torch.arange(kernel_size, dtype=torch.float32)
    x = x - (kernel_size - 1) / 2
    gaussian_1d = torch.exp(-(x ** 2) / (2 * sigma ** 2))
    gaussian_1d = gaussian_1d / gaussian_1d.sum()
    # 转换为2D核（可分离卷积）
    gaussian_2d = torch.outer(gaussian_1d, gaussian_1d).unsqueeze(0).unsqueeze(0)
    gaussian_kernel = gaussian_2d.repeat(channels, 1, 1, 1)
    return gaussian_kernel.to('cuda')


### 全的引入梯度的代码
class FeatureFusion(nn.Module):
    def __init__(self):
        super(FeatureFusion, self).__init__()
        self.conv1_shallow = nn.Conv2d(1024, 256, kernel_size=1)
        # self.conv1 = nn.Conv2d(256, 256, kernel_size=1)
        self.conv3 = nn.Conv2d(512, 256, kernel_size=1)
        self.conv2 = nn.Conv2d(2048, 512, kernel_size=1)

    def forward(self, deep_feature, shallow_feature,two_way_information_flow):  # deep_feature[256,64,64],shallow_feature[4,64,64,1024]
        
        def compute_gradient(x):
            assert x.dim() == 3 and x.shape[0] == 256, "输入张量的尺寸必须为 [1, 256, 64, 64]"
            channels, height, width = x.shape
            sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(x.device)
            sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(x.device)
            sobel_x = sobel_x.repeat(channels, 1, 1, 1)
            sobel_y = sobel_y.repeat(channels, 1, 1, 1)
            grad_x = F.conv2d(x, sobel_x, padding=1, groups=channels)
            grad_y = F.conv2d(x, sobel_y, padding=1, groups=channels)
            gradient = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8)  # 添加一个小的正数避免数值不稳定
            return gradient
        
        shallow_feature = shallow_feature.permute(0,3,1,2)
        
        # 1x1卷积调整浅层特征通道数
        shallow_feature_conv_0 = self.conv1_shallow(shallow_feature[0])
        shallow_feature_conv_1 = self.conv1_shallow(shallow_feature[1])
        shallow_feature_conv_2 = self.conv1_shallow(shallow_feature[2])
        shallow_feature_conv_3 = self.conv1_shallow(shallow_feature[3])
        
 
        
        shallow_gradient_0 = compute_gradient(shallow_feature_conv_0)
        shallow_gradient_1 = compute_gradient(shallow_feature_conv_1)
        shallow_gradient_2 = compute_gradient(shallow_feature_conv_2)
        shallow_gradient_3 = compute_gradient(shallow_feature_conv_3)

        laplacian_shallow_0  = laplacian_enhancement(shallow_feature_conv_0, sigma=0.5, alpha=1.0)
        laplacian_shallow_1  = laplacian_enhancement(shallow_feature_conv_1, sigma=0.5, alpha=1.0)
        laplacian_shallow_2  = laplacian_enhancement(shallow_feature_conv_2, sigma=0.5, alpha=1.0)
        laplacian_shallow_3  = laplacian_enhancement(shallow_feature_conv_3, sigma=0.5, alpha=1.0)
        
        # 连接特征和梯度
        shallow_feature_with_grad_0 = torch.cat([shallow_feature_conv_0 , shallow_gradient_0 + laplacian_shallow_0], dim=0)
        shallow_feature_with_grad_1 = torch.cat([shallow_feature_conv_1 , shallow_gradient_1 + laplacian_shallow_1 ], dim=0)
        shallow_feature_with_grad_2 = torch.cat([shallow_feature_conv_2 , shallow_gradient_2 + laplacian_shallow_2 ], dim=0)
        shallow_feature_with_grad_3 = torch.cat([shallow_feature_conv_3 , shallow_gradient_3 + laplacian_shallow_3 ], dim=0)

        # 连接浅层和深层特征 [1,1280,64,64]
        combined_features = torch.cat([shallow_feature_with_grad_0,shallow_feature_with_grad_1,shallow_feature_with_grad_2,shallow_feature_with_grad_3], dim=0)

        # 再次调整通道数
        combined_features = self.conv2(combined_features)  # [1280,64,64]

        # 元素相乘 [1,256,64,64]
        shallow_feature_fused_0 = shallow_feature_with_grad_0 * combined_features
        shallow_feature_fused_1 = shallow_feature_with_grad_1 * combined_features
        shallow_feature_fused_2 = shallow_feature_with_grad_2 * combined_features
        shallow_feature_fused_3 = shallow_feature_with_grad_3 * combined_features

        # 再次连接
        x = torch.cat([shallow_feature_fused_0,shallow_feature_fused_1,shallow_feature_fused_2,shallow_feature_fused_3], dim=0)   # [1280,64,64]
        # 通过3x3卷积块
        x = self.conv2(x) #[256,64,64]
        x = self.conv3(x) #[256,64,64]

        if two_way_information_flow==True:
            return x.unsqueeze(0)
        # else:
        #     return deep_feature.unsqueeze(0)



