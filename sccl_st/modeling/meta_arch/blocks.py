import torch
from torch import nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    """
    基础卷积块: Conv + BatchNorm + ReLU
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

############### Domain Specific Encoder ##############
class DomainSpecificEncoder(nn.Module):
    """
    Domain-specific feature encoder
    包含4个卷积块,将输入特征(如res2)下采样到目标尺寸(如res4)
    """
    def __init__(self, in_channels, out_channels):
        """
        Args:
            in_channels: 输入特征的通道数(从ds_hook层,如res2)
            out_channels: 输出特征的通道数(你指定的)
            target_spatial_size: 目标空间尺寸 (H, W),与dis_type层相同
                如果为None,则在forward时动态匹配
        """
        super(DomainSpecificEncoder, self).__init__()
        
        # 设计4个卷积块的通道数,逐步过渡到out_channels
        # 可以根据实际需求调整
        hidden_dim1 = max(in_channels, out_channels // 2)
        hidden_dim2 = max(in_channels, out_channels * 3 // 4)
        hidden_dim3 = out_channels
        
        # 4个卷积块
        # 在适当位置使用stride=2进行下采样
        # 例如: res2 -> res4 需要下采样2次 (1/4 spatial size)
        self.conv1 = ConvBlock(in_channels, hidden_dim1, kernel_size=3, stride=2, padding=1)  # 下采样
        self.conv2 = ConvBlock(hidden_dim1, hidden_dim2, kernel_size=3, stride=1, padding=1)
        self.conv3 = ConvBlock(hidden_dim2, hidden_dim3, kernel_size=3, stride=2, padding=1)  # 下采样
        self.conv4 = ConvBlock(hidden_dim3, out_channels, kernel_size=3, stride=1, padding=1)
        
        self.out_channels = out_channels
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x, target_size):
        """
        Args:
            x: shape [B, in_channels, H_in, W_in] (如res2的特征)
            target_size: 目标空间尺寸 (H_target, W_target),如果提供则覆盖初始化时的设置
        Returns:
            output: shape [B, out_channels, H_target, W_target] (与dis_type层空间尺寸相同)
        """
        # 通过4个卷积块
        x = self.conv1(x)  # stride=2, 空间尺寸 /2
        x = self.conv2(x)  # stride=1, 空间尺寸不变
        x = self.conv3(x)  # stride=2, 空间尺寸 /2 (总共 /4)
        x = self.conv4(x)  # stride=1, 空间尺寸不变
        
        # 如果需要精确匹配目标尺寸,使用插值
        final_target_size = target_size
        if final_target_size is not None and x.shape[2:] != final_target_size:
            x = F.interpolate(x, size=final_target_size, mode='bilinear', align_corners=False)
        
        return x  # [B, out_channels, H_target, W_target]
#################################

############### Image discriminator ##############
class shuffle_classifier(nn.Module):
    def __init__(self, num_classes, ndf1=256, ndf2=128):
        super(shuffle_classifier, self).__init__()

        self.conv1 = nn.Conv2d(num_classes, ndf1, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(ndf1, ndf2, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(ndf2, ndf2, kernel_size=3, padding=1)
        self.classifier = nn.Conv2d(ndf2, 1, kernel_size=3, padding=1)

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.leaky_relu(x)
        x = self.conv2(x)
        x = self.leaky_relu(x)
        x = self.conv3(x)
        x = self.leaky_relu(x)
        x = self.classifier(x)
        return x
#################################

class ProjectionHead1(nn.Module):
    """
    第一个projection head: 两层全连接 + ReLU
    将Gram matrix映射为64维style向量
    """
    def __init__(self, input_dim, hidden_dim=256, output_dim=64):
        """
        Args:
            input_dim: Gram matrix展平后的维度 (C*C,其中C是特征通道数)
            hidden_dim: 隐藏层维度
            output_dim: 输出style向量维度,默认64
        """
        super(ProjectionHead1, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu2 = nn.ReLU(inplace=True)
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
    def forward(self, gram_matrix):
        """
        Args:
            gram_matrix: shape [B, C, C] 或 [B, C*C]
        Returns:
            style_vector: shape [B, 64]
        """
        # 如果是 [B, C, C] 格式,展平为 [B, C*C]
        if gram_matrix.dim() == 3:
            B, C, _ = gram_matrix.shape
            gram_matrix = gram_matrix.view(B, C * C)
        
        x = self.fc1(gram_matrix)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        return x


class ProjectionHead2(nn.Module):
    """
    第二个projection head: 三层全连接 + ReLU
    将Gram matrix映射为64维style向量
    """
    def __init__(self, input_dim, hidden_dim1=512, hidden_dim2=256, output_dim=64):
        """
        Args:
            input_dim: Gram matrix展平后的维度 (C*C)
            hidden_dim1: 第一个隐藏层维度
            hidden_dim2: 第二个隐藏层维度
            output_dim: 输出style向量维度,默认64
        """
        super(ProjectionHead2, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.relu1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.relu2 = nn.ReLU(inplace=True)
        self.fc3 = nn.Linear(hidden_dim2, output_dim)
        self.relu3 = nn.ReLU(inplace=True)
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
    def forward(self, gram_matrix):
        """
        Args:
            gram_matrix: shape [B, C, C] 或 [B, C*C]
        Returns:
            style_vector: shape [B, 64]
        """
        # 如果是 [B, C, C] 格式,展平为 [B, C*C]
        if gram_matrix.dim() == 3:
            B, C, _ = gram_matrix.shape
            gram_matrix = gram_matrix.view(B, C * C)
        
        x = self.fc1(gram_matrix)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.relu3(x)
        return x
    
class ResidualBlock(nn.Module):
    """
    残差块: Conv-BN-ReLU-Conv-BN + skip connection
    """
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
    
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += identity  # skip connection
        out = self.relu(out)
        
        return out


class SharedDecoder(nn.Module):
    """
    共享解码器: 2个残差块 + 2个反卷积层
    将concat后的特征重建回hook层的原始特征
    """
    def __init__(self, in_channels, out_channels, upsample_factor=4):
        """
        Args:
            in_channels: 输入通道数 (dis_type_channels + ds_out_channels)
            out_channels: 输出通道数 (ds_hook层的通道数,即要重建的特征通道数)
            upsample_factor: 上采样倍数 (例如从res4到res2是4倍)
                2: 1个stride=2的反卷积
                4: 2个stride=2的反卷积
                8: 3个stride=2的反卷积
        """
        super(SharedDecoder, self).__init__()
        
        # 计算需要几次stride=2的反卷积
        import math
        num_deconv = int(math.log2(upsample_factor))
        assert 2 ** num_deconv == upsample_factor, "upsample_factor must be power of 2"
        self.num_deconv = num_deconv
        
        # 中间通道数设计
        # 从in_channels逐步过渡到out_channels
        if num_deconv == 2:
            # 需要2个反卷积层
            mid_channels = (in_channels + out_channels) // 2
            deconv_channels = [in_channels, mid_channels, out_channels]
        elif num_deconv == 1:
            # 需要1个反卷积层,但论文说有2个,所以第二个不改变空间尺寸
            mid_channels = (in_channels + out_channels) // 2
            deconv_channels = [in_channels, mid_channels, out_channels]
        elif num_deconv == 3:
            # 需要3个反卷积层,但论文说有2个,可能需要调整
            mid_channels1 = in_channels * 3 // 4
            mid_channels2 = in_channels // 2
            deconv_channels = [in_channels, mid_channels1, mid_channels2, out_channels]
        else:
            # 默认情况
            deconv_channels = [in_channels, out_channels]
        
        # 2个残差块 (在反卷积之前)
        self.res_block1 = ResidualBlock(in_channels)
        self.res_block2 = ResidualBlock(in_channels)
        
        # 2个反卷积层
        # 根据upsample_factor决定stride
        if num_deconv == 3:
            self.deconv1 = nn.ConvTranspose2d(
                deconv_channels[0], 
                deconv_channels[1], 
                kernel_size=4, 
                stride=2, 
                padding=1, 
                bias=False
            )
            self.bn1 = nn.BatchNorm2d(deconv_channels[1])
            self.relu1 = nn.ReLU(inplace=True)
            
            self.deconv2 = nn.ConvTranspose2d(
                deconv_channels[1], 
                deconv_channels[2], 
                kernel_size=4, 
                stride=2, 
                padding=1, 
                bias=False
            )
            self.bn2 = nn.BatchNorm2d(deconv_channels[2])
            self.relu2 = nn.ReLU(inplace=True)
            
            self.deconv3 = nn.ConvTranspose2d(
                deconv_channels[2], 
                deconv_channels[3], 
                kernel_size=4, 
                stride=2, 
                padding=1, 
                bias=False
            )
            self.bn3 = nn.BatchNorm2d(deconv_channels[3])
            self.relu3 = nn.ReLU(inplace=True)
                
        elif num_deconv == 2:
            # 两个都是stride=2
            self.deconv1 = nn.ConvTranspose2d(
                deconv_channels[0], 
                deconv_channels[1], 
                kernel_size=4, 
                stride=2, 
                padding=1, 
                bias=False
            )
            self.bn1 = nn.BatchNorm2d(deconv_channels[1])
            self.relu1 = nn.ReLU(inplace=True)
            
            self.deconv2 = nn.ConvTranspose2d(
                deconv_channels[1], 
                deconv_channels[2], 
                kernel_size=4, 
                stride=2, 
                padding=1, 
                bias=False
            )
            self.bn2 = nn.BatchNorm2d(deconv_channels[2])
            self.relu2 = nn.ReLU(inplace=True)
            
        elif num_deconv == 1:
            # 第一个stride=2,第二个stride=1
            self.deconv1 = nn.ConvTranspose2d(
                deconv_channels[0], 
                deconv_channels[1], 
                kernel_size=4, 
                stride=2, 
                padding=1, 
                bias=False
            )
            self.bn1 = nn.BatchNorm2d(deconv_channels[1])
            self.relu1 = nn.ReLU(inplace=True)
            
            self.deconv2 = nn.ConvTranspose2d(
                deconv_channels[1], 
                deconv_channels[2], 
                kernel_size=3, 
                stride=1, 
                padding=1, 
                bias=False
            )
            self.bn2 = nn.BatchNorm2d(deconv_channels[2])
            self.relu2 = nn.ReLU(inplace=True)
        
        self.out_channels = out_channels
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x, target_size=None):
        """
        Args:
            x: concat后的特征 [B, in_channels, H, W]
            target_size: 目标空间尺寸 (H_target, W_target),即ds_hook层的空间尺寸
        Returns:
            reconstructed: 重建的特征 [B, out_channels, H_target, W_target]
        """
        # 通过2个残差块
        x = self.res_block1(x)
        x = self.res_block2(x)
        
        # 通过反卷积层
        x = self.deconv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        
        x = self.deconv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        
        if self.num_deconv >= 3:
            x = self.deconv3(x)
            x = self.bn3(x)
            x = self.relu3(x)
        
        # 如果需要精确匹配目标尺寸,使用插值微调
        if target_size is not None and (x.shape[2] != target_size[0] or x.shape[3] != target_size[1]):
            x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
        
        return x
    

class ContrastiveProjector(nn.Module):
    """
    基于MLP的非线性投影器 (one hidden layer)
    将domain-specific feature映射为单位向量用于对比学习
    """
    def __init__(self, input_dim, hidden_dim=2048, output_dim=128):
        """
        Args:
            input_dim: 输入特征维度 (域特定编码器输出的通道数)
            hidden_dim: 隐藏层维度,默认2048
            output_dim: 输出embedding维度,默认128
        """
        super(ContrastiveProjector, self).__init__()
        
        # 1个隐藏层的MLP (2层全连接)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        # 不需要最后的ReLU,因为输出要归一化为单位向量
    
    def forward(self, z):
        """
        Args:
            z: domain-specific feature [B, C, H, W] 或 [B, C, 1, 1] 或 [B, C]
        Returns:
            u: 单位向量 [B, output_dim]
        """
        # 如果输入是特征图,先进行全局平均池化
        if z.dim() == 4:
            z = F.adaptive_avg_pool2d(z, (1, 1))  # [B, C, H, W] -> [B, C, 1, 1]
            z = z.flatten(1)  # [B, C, 1, 1] -> [B, C]
        elif z.dim() == 3:
            z = z.flatten(1)
        
        # 通过MLP
        h = self.fc1(z)      # [B, C] -> [B, hidden_dim]
        h = self.bn(h)
        h = self.relu(h)     # ReLU激活
        h = self.fc2(h)      # [B, hidden_dim] -> [B, output_dim]
        
        # 归一化为单位向量: u = h / ||h||
        u = F.normalize(h, p=2, dim=1)  # L2归一化,每个样本的向量模为1
        
        return u
    