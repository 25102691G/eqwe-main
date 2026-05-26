"""
皮肤分析服务
包含油性检测和缺水程度检测功能
"""

import cv2
import numpy as np
import os
from pathlib import Path

class SkinAnalysisService:
    def __init__(self):
        """初始化皮肤分析服务"""
        pass

    def calculate_oiliness_enhanced(self, image, skin_mask):
        """
        增强版油性计算 - 基于反光检测和局部对比度
        
        油性皮肤特征：
        1. 高反光区域（光泽度高）
        2. 局部亮度突出
        3. 相对于周围区域更亮
        
        Args:
            image: 输入图像
            skin_mask: 皮肤区域掩码
            
        Returns:
            oiliness_map: 油性分布图
            oil_score: 整体油性评分
        """
        # 转换为灰度图
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 应用皮肤掩码
        skin_roi = cv2.bitwise_and(gray, gray, mask=skin_mask)
        
        # 使用高斯滤波减少噪点
        blurred = cv2.GaussianBlur(skin_roi, (5, 5), 1.0)
        
        # 计算皮肤区域的亮度分布
        skin_pixels = blurred[skin_mask > 0]
        if len(skin_pixels) == 0:
            return np.zeros_like(gray, dtype=np.float32), 0
        
        # 计算皮肤亮度的统计信息
        skin_mean = np.mean(skin_pixels)
        skin_std = np.std(skin_pixels)
        skin_median = np.median(skin_pixels)
        
        # 调整油性检测参数
        brightness_threshold = skin_mean + skin_std * 0.1
        print(f"亮度阈值: {brightness_threshold:.1f}, 均值: {skin_mean:.1f}, 标准差: {skin_std:.1f}")
        
        # 创建基础油性图
        oiliness_map = np.zeros_like(blurred, dtype=np.float32)

        # 局部对比度检测
        kernel_size = 15
        kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
        local_mean = cv2.filter2D(blurred.astype(np.float32), -1, kernel)
        
        # 计算每个像素相对于局部均值的亮度比
        local_contrast = blurred.astype(np.float32) - local_mean
        
        # 油性检测条件
        for y in range(blurred.shape[0]):
            for x in range(blurred.shape[1]):
                if skin_mask[y, x] > 0:
                    pixel_brightness = blurred[y, x]
                    local_diff = local_contrast[y, x]
                    
                    oil_intensity = 0
                    
                    if pixel_brightness > brightness_threshold:
                        # 高亮度区域
                        if local_diff > 3:
                            brightness_score = min(100, (pixel_brightness - brightness_threshold) / (255 - brightness_threshold) * 100)
                            contrast_score = min(100, local_diff / 30 * 100)
                            oil_intensity = (brightness_score * 0.8 + contrast_score * 0.2)
                    
                    elif pixel_brightness > skin_median:
                        # 中等亮度区域
                        if local_diff > 8:
                            brightness_score = min(100, (pixel_brightness - skin_median) / (brightness_threshold - skin_median) * 60)
                            contrast_score = min(100, local_diff / 40 * 100)
                            oil_intensity = (brightness_score * 0.5 + contrast_score * 0.5)
                    
                    # 应用油性强度
                    if oil_intensity > 0:
                        oiliness_map[y, x] = oil_intensity
        
        # 形态学处理，去除噪点
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        oiliness_map = cv2.morphologyEx(oiliness_map, cv2.MORPH_OPEN, kernel)
        
        # 高斯滤波平滑结果
        oiliness_map = cv2.GaussianBlur(oiliness_map, (3, 3), 0.5)
        
        # 仅保留皮肤区域
        oiliness_map = oiliness_map * (skin_mask / 255.0)
        
        # 计算整体评分
        oil_pixels = oiliness_map[skin_mask > 0]
        oil_score = np.mean(oil_pixels) if len(oil_pixels) > 0 else 0
        
        # 输出调试信息
        oil_area_pixels = np.sum(oiliness_map > 0)
        total_skin_pixels = np.sum(skin_mask > 0)
        oil_coverage = (oil_area_pixels / total_skin_pixels * 100) if total_skin_pixels > 0 else 0
        
        print(f"🔍 油性检测调试信息:")
        print(f"   油性区域覆盖: {oil_coverage:.1f}% ({oil_area_pixels}/{total_skin_pixels} 像素)")
        print(f"   整体油性评分: {oil_score:.2f}")
        
        return oiliness_map, oil_score

    def detect_oil_regions_hsv(self, image, skin_mask):
        """
        基于HSV/灰度的油脂区域检测（融合原始方法）
        逻辑：高亮度(直方图均衡后的灰度) + 低饱和度，使用自适应阈值与形态学清理
        仅在皮肤掩码内进行。
        
        Args:
            image: 输入图像
            skin_mask: 皮肤区域掩码
            
        Returns:
            oil_mask: np.uint8 二值图，255 表示油脂区域
        """
        # 仅皮肤区域图像
        skin_region = cv2.bitwise_and(image, image, mask=skin_mask)

        # 转换为 HSV 与灰度
        hsv = cv2.cvtColor(skin_region, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(skin_region, cv2.COLOR_BGR2GRAY)

        # 灰度直方图均衡增强亮度对比
        gray_eq = cv2.equalizeHist(gray)

        # 饱和度通道（油脂区域往往饱和度偏低）
        saturation = hsv[:, :, 1]
        sat_inv = cv2.bitwise_not(saturation)

        # 组合特征：高亮 + 低饱和
        oil_likelihood = cv2.addWeighted(gray_eq, 0.7, sat_inv, 0.3, 0)

        # 平滑
        oil_likelihood = cv2.GaussianBlur(oil_likelihood, (5, 5), 0)

        # 自适应阈值生成初始掩码
        # 归一化到8位范围，提升阈值稳定性
        oil_likelihood_8u = cv2.normalize(oil_likelihood, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # 应用皮肤掩码后，取非零像素均值作为固定阈值
        oil_likelihood_8u_masked = cv2.bitwise_and(oil_likelihood_8u, oil_likelihood_8u, mask=skin_mask)
        nonzero_vals = oil_likelihood_8u_masked[oil_likelihood_8u_masked > 0]
        if nonzero_vals.size > 0:
            mu = float(np.mean(nonzero_vals))
            sigma = float(np.std(nonzero_vals))
            k = 0.9  # 可调：0.5~1.0，越大越"保守"，只保留更高亮部分
            fixed_thr = int(np.clip(mu + k * sigma, 0, 255))
        else:
            fixed_thr = 190

        # 使用"均值+标准差"阈值进行固定二值化
        _, oil_mask_fixed = cv2.threshold(oil_likelihood_8u, fixed_thr, 255, cv2.THRESH_BINARY)

        # 调整自适应阈值参数：更大窗口、更正的C，确保高亮区域→白
        oil_mask = cv2.adaptiveThreshold(
            oil_likelihood_8u, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 15, 5
        )

        # 形态学开闭去噪
        kernel = np.ones((3, 3), np.uint8)
        oil_mask = cv2.morphologyEx(oil_mask, cv2.MORPH_OPEN, kernel)
        oil_mask = cv2.morphologyEx(oil_mask, cv2.MORPH_CLOSE, kernel)

        # 只保留皮肤区域
        oil_mask = cv2.bitwise_and(oil_mask, skin_mask)
        
        # 计算油性评分
        # 使用固定阈值的掩码来计算评分
        oil_mask_final = cv2.bitwise_and(oil_mask_fixed, skin_mask)
        
        # 计算油性区域占皮肤区域的比例
        skin_pixels = np.sum(skin_mask > 0)
        oil_pixels = np.sum(oil_mask_final > 0)
        
        if skin_pixels > 0:
            oil_ratio = oil_pixels / skin_pixels
            # 将比例转换为0-100的评分，并应用一些权重调整
            oil_score = min(100, oil_ratio * 300)  # 乘以500是为了放大评分范围
            
            # 添加调试信息
            print(f"🔍 HSV油性检测调试信息:")
            print(f"   油性区域覆盖: {oil_ratio*100:.1f}% ({oil_pixels}/{skin_pixels} 像素)")
            print(f"   HSV油性评分: {oil_score:.2f}")
        else:
            oil_score = 0.0
            print("🔍 HSV油性检测: 未找到有效皮肤区域")

        return oil_mask_fixed, oil_score  # 返回掩码和评分

    def calculate_moisture_enhanced(self, image, skin_mask):
        """
        增强版水分计算 
        基于纹理粗糙度和颜色特征计算干燥程度，然后转换为水分评分
        
        Args:
            image: 输入图像  
            skin_mask: 皮肤区域掩码
            
        Returns:
            moisture_map: 水分分布图 (0-100, 值越高水分越充足)
            moisture_score: 整体水分评分 (0-100)
        """
        # 计算干燥分数图 (0-1, 越大越干)
        dryness_map = self._compute_dryness_score_map(image, skin_mask)
        
        # 转换为水分图 (水分 = 1 - 干燥度)
        moisture_map = (1.0 - dryness_map) * 100.0
        
        # 仅保留皮肤区域
        moisture_map = moisture_map * (skin_mask / 255.0)
        
        # 计算整体评分
        skin_pixels = moisture_map[skin_mask > 0]
        moisture_score = np.mean(skin_pixels) if len(skin_pixels) > 0 else 0
        
        print(f"💧 水分检测调试信息:")
        print(f"   平均干燥度: {np.mean(dryness_map[skin_mask > 0]):.3f}")
        print(f"   水分评分: {moisture_score:.2f}")
        
        return moisture_map, moisture_score

    def _compute_dryness_score_map(self, image, skin_mask):
        """
        计算像素级干燥分数图 (来自Facial_water_deficiency_detection.py)
        
        Args:
            image: 输入图像
            skin_mask: 皮肤区域掩码
            
        Returns:
            score_map: 干燥度分数图 (0-1, 越大越干)
        """
        # 应用皮肤掩码
        masked_image = cv2.bitwise_and(image, image, mask=skin_mask)
        
        # 转换颜色空间
        hsv = cv2.cvtColor(masked_image, cv2.COLOR_BGR2HSV)
        s = hsv[:, :, 1].astype(np.float32) / 255.0
        v = hsv[:, :, 2].astype(np.float32) / 255.0

        gray = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

        # 纹理多特征计算
        # 1) Scharr 梯度幅值（对细纹敏感）
        gx = cv2.Scharr(gray, cv2.CV_32F, 1, 0)
        gy = cv2.Scharr(gray, cv2.CV_32F, 0, 1)
        grad = cv2.magnitude(gx, gy)

        # 2) 多尺度 Laplacian 高频响应
        lap3 = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
        lap5 = cv2.Laplacian(gray, cv2.CV_32F, ksize=5)
        hf = 0.5 * np.abs(lap3) + 0.5 * np.abs(lap5)

        # 3) 局部标准差（纹理波动）
        mu = cv2.GaussianBlur(gray, (0, 0), 1.0)
        mu2 = cv2.GaussianBlur(gray * gray, (0, 0), 1.0)
        local_std = np.sqrt(np.clip(mu2 - mu * mu, 0.0, 1.0))

        # 分位归一化（逐像素图）
        def rnorm_map(x, q=95):
            p = float(np.percentile(x[skin_mask > 0], q)) + 1e-6
            return np.clip(x / p, 0.0, 1.0)

        grad_n = rnorm_map(grad)
        hf_n = rnorm_map(hf)
        std_n = rnorm_map(local_std)

        # 组合纹理粗糙度
        rough = 0.45 * grad_n + 0.4 * hf_n + 0.15 * std_n

        # 最终干燥度评分
        # 权重: 纹理粗糙度0.6 + 低饱和度0.25 + 低亮度0.15
        score = 0.6 * rough + 0.25 * (1.0 - s) + 0.15 * (1.0 - v)
        
        # 平滑处理
        score = cv2.GaussianBlur(score, (0, 0), 1.0)
        score = np.clip(score, 0.0, 1.0)
        
        # 只保留皮肤区域的分数
        score = score * (skin_mask / 255.0)
        
        return score


    def apply_orange_mask(self, image, oil_mask, alpha=0.5):
        """
        将油脂掩码以橙色覆盖到原图上
        
        Args:
            image: 原始图像
            oil_mask: 油脂掩码
            alpha: 透明度
            
        Returns:
            result: 覆盖后的图像
        """
        result = image.copy()
        orange = np.zeros_like(image, dtype=np.uint8)
        orange[oil_mask > 0] = [0, 165, 255]  # BGR 橙色
        cv2.addWeighted(orange, alpha, result, 1 - alpha, 0, result)
        return result

    def create_oil_visualization_enhanced(self, image, oiliness_map, skin_mask):
        """
        创建增强版油性可视化 (多级橙黄色掩码覆盖)
        
        Args:
            image: 原始图像
            oiliness_map: 油性分布图
            skin_mask: 皮肤掩码
            
        Returns:
            result: 可视化结果图像
        """
        result = image.copy()
        
        # 归一化油性图
        normalized_oil = oiliness_map / 100.0
        
        # 多级油性阈值设定
        oil_thresholds = {
            'light': 0.25,    # 25%以上 - 轻度油性
            'moderate': 0.45, # 45%以上 - 中度油性  
            'heavy': 0.65     # 65%以上 - 重度油性
        }
        
        # 多级橙黄色掩码颜色 (BGR格式)
        oil_colors = {
            'light': np.array([0, 200, 255], dtype=np.uint8),    # 浅橙黄色
            'moderate': np.array([0, 165, 255], dtype=np.uint8), # 标准橙黄色
            'heavy': np.array([0, 140, 255], dtype=np.uint8)     # 深橙黄色
        }
        
        # 按油性程度从重到轻应用掩码
        oil_levels = [
            ('heavy', oil_thresholds['heavy'], oil_colors['heavy']),
            ('moderate', oil_thresholds['moderate'], oil_colors['moderate']),
            ('light', oil_thresholds['light'], oil_colors['light'])
        ]
        
        for level_name, threshold, color in oil_levels:
            oil_mask = (normalized_oil > threshold) & (skin_mask > 0)
            
            if np.any(oil_mask):
                # 获取当前级别油性区域坐标
                oil_coords = np.where(oil_mask)
                
                # 根据油性级别和强度调整透明度
                for i in range(len(oil_coords[0])):
                    y, x = oil_coords[0][i], oil_coords[1][i]
                    oil_intensity = normalized_oil[y, x]
                    
                    # 根据级别调整透明度范围
                    if level_name == 'heavy':
                        alpha = np.clip(oil_intensity * 0.8, 0.4, 0.8)  # 重度油性：40%-80%
                    elif level_name == 'moderate':
                        alpha = np.clip(oil_intensity * 0.6, 0.3, 0.6)  # 中度油性：30%-60%
                    else:  # light
                        alpha = np.clip(oil_intensity * 0.4, 0.2, 0.4)  # 轻度油性：20%-40%
                    
                    # 颜色混合
                    orig_pixel = result[y, x].astype(np.float32)
                    overlay_pixel = color.astype(np.float32)
                    
                    blended = orig_pixel * (1 - alpha) + overlay_pixel * alpha
                    result[y, x] = blended.astype(np.uint8)
        
        return result

    def create_moisture_visualization_enhanced(self, image, moisture_map, masks):
        """
        创建增强版水分可视化 - 纯白背景热力图风格
        
        Args:
            image: 原始图像
            moisture_map: 水分分布图  
            masks: 各区域掩码字典
            
        Returns:
            result: 纯白背景上的热力图可视化结果
        """
        h, w = image.shape[:2]
        
        # 创建纯白背景画布
        canvas = np.full((h, w, 3), 255, dtype=np.uint8)
        
        # 获取皮肤掩码
        skin_mask = masks['skin_only']
        
        if np.any(skin_mask > 0):
            # 归一化水分图 (反转，低水分=高缺水程度)
            normalized_moisture = moisture_map / 100.0
            dehydration_level = 1 - normalized_moisture  # 缺水程度 0-1
            
            # 使用参考文件的颜色映射方案 (BGR格式)
            # 按缺水程度分区间着色
            def get_color_for_dehydration(dehydration_score):
                if dehydration_score < 0.4:
                    return np.array([250, 152, 152], dtype=np.uint8)  # 浅色 - 轻微缺水
                elif dehydration_score < 0.55:
                    return np.array([203, 192, 255], dtype=np.uint8)  # 粉紫色 - 轻度缺水
                elif dehydration_score < 0.65:
                    return np.array([226, 43, 138], dtype=np.uint8)   # 紫红色 - 中度缺水
                elif dehydration_score < 0.75:
                    return np.array([235, 206, 135], dtype=np.uint8)  # 浅蓝色 - 重度缺水
                else:
                    return np.array([255, 0, 0], dtype=np.uint8)      # 蓝色 - 极度缺水
            
            # 获取皮肤区域坐标并着色
            skin_coords = np.where(skin_mask > 0)
            
            for i in range(len(skin_coords[0])):
                y, x = skin_coords[0][i], skin_coords[1][i]
                dehydration = dehydration_level[y, x]
                color = get_color_for_dehydration(dehydration)
                canvas[y, x] = color
        
        # 添加颜色图例
        self._draw_moisture_legend(canvas)
        
        return canvas
    
    def _draw_moisture_legend(self, img, origin=(10, 10), size=(220, 20)):
        """
        在图像左上角绘制水分缺失程度的颜色图例
        """
        x0, y0 = origin
        w, h = size
        H, W = img.shape[:2]
        
        if y0 + h > H or x0 + w > W:
            return
        
        # 定义颜色区间 (BGR, 标签)
        legend_items = [
            ((250, 152, 152), "0.4"),
            ((203, 192, 255), "0.55"),
            ((226, 43, 138),  "0.65"),
            ((235, 206, 135), "0.75"),
            ((255, 0, 0),     ">0.75"),
        ]
        
        n = len(legend_items)
        bw = max(1, w // n)
        
        # 绘制颜色条
        for i, (bgr, _) in enumerate(legend_items):
            x1 = x0 + i * bw
            x2 = x0 + (i + 1) * bw if i < n - 1 else x0 + w
            cv2.rectangle(img, (x1, y0), (x2 - 1, y0 + h - 1), bgr, thickness=-1)
        
        # 添加文字标签
        for i, (_, label) in enumerate(legend_items):
            tx = x0 + i * bw + 4
            ty = y0 + h + 16
            cv2.putText(img, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
        
        # 添加标题
        cv2.putText(img, "Dehydration Level", (x0, y0 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)