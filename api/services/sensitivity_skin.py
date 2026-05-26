"""
皮肤敏感性分析服务
基于红度和纹理特征进行皮肤敏感性评估
"""

import os
import math
from typing import Optional, Tuple, List
import numpy as np
import cv2



# Optional MediaPipe support
try:
    from api.services.mediapipe_compat import create_face_mesh
    MP_AVAILABLE = True
except Exception:
    create_face_mesh = None
    MP_AVAILABLE = False


class SkinSensitivityService:
    """皮肤敏感性分析服务类"""
    
    def __init__(self):
        self.mp_available = MP_AVAILABLE

    def to_bgr(self, img_rgb: np.ndarray) -> np.ndarray:
        """RGB转BGR"""
        return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    def gaussian_smooth01(self, arr01: np.ndarray, ksize: int = 11) -> np.ndarray:
        """高斯平滑处理"""
        if ksize % 2 == 0:
            ksize += 1
        arr = (arr01 * 255).astype(np.uint8)
        arr = cv2.GaussianBlur(arr, (ksize, ksize), 0)
        arr = arr.astype(np.float32) / 255.0
        return np.clip(arr, 0.0, 1.0)

    def normalize01(self, x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        """归一化到0-1范围"""
        vmin = float(np.nanmin(x))
        vmax = float(np.nanmax(x))
        if vmax - vmin < eps:
            return np.zeros_like(x, dtype=np.float32)
        return ((x - vmin) / (vmax - vmin)).astype(np.float32)

    def compute_redness_a(self, img_rgb: np.ndarray) -> np.ndarray:
        """计算红度（使用Lab颜色空间的a*通道）"""
        img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
        a = img_lab[:, :, 1].astype(np.float32)  # 0..255, 128 is neutral
        a_centered = a - 128.0
        a_pos = np.clip(a_centered, 0, None)  # only positive (towards red)
        return self.normalize01(a_pos)

    def compute_texture_laplacian(self, img_rgb: np.ndarray) -> np.ndarray:
        """计算纹理特征（使用拉普拉斯算子）"""
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
        lap_abs = np.abs(lap)
        return self.normalize01(lap_abs)

    def dashed_polyline(self, img: np.ndarray, pts: np.ndarray, color: Tuple[int, int, int],
                        thickness: int = 2, dash_len: int = 12, gap_len: int = 8, closed: bool = True) -> None:
        """绘制虚线多边形"""
        pts = pts.astype(np.int32)
        n = len(pts)
        indices = list(range(n)) + ([0] if closed else list(range(n - 1)))
        for i in range(len(indices) - 1):
            p1 = pts[indices[i]]
            p2 = pts[indices[i + 1]]
            seg = p2 - p1
            seg_len = math.hypot(float(seg[0]), float(seg[1]))
            if seg_len == 0:
                continue
            direction = seg / seg_len
            dist = 0.0
            draw = True
            while dist < seg_len:
                start = p1 + (direction * dist)
                end = p1 + (direction * min(dist + (dash_len if draw else gap_len), seg_len))
                if draw:
                    cv2.line(img,
                             (int(round(start[0])), int(round(start[1]))),
                             (int(round(end[0])), int(round(end[1]))),
                             color, thickness, lineType=cv2.LINE_AA)
                dist += (dash_len if draw else gap_len)
                draw = not draw

    def try_facemesh_landmarks(self, img_rgb: np.ndarray) -> Optional[np.ndarray]:
        """尝试使用MediaPipe获取面部特征点"""
        if not self.mp_available:
            return None
        h, w = img_rgb.shape[:2]
        if create_face_mesh is None:
            return None
        fm = create_face_mesh(static_image_mode=True, refine_landmarks=True, max_num_faces=1)
        if fm is None:
            return None
        with fm:
            result = fm.process(img_rgb)
        if not result.multi_face_landmarks:
            return None
        lm = result.multi_face_landmarks[0]
        pts = []
        for p in lm.landmark:
            pts.append([p.x * w, p.y * h])
        return np.array(pts, dtype=np.float32)  # (468, 2)

    def face_bbox_from_points(self, pts: np.ndarray, img_shape: Tuple[int, int]) -> Tuple[int, int, int, int]:
        """从特征点计算面部边界框"""
        h, w = img_shape[:2]
        x_min = max(0, int(np.min(pts[:, 0])))
        y_min = max(0, int(np.min(pts[:, 1])))
        x_max = min(w - 1, int(np.max(pts[:, 0])))
        y_max = min(h - 1, int(np.max(pts[:, 1])))
        return x_min, y_min, x_max, y_max

    def fallback_bbox(self, img_rgb: np.ndarray) -> Tuple[int, int, int, int]:
        """回退方案：估算面部边界框"""
        h, w = img_rgb.shape[:2]
        cx, cy = w // 2, h // 2
        bw, bh = int(w * 0.5), int(h * 0.6)
        x1 = max(0, cx - bw // 2)
        y1 = max(0, cy - bh // 2)
        x2 = min(w - 1, cx + bw // 2)
        y2 = min(h - 1, cy + bh // 2)
        return x1, y1, x2, y2

    def regions_from_bbox(self, x1: int, y1: int, x2: int, y2: int) -> List[np.ndarray]:
        """从边界框生成面部区域"""
        w = x2 - x1 + 1
        h = y2 - y1 + 1
        # Forehead: top 30%
        fh = int(h * 0.30)
        forehead = np.array([
            [x1 + int(w * 0.08), y1 + int(fh * 0.15)],
            [x1 + int(w * 0.92), y1 + int(fh * 0.15)],
            [x1 + int(w * 0.85), y1 + fh],
            [x1 + int(w * 0.15), y1 + fh],
        ], dtype=np.int32)
        # Left cheek: left lower-mid
        lc = np.array([
            [x1 + int(w * 0.05), y1 + int(h * 0.45)],
            [x1 + int(w * 0.40), y1 + int(h * 0.45)],
            [x1 + int(w * 0.38), y1 + int(h * 0.85)],
            [x1 + int(w * 0.10), y1 + int(h * 0.90)],
        ], dtype=np.int32)
        # Right cheek: right lower-mid
        rc = np.array([
            [x1 + int(w * 0.60), y1 + int(h * 0.45)],
            [x1 + int(w * 0.95), y1 + int(h * 0.45)],
            [x1 + int(w * 0.90), y1 + int(h * 0.90)],
            [x1 + int(w * 0.62), y1 + int(h * 0.85)],
        ], dtype=np.int32)
        return [forehead, lc, rc]

    def create_face_mask_from_landmarks(self, img_shape: Tuple[int, int], lms: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int, int, int], List[np.ndarray]]:
        """从特征点创建面部掩码"""
        h, w = img_shape[:2]
        hull = cv2.convexHull(lms.astype(np.float32))
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(mask, hull.astype(np.int32), 255)

        x1, y1, x2, y2 = self.face_bbox_from_points(hull.reshape(-1, 2), img_shape)
        regions = self.regions_from_bbox(x1, y1, x2, y2)
        return mask, (x1, y1, x2, y2), regions

    def create_face_mask_fallback(self, img_shape: Tuple[int, int], bbox: Tuple[int, int, int, int]) -> Tuple[np.ndarray, List[np.ndarray]]:
        """回退方案：创建面部掩码"""
        h, w = img_shape[:2]
        x1, y1, x2, y2 = bbox
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, thickness=-1)
        regions = self.regions_from_bbox(x1, y1, x2, y2)
        return mask, regions

    def blend_white_to_red(self, base_rgb: np.ndarray, sens01: np.ndarray, alpha_scale: float = 0.6, 
                          sensitivity_threshold: float = 0.3, min_alpha: float = 0.1) -> np.ndarray:
        """将敏感性映射为白色到红色的渐变，只突出高敏感区域"""
        s = np.clip(sens01, 0.0, 1.0)[..., None]
        
        # 创建阈值掩码，只有超过阈值的区域才显示明显的红色
        high_sensitivity_mask = (s[:, :, 0] > sensitivity_threshold).astype(np.float32)
        
        # 对于低敏感区域，重新映射敏感性值
        s_adjusted = s.copy()
        # 低敏感区域：线性映射到很小的值
        low_sens_areas = s[:, :, 0] <= sensitivity_threshold
        s_adjusted[low_sens_areas, 0] = s[low_sens_areas, 0] * min_alpha / sensitivity_threshold
        
        # 高敏感区域：重新映射到 [min_alpha, 1.0] 范围
        high_sens_areas = s[:, :, 0] > sensitivity_threshold
        if np.any(high_sens_areas):
            s_high = s[high_sens_areas, 0]
            # 将 [threshold, 1.0] 映射到 [min_alpha, 1.0]
            s_adjusted[high_sens_areas, 0] = min_alpha + (s_high - sensitivity_threshold) * (1.0 - min_alpha) / (1.0 - sensitivity_threshold)
        
        # 创建红色覆盖层
        red = np.ones_like(base_rgb, dtype=np.float32) * 255.0
        red[:, :, 1] = 255.0 * (1.0 - s_adjusted[:, :, 0])  # G通道
        red[:, :, 2] = 255.0 * (1.0 - s_adjusted[:, :, 0])  # B通道
        
        # 计算透明度，低敏感区域几乎透明
        alpha = (s_adjusted * alpha_scale).astype(np.float32)
        
        # 混合颜色
        out = (base_rgb.astype(np.float32) * (1 - alpha) + red * alpha).clip(0, 255).astype(np.uint8)
        return out

    def render_white_background(self, h: int, w: int) -> np.ndarray:
        """创建白色背景"""
        return np.ones((h, w, 3), dtype=np.uint8) * 255
    
    def create_adaptive_sensitivity_visualization(self, base_rgb: np.ndarray, sens01: np.ndarray, 
                                                alpha_scale: float = 0.6) -> np.ndarray:
        """创建自适应敏感性可视化，根据数据分布动态调整阈值"""
        s = np.clip(sens01, 0.0, 1.0)
        
        # 计算敏感性数据的统计信息
        valid_mask = s > 0.01  # 排除几乎为0的区域
        if np.sum(valid_mask) == 0:
            return base_rgb
            
        valid_sens = s[valid_mask]
        mean_sens = np.mean(valid_sens)
        std_sens = np.std(valid_sens)
        max_sens = np.max(valid_sens)
        
        print(f"敏感性统计: 均值={mean_sens:.3f}, 标准差={std_sens:.3f}, 最大值={max_sens:.3f}")
        
        # 动态计算阈值
        low_threshold = max(0.05, mean_sens - 0.5 * std_sens)
        medium_threshold = mean_sens
        high_threshold = mean_sens + 0.5 * std_sens
        very_high_threshold = mean_sens + 1.0 * std_sens
        
        print(f"动态阈值: 低={low_threshold:.3f}, 中={medium_threshold:.3f}, 高={high_threshold:.3f}, 极高={very_high_threshold:.3f}")
        
        # 创建掩码
        mask_very_low = s < low_threshold
        mask_low = (s >= low_threshold) & (s < medium_threshold)
        mask_medium = (s >= medium_threshold) & (s < high_threshold)
        mask_high = (s >= high_threshold) & (s < very_high_threshold)
        mask_very_high = s >= very_high_threshold
        
        # 创建透明度映射
        alpha = np.zeros_like(s)
        
        # 很低敏感性：几乎透明
        alpha[mask_very_low] = 0.0
        
        # 低敏感性：很淡
        alpha[mask_low] = 0.02
        
        # 中等敏感性：淡红色
        alpha[mask_medium] = 0.08
        
        # 高敏感性：明显红色
        alpha[mask_high] = 0.25
        
        # 很高敏感性：强烈红色
        alpha[mask_very_high] = 0.5
        
        # 应用alpha_scale
        alpha = alpha * alpha_scale
        alpha = np.clip(alpha, 0, 1)
        
        # 创建颜色覆盖层
        overlay = np.zeros_like(base_rgb.astype(np.float32))
        
        # 为不同区域设置不同颜色
        overlay[mask_very_low] = [255, 255, 255]  # 纯白色
        overlay[mask_low] = [255, 250, 250]       # 几乎白色
        overlay[mask_medium] = [255, 220, 220]    # 淡粉色
        overlay[mask_high] = [255, 120, 120]      # 明显红色
        overlay[mask_very_high] = [255, 50, 50]   # 鲜艳红色
        
        # 混合颜色
        result = base_rgb.astype(np.float32)
        alpha_expanded = alpha[..., np.newaxis]
        result = result * (1 - alpha_expanded) + overlay * alpha_expanded
        
        return result.clip(0, 255).astype(np.uint8)

    def create_enhanced_sensitivity_visualization(self, base_rgb: np.ndarray, sens01: np.ndarray, 
                                                alpha_scale: float = 0.6) -> np.ndarray:
        """创建增强的敏感性可视化，使用更精细的颜色映射"""
        s = np.clip(sens01, 0.0, 1.0)
        h, w = s.shape[:2]
        
        # 创建输出图像
        result = base_rgb.astype(np.float32).copy()
        
        # 使用向量化操作提高性能
        # 创建不同敏感性等级的掩码 - 调整阈值让敏感区域更突出
        mask_very_low = s < 0.15     # 0.0-0.15: 几乎透明
        mask_low = (s >= 0.15) & (s < 0.3)   # 0.15-0.3: 很淡
        mask_medium = (s >= 0.3) & (s < 0.5) # 0.3-0.5: 淡红色
        mask_high = (s >= 0.5) & (s < 0.7)   # 0.5-0.7: 明显红色
        mask_very_high = s >= 0.7            # 0.7-1.0: 强烈红色
        
        # 创建透明度映射 - 大幅增加高敏感区域的透明度
        alpha = np.zeros_like(s)
        
        # 很低敏感性：几乎透明
        alpha[mask_very_low] = s[mask_very_low] * 0.02  # 最大0.003
        
        # 低敏感性：很淡
        alpha[mask_low] = 0.003 + (s[mask_low] - 0.15) * 0.05  # 0.003-0.01
        
        # 中等敏感性：淡红色
        alpha[mask_medium] = 0.01 + (s[mask_medium] - 0.3) * 0.2  # 0.01-0.05
        
        # 高敏感性：明显红色
        alpha[mask_high] = 0.05 + (s[mask_high] - 0.5) * 0.5  # 0.05-0.15
        
        # 很高敏感性：强烈红色
        alpha[mask_very_high] = 0.15 + (s[mask_very_high] - 0.7) * 1.0  # 0.15-0.45
        
        # 应用alpha_scale
        alpha = alpha * alpha_scale
        alpha = np.clip(alpha, 0, 1)
        
        # 创建颜色覆盖层
        overlay = np.zeros_like(result)
        
        # 为不同区域设置不同颜色 - 让高敏感区域更加鲜艳
        overlay[mask_very_low] = [255, 255, 255]  # 纯白色
        overlay[mask_low] = [255, 245, 245]       # 几乎白色
        overlay[mask_medium] = [255, 200, 200]    # 淡粉色
        overlay[mask_high] = [255, 100, 100]      # 明显红色
        overlay[mask_very_high] = [255, 30, 30]   # 鲜艳红色
        
        # 混合颜色
        alpha_expanded = alpha[..., np.newaxis]
        result = result * (1 - alpha_expanded) + overlay * alpha_expanded
        
        return result.clip(0, 255).astype(np.uint8)

    def apply_mask(self, arr: np.ndarray, mask01: np.ndarray) -> np.ndarray:
        """应用掩码"""
        return arr * mask01 if arr.ndim == 2 else arr * mask01[..., None]

    def analyze_skin_sensitivity(self, img_bgr: np.ndarray, 
                                alpha: float = 0.65, 
                                wr: float = 0.75, 
                                wt: float = 0.25, 
                                smooth: int = 11,
                                use_mediapipe: bool = True,
                                save_debug_images: bool = False,
                                save_dir: str = None,
                                mask: np.ndarray = None) -> dict:
        """
        分析皮肤敏感性
        
        Args:
            img_bgr: BGR格式的输入图像
            alpha: 覆盖层最大透明度
            wr: 红度权重
            wt: 纹理权重
            smooth: 高斯平滑核大小
            use_mediapipe: 是否使用MediaPipe
            save_debug_images: 是否保存调试图像
            save_dir: 保存目录
            
        Returns:
            dict: 分析结果
        """
        # 转换为RGB
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]


        # 使用传入的掩码或创建新的掩码
        if mask is not None:
            # 使用传入的掩码
            face_mask = mask
            print("使用传入的面部掩码")
        else:
            # 面部检测和区域定义（回退方案）
            landmarks = self.try_facemesh_landmarks(img_rgb)
            if landmarks is not None:
                face_mask, bbox, regions = self.create_face_mask_from_landmarks(img_rgb.shape, landmarks)
                print("使用MediaPipe创建掩码")
            else:
                bbox = self.fallback_bbox(img_rgb)
                face_mask, regions = self.create_face_mask_fallback(img_rgb.shape, bbox)
                print("使用回退方案创建掩码")

        mask01 = (face_mask.astype(np.float32) / 255.0)

        # 计算敏感性指标
        R = self.compute_redness_a(img_rgb)  # 红度
        T = self.compute_texture_laplacian(img_rgb)  # 纹理

        # 组合敏感性评分
        S = wr * R + wt * T
        S = np.clip(S, 0.0, 1.0)

        # 应用面部掩码并平滑
        S = self.apply_mask(S, mask01)
        S = self.gaussian_smooth01(S, ksize=smooth)
        S = self.normalize01(S)  # 按图像归一化

        # 计算整体敏感性评分
        face_sensitivity_pixels = S[face_mask > 0]
        if len(face_sensitivity_pixels) > 0:
            raw_sensitivity_index = float(np.mean(face_sensitivity_pixels)) * 100
            max_sensitivity = float(np.max(face_sensitivity_pixels)) * 100
            sensitivity_std = float(np.std(face_sensitivity_pixels)) * 100
        else:
            raw_sensitivity_index = 0.0
            max_sensitivity = 0.0
            sensitivity_std = 0.0

        # 生成可视化图像
        base_white = self.render_white_background(h, w)
        
        # 使用自适应可视化方法，根据数据分布动态调整
        overlay_white = self.create_adaptive_sensitivity_visualization(
            base_white, S, alpha_scale=alpha
        )
        
        # 备选方案1：固定阈值增强版（如果自适应效果不好，可以取消注释）
        # overlay_white = self.create_enhanced_sensitivity_visualization(
        #     base_white, S, alpha_scale=alpha
        # )
        
        # 备选方案2：简单阈值方法（如果需要更简单的效果，可以取消注释）
        # overlay_white = self.blend_white_to_red(
        #     base_white, S, 
        #     alpha_scale=alpha, 
        #     sensitivity_threshold=0.3,  # 只有敏感性超过30%才显示明显红色
        #     min_alpha=0.02  # 低敏感区域的最小透明度，几乎透明
        # )
        
        # 敏感性灰度图
        S_gray = (S * 255).astype(np.uint8)

        # 保存调试图像或生成字节流
        saved_files = {}
        image_bytes = {}
        
        # 生成白底覆盖图的字节流（总是生成，用于上传到MinIO）
        _, overlay_encoded = cv2.imencode('.jpg', self.to_bgr(overlay_white))
        image_bytes['overlay_on_white'] = overlay_encoded.tobytes()
        
        # 如果需要本地保存（调试模式）
        if save_debug_images and save_dir:
            os.makedirs(save_dir, exist_ok=True)
            
            # 保存敏感性灰度图（仅本地，用于调试）
            sensitivity_gray_path = os.path.join(save_dir, "sensitivity_gray.jpg")
            cv2.imwrite(sensitivity_gray_path, S_gray)
            saved_files['sensitivity_gray'] = sensitivity_gray_path
            
            # 保存白底覆盖图（本地+MinIO）
            overlay_white_path = os.path.join(save_dir, "overlay_on_white.jpg")
            cv2.imwrite(overlay_white_path, self.to_bgr(overlay_white))
            saved_files['overlay_on_white'] = overlay_white_path

        # 分类敏感性等级
        stability_score = max(0, min(100, int(round(100.0 - raw_sensitivity_index))))
        sensitivity_level = self.classify_sensitivity_level(stability_score)

        return {
            'sensitivity_score': stability_score,
            'sensitivity_index': round(raw_sensitivity_index, 2),
            'max_sensitivity': round(max_sensitivity, 2),
            'sensitivity_std': round(sensitivity_std, 2),
            'sensitivity_level': sensitivity_level,
            'analysis_params': {
                'redness_weight': wr,
                'texture_weight': wt,
                'alpha': alpha,
                'smooth_kernel': smooth,

            },
            # 'face_detection': {
            #     'bbox': bbox,
            #     'regions_count': len(regions),
            #     'face_area_pixels': int(np.sum(face_mask > 0))
            # },
            'saved_files': saved_files,
            'image_bytes': image_bytes,  # 新增：图片字节流
            'sensitivity_map': S,  # 用于进一步处理
            'overlay_image': overlay_white  # 用于返回
        }

    def classify_sensitivity_level(self, score: float) -> str:
        """根据稳定度评分分类敏感性等级。

        `score` 采用 0-100 的正向质量分，高分表示更稳定、低分表示更敏感。
        """
        if score >= 80:
            return "低敏感"
        elif score >= 60:
            return "轻度敏感"
        elif score >= 40:
            return "中度敏感"
        elif score >= 20:
            return "高度敏感"
        else:
            return "极度敏感"

    def create_sensitivity_result(self, analysis_result: dict) -> dict:
        """创建敏感性分析结果字典"""
        return {
            'sensitivity_score': analysis_result['sensitivity_score'],
            'sensitivity_index': analysis_result.get('sensitivity_index'),
            'sensitivity_level': analysis_result['sensitivity_level'],
            'max_sensitivity': analysis_result['max_sensitivity'],
            'sensitivity_std': analysis_result['sensitivity_std'],
            'analysis_method': 'Redness + Texture Stability Score',
            'description': self._get_sensitivity_description(analysis_result['sensitivity_level']),
            'analysis_params': analysis_result['analysis_params'],
            # 'face_detection': analysis_result['face_detection']
        }

    def _get_sensitivity_description(self, level: str) -> str:
        """获取敏感性等级描述"""
        descriptions = {
            "低敏感": "当前图像提示敏感倾向较低，建议保持温和清洁、保湿和防晒，避免过度去角质。",
            "轻度敏感": "当前图像提示轻度敏感倾向，面部可能有轻微泛红，建议减少刺激成分并加强保湿修护。",
            "中度敏感": "当前图像提示中度敏感倾向，泛红较明显，建议优先温和清洁、保湿修护和日间防晒。",
            "高度敏感": "当前图像提示较高敏感倾向，建议暂停刺激性护理，先做舒缓保湿；如伴随不适，建议线下咨询专业人士。",
            "极度敏感": "当前图像提示很高敏感倾向，建议避免刺激性护理并加强舒缓保湿；如持续泛红或不适，建议线下咨询专业人士。",
        }
        return descriptions.get(level, "当前图像的敏感倾向暂无法判断，建议结合日常感受和清晰照片复核。")
