"""
肤色分类服务
基于ITA (Individual Typology Angle) 算法进行肤色分类
"""

import cv2
import numpy as np
from skimage import color
import math
import os

try:
    from api.services.mediapipe_compat import create_face_mesh
    _MP_AVAILABLE = True
except Exception:
    create_face_mesh = None
    _MP_AVAILABLE = False

# PIL 用于中文文本绘制（OpenCV Hershey 字体不支持中文）
try:
    from PIL import ImageFont, ImageDraw, Image
    _PIL_AVAILABLE = True
except Exception:
    ImageFont = ImageDraw = Image = None
    _PIL_AVAILABLE = False


class SkinClassificationService:
    """肤色分类服务类"""
    
    def __init__(self):
        # 面部区域定义
        self.face_regions = {
            'left_eye': [
                33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246,
            ],
            'right_eye': [
                362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398,
            ],
            'nostrils': [
                94, 98, 97, 2, 326, 327, 294, 278, 344, 440, 275, 4, 45, 220, 115, 48, 64, 98
            ],
            'mouth': [
                0, 37, 39, 40, 185, 61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267
            ],
            'mouth_interior': [
                0, 37, 39, 40, 185, 61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267
            ],
            'eyebrows_left': [
                70, 63, 105, 66, 107, 55, 65, 52, 53, 46,
            ],
            'eyebrows_right': [
                300, 293, 334, 296, 336, 285, 295, 282, 283, 276
            ]
        }
        
        self.skin_contour_indices = [
            10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
            397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
            172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109
        ]

    def gray_world_white_balance(self, img_bgr):
        """Gray World 白平衡算法"""
        img = img_bgr.astype(np.float32)
        mean_b = np.mean(img[:, :, 0])
        mean_g = np.mean(img[:, :, 1])
        mean_r = np.mean(img[:, :, 2])
        mean_gray = (mean_b + mean_g + mean_r) / 3.0 + 1e-6
        gain_b = mean_gray / (mean_b + 1e-6)
        gain_g = mean_gray / (mean_g + 1e-6)
        gain_r = mean_gray / (mean_r + 1e-6)
        img[:, :, 0] *= gain_b
        img[:, :, 1] *= gain_g
        img[:, :, 2] *= gain_r
        img = np.clip(img, 0, 255).astype(np.uint8)
        return img

    def simple_skin_mask(self, img_bgr):
        """简易皮肤掩码（HSV 范围）"""
        img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        # 常见皮肤色范围
        lower1 = np.array([0, 30, 50], dtype=np.uint8)
        upper1 = np.array([25, 255, 255], dtype=np.uint8)
        mask1 = cv2.inRange(img_hsv, lower1, upper1)

        lower2 = np.array([160, 30, 50], dtype=np.uint8)
        upper2 = np.array([180, 255, 255], dtype=np.uint8)
        mask2 = cv2.inRange(img_hsv, lower2, upper2)

        mask = cv2.bitwise_or(mask1, mask2)
        # 形态学清理
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        return mask

    def detect_mediapipe_landmarks(self, image_bgr):
        """使用MediaPipe检测面部特征点"""
        if not _MP_AVAILABLE:
            return None
        try:
            h, w = image_bgr.shape[:2]
            face_mesh_factory = create_face_mesh
            if face_mesh_factory is None:
                return None
            face_mesh = face_mesh_factory(
                static_image_mode=True,
                refine_landmarks=True,
                max_num_faces=1,
                min_detection_confidence=0.5
            )
            if face_mesh is None:
                return None
            with face_mesh:
                rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(rgb)
                if not results.multi_face_landmarks:
                    return None
                lms = results.multi_face_landmarks[0]
                pts = []
                for lm in lms.landmark:
                    x = int(lm.x * w)
                    y = int(lm.y * h)
                    pts.append((x, y))
                return pts
        except Exception:
            return None

    def create_region_mask(self, image, landmarks, region_indices):
        """创建特定区域的掩码"""
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        if landmarks is None:
            return mask
        region_points = []
        for idx in region_indices:
            if idx < len(landmarks):
                region_points.append(landmarks[idx])
        if len(region_points) > 2:
            region_points = np.array(region_points, dtype=np.int32)
            cv2.fillPoly(mask, [region_points], 255)
        return mask

    def create_skin_only_mask(self, image):
        """创建纯皮肤区域掩码（基于MediaPipe）"""
        landmarks = self.detect_mediapipe_landmarks(image)
        if landmarks is None:
            return None
        h, w = image.shape[:2]
        # 面部轮廓
        face_mask = np.zeros((h, w), dtype=np.uint8)
        face_points = []
        for idx in self.skin_contour_indices:
            if idx < len(landmarks):
                face_points.append(landmarks[idx])
        if len(face_points) > 2:
            face_points = np.array(face_points, dtype=np.int32)
            cv2.fillPoly(face_mask, [face_points], 255)
        # 非皮肤区域
        non_skin = np.zeros((h, w), dtype=np.uint8)
        for key in ['left_eye', 'right_eye', 'nostrils', 'mouth_interior', 'eyebrows_left', 'eyebrows_right']:
            non_skin = cv2.bitwise_or(non_skin, self.create_region_mask(image, landmarks, self.face_regions[key]))
        # skin_only = face - 非皮肤
        skin_mask = cv2.bitwise_and(face_mask, cv2.bitwise_not(non_skin))
        # 轻微形态学清理
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
        return skin_mask

    def get_skin_mask(self, image_bgr):
        """获取皮肤掩码（优先使用MediaPipe，失败则回退到HSV）"""
        mp_mask = self.create_skin_only_mask(image_bgr)
        if mp_mask is not None and np.count_nonzero(mp_mask) > 0:
            return mp_mask
        return self.simple_skin_mask(image_bgr)

    def draw_text_cn(self, img_bgr, text, pos=(10, 30), color=(40, 220, 40), font_size=32):
        """在BGR图像上绘制中文文本"""
        if not _PIL_AVAILABLE:
            # 回退：OpenCV 只能正常显示英文字母与数字
            cv2.putText(img_bgr, text if isinstance(text, str) else str(text),
                        pos, cv2.FONT_HERSHEY_SIMPLEX, font_size / 32.0, color, 2)
            return img_bgr

        # BGR -> RGB -> PIL
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        draw = ImageDraw.Draw(img_pil)

        # 常见中文字体路径（Windows）
        font_paths = [
            r"C:\Windows\Fonts\msyh.ttc",     # 微软雅黑
            r"C:\Windows\Fonts\msyhbd.ttc",   # 微软雅黑Bold
            r"C:\Windows\Fonts\simhei.ttf",   # 黑体
            r"C:\Windows\Fonts\simsun.ttc"    # 宋体
        ]
        font = None
        for fp in font_paths:
            try:
                font = ImageFont.truetype(fp, font_size)
                break
            except Exception:
                continue
        if font is None:
            try:
                font = ImageFont.load_default()
            except Exception:
                font = None

        # 注意：Pillow 使用 RGB 颜色；将 BGR 转为 RGB
        rgb_color = (color[2], color[1], color[0])
        try:
            draw.text(pos, text, font=font, fill=rgb_color)
        except Exception:
            # 万一 text 有编码问题，做一次安全转换
            safe_text = str(text)
            draw.text(pos, safe_text, font=font, fill=rgb_color)

        # PIL -> RGB -> BGR
        img_rgb2 = np.array(img_pil)
        return cv2.cvtColor(img_rgb2, cv2.COLOR_RGB2BGR)

    def compute_ita_from_lab(self, l_channel, b_channel, mask):
        """从Lab颜色空间计算ITA值"""
        # 使用掩膜区域内像素
        sel = mask > 0
        l_vals = l_channel[sel].astype(np.float32)
        b_vals = b_channel[sel].astype(np.float32)
        if l_vals.size < 100:  # 像素太少可能定位失败
            return None
        # 稳健统计：中位数抗异常
        L_med = float(np.median(l_vals))
        B_med = float(np.median(b_vals))
        ita = math.degrees(math.atan2((L_med - 50.0), (B_med if abs(B_med) > 1e-6 else 1e-6)))
        return ita

    def ita_to_category(self, ita):
        """将ITA值映射到肤色类别"""
        if ita is None:
            return "未检测到足够皮肤区域"
        if ita > 55:
            return "透白"
        elif ita > 40:
            return "白色"
        elif ita > 20:
            return "自然"
        elif ita > 10:
            return "小麦"
        elif ita > -5:
            return "棕色"
        else:
            return "深棕"

    def classify_skin_tone(self, image_bgr, save_visualization=False, save_path=None):
        """
        肤色分类主函数
        
        Args:
            image_bgr: BGR格式的输入图像
            save_visualization: 是否保存可视化结果
            save_path: 可视化结果保存路径
            
        Returns:
            tuple: (ita_score, category, visualization_image)
        """
        # 白平衡
        wb_bgr = self.gray_world_white_balance(image_bgr)

        # 皮肤 mask（优先 MediaPipe 的 skin_only，失败回退 HSV）
        mask = self.get_skin_mask(wb_bgr)

        # 转 Lab（skimage 的 rgb2lab 需要 RGB，返回 L* [0-100]）
        img_rgb = cv2.cvtColor(wb_bgr, cv2.COLOR_BGR2RGB)
        img_lab = color.rgb2lab(img_rgb).astype(np.float32)
        L = img_lab[:, :, 0]
        B = img_lab[:, :, 2]  # 使用 b* 与 L* 计算 ITA

        ita = self.compute_ita_from_lab(L, B, mask)
        category = self.ita_to_category(ita)

        # 创建可视化图像
        visualization_image = None
        if save_visualization or save_path:
            # 可视化：原图（白平衡后）、皮肤区域、mask
            skin = cv2.bitwise_and(wb_bgr, wb_bgr, mask=mask)
            mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            h, w = wb_bgr.shape[:2]
            scale = 640.0 / max(h, w)
            
            def resize(x):
                return cv2.resize(x, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
            
            panel = np.hstack([resize(wb_bgr), resize(skin), resize(mask_rgb)])
            text_ita = f"ITA: {ita:.2f}" if ita is not None else "ITA: N/A"
            panel = self.draw_text_cn(panel, text_ita, (10, 30), (40, 220, 40), 32)
            panel = self.draw_text_cn(panel, f"类别: {category}", (10, 70), (40, 220, 40), 32)
            
            visualization_image = panel
            
            # 只在需要本地保存时才写入文件
            if save_path:
                # 检查是否启用本地保存
                import os
                save_local = os.getenv('SAVE_LOCAL_IMAGES', 'false').lower() == 'true'
                if save_local:
                    cv2.imwrite(save_path, panel)
                    print(f"肤色分类可视化结果已保存到: {save_path}")

        return ita, category, visualization_image

    def create_classification_result(self, ita_score, category):
        """创建分类结果字典"""
        return {
            'ita_score': round(ita_score, 2) if ita_score is not None else None,
            'skin_tone_category': category,
            'classification_method': 'ITA (Individual Typology Angle)',
            'description': self._get_category_description(category)
        }

    def _get_category_description(self, category):
        """获取肤色类别描述"""
        descriptions = {
            "透白": "非常白皙的肌肤，通常容易晒伤",
            "白色": "白皙肌肤，较容易晒伤",
            "自然": "自然肤色，适中的黑色素含量",
            "小麦": "小麦色肌肤，色素分布浓度偏高，注意做好防晒，避免长时间紫外线照射从而增加黑素合成，面对高强度紫外线照射需要高强度保湿",
            "棕色": "棕色肌肤，黑色素分布浓度较高，皮肤虽然不易被晒伤，但容易晒后色素沉淀，需要做好晒后修复",
            "深棕": "深棕色肌肤，黑色素含量很高，虽然不易晒伤，但容易晒后色素沉淀，需要做好晒后修复",
            "未检测到足够皮肤区域": "无法准确检测肤色"
        }
        return descriptions.get(category, "未知肤色类别")
