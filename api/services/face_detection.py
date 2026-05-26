"""
人脸检测和特征点提取服务
基于MediaPipe和dlib实现人脸特征点检测和面部区域分割
"""

import cv2
import numpy as np
import os
from pathlib import Path
from types import SimpleNamespace

from api.services.mediapipe_compat import (
    create_face_mesh,
    get_drawing_styles,
    get_drawing_utils,
    synthesize_face_landmarks,
)

class FaceDetectionService:
    def __init__(self, predictor_path="shape_predictor_68_face_landmarks.dat"):
        """
        初始化人脸检测服务
        

        """
        
        # MediaPipe初始化；新版 mediapipe 可能没有 mp.solutions，缺失时用 Haar 兜底。
        self.mp_face_mesh = SimpleNamespace(FaceMesh=create_face_mesh)
        self.mp_drawing = get_drawing_utils()
        self.mp_drawing_styles = get_drawing_styles()

        self.face_mesh = create_face_mesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )
        
        # 定义面部区域的关键点索引 (基于MediaPipe 468个特征点)
        self.face_regions = {
            # 左眼区域
            'left_eye': [
                33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246,
            ],
            # 右眼区域
            'right_eye': [
                362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398,
            ],
            # 鼻孔区域
            'nostrils': [
                94, 98, 97, 2, 326, 327, 294, 278, 344, 440, 275, 4, 45, 220, 115, 48, 64, 98
            ],
            'nose': [
                6, 122, 188, 174, 236, 198, 209, 129, 98, 97, 2, 326, 327, 58, 429, 420, 456, 399, 412, 351
            ],
            # 嘴巴区域
            'mouth': [
                0, 37, 39, 40, 185, 61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267
            ],
            # 口腔内部区域
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
        
        # 皮肤区域轮廓索引
        self.skin_contour_indices = [
            10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
            397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
            172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109
        ]
        
        # 特定皮肤分析区域的关键点索引
        self.skin_analysis_regions = {
            # 额头区域
            'forehead': [
                103, 104, 105, 66, 107, 9, 336, 296, 334, 333, 332, 297, 338, 10, 109, 67
            ],
            
            # 鼻梁区域
            'nose_bridge': [
                6, 122, 188, 174, 236, 198, 209, 129, 98, 97, 2, 326, 327, 358, 429, 420, 456, 399, 412, 351
            ],
            
            # 左脸颊区域
            'left_cheek': [
                350, 349, 348, 347, 346, 340, 345, 352, 411, 427, 436, 426, 423, 358, 355, 277
            ],
            
            # 右脸颊区域  
            'right_cheek': [
                121, 120, 119, 118, 117, 111, 116, 123, 187, 207, 216, 206, 203, 129, 126, 47
            ],
            
            # 下巴区域
            'chin': [
                18, 83, 182, 106, 204, 211, 170, 149, 176, 148, 152, 377, 400, 378, 395, 431, 424, 335, 406, 313
            ]
        }


    def detect_mediapipe_landmarks(self, image):
        """
        使用MediaPipe检测面部网格点
        
        Args:
            image: 输入图像
            
        Returns:
            landmarks: 468个面部网格点坐标
        """
        if self.face_mesh is None:
            return synthesize_face_landmarks(image)

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_image)
        
        if not results.multi_face_landmarks:
            return None
        
        # 获取第一个检测到的面部
        face_landmarks = results.multi_face_landmarks[0]
        
        # 转换为像素坐标
        h, w = image.shape[:2]
        landmarks = []
        
        for landmark in face_landmarks.landmark:
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            landmarks.append((x, y))
        
        return landmarks

    def draw_landmarks(self, image, landmarks, color=(0, 255, 0), radius=2):
        """
        在图像上绘制特征点
        
        Args:
            image: 输入图像
            landmarks: 特征点坐标列表
            color: 绘制颜色
            radius: 点的半径
            
        Returns:
            result: 绘制了特征点的图像
        """
        result = image.copy()
        
        if landmarks is None:
            return result
        
        for point in landmarks:
            cv2.circle(result, point, radius, color, -1)
        
        return result

    def create_region_mask(self, image, landmarks, region_indices):
        """
        创建特定区域的掩码
        
        Args:
            image: 输入图像
            landmarks: MediaPipe面部特征点
            region_indices: 区域特征点索引列表
            
        Returns:
            mask: 区域掩码
        """
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        
        if landmarks is None or len(landmarks) < max(region_indices):
            return mask
        
        # 获取区域点坐标
        region_points = []
        for idx in region_indices:
            if idx < len(landmarks):
                region_points.append(landmarks[idx])
        
        if len(region_points) > 2:
            region_points = np.array(region_points, dtype=np.int32)
            cv2.fillPoly(mask, [region_points], 255)
        
        return mask

    def create_comprehensive_masks(self, image, landmarks=None):
        """
        创建全面的面部区域掩码
        
        Args:
            image: 输入图像
            landmarks: 可选的预检测特征点
            
        Returns:
            masks: 包含各区域掩码的字典
        """
        # 如果没有提供特征点，则检测
        if landmarks is None:
            landmarks = self.detect_mediapipe_landmarks(image)
        
        if landmarks is None:
            print("未检测到MediaPipe面部特征点")
            return None
        
        masks = {}
        h, w = image.shape[:2]
        
        # 创建各个非皮肤区域的掩码
        masks['left_eye'] = self.create_region_mask(image, landmarks, self.face_regions['left_eye'])
        masks['right_eye'] = self.create_region_mask(image, landmarks, self.face_regions['right_eye'])
        masks['nostrils'] = self.create_region_mask(image, landmarks, self.face_regions['nostrils'])
        masks['mouth'] = self.create_region_mask(image, landmarks, self.face_regions['mouth'])
        masks['mouth_interior'] = self.create_region_mask(image, landmarks, self.face_regions['mouth_interior'])
        masks['eyebrows_left'] = self.create_region_mask(image, landmarks, self.face_regions['eyebrows_left'])
        masks['eyebrows_right'] = self.create_region_mask(image, landmarks, self.face_regions['eyebrows_right'])


        
        masks['face_contour'] = self.create_region_mask(image, landmarks, self.skin_contour_indices)
        masks['nose'] = self.create_region_mask(image, landmarks, self.face_regions['nose'])

        # 创建非皮肤区域的联合掩码
        non_skin_mask = np.zeros((h, w), dtype=np.uint8)
        for region in ['left_eye', 'right_eye', 'nostrils', 'mouth_interior', 'eyebrows_left', 'eyebrows_right']:
            non_skin_mask = cv2.bitwise_or(non_skin_mask, masks[region])
        
        # 创建纯皮肤区域掩码
        skin_mask = cv2.bitwise_and(masks['face_contour'], cv2.bitwise_not(non_skin_mask))
        
        # 形态学操作清理掩码
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel) 
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
        
        masks['skin_only'] = skin_mask
        masks['non_skin'] = non_skin_mask

        # 创建鼻子专用检测掩码（只包含鼻子区域）
        nose_only_mask = masks['nose'].copy()
        # 形态学操作优化鼻子掩码
        kernel_nose = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        nose_only_mask = cv2.morphologyEx(nose_only_mask, cv2.MORPH_CLOSE, kernel_nose)
        nose_only_mask = cv2.morphologyEx(nose_only_mask, cv2.MORPH_OPEN, kernel_nose)
        
        masks['nose_only'] = nose_only_mask
        
        return masks

    def build_skin_mask_advanced(self, image, face_rect=None):
        """
        基于肤色的像素级掩膜（来自Facial_water_deficiency_detection.py）
        - 在 HSV 与 YCrCb 空间阈值取交集
        - 形态学清理 + 连通域筛选
        
        Args:
            image: 输入图像
            face_rect: 可选的人脸框 (x, y, w, h)
            
        Returns:
            mask: 皮肤掩码 (H, W) uint8, 值为 0/255
        """
        H, W = image.shape[:2]

        # HSV 肤色阈值
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_hsv1 = np.array([0, 30, 50], dtype=np.uint8)
        upper_hsv1 = np.array([25, 180, 255], dtype=np.uint8)
        mask_hsv1 = cv2.inRange(hsv, lower_hsv1, upper_hsv1)
        
        # 红色端
        lower_hsv2 = np.array([160, 30, 50], dtype=np.uint8)
        upper_hsv2 = np.array([179, 180, 255], dtype=np.uint8)
        mask_hsv2 = cv2.inRange(hsv, lower_hsv2, upper_hsv2)
        mask_hsv = cv2.bitwise_or(mask_hsv1, mask_hsv2)

        # YCrCb 肤色范围
        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        mask_ycrcb = cv2.inRange(
            ycrcb,
            np.array([0, 135, 85], dtype=np.uint8),
            np.array([255, 180, 135], dtype=np.uint8)
        )

        # 交集 + 形态学清理
        mask = cv2.bitwise_and(mask_hsv, mask_ycrcb)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.GaussianBlur(mask, (0, 0), 1.0)
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

        # 若提供人脸框，只保留与其重叠最大的连通域
        if face_rect is not None:
            x, y, w, h = face_rect
            x = max(0, x); y = max(0, y)
            w = min(w, W - x); h = min(h, H - y)
            if w > 0 and h > 0:
                num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
                if num_labels > 1:
                    face_box = np.array([x, y, x + w, y + h])
                    best_label, best_iou = 0, 0.0
                    for lab in range(1, num_labels):
                        x0, y0, bw, bh, area = stats[lab]
                        comp_box = np.array([x0, y0, x0 + bw, y0 + bh])
                        xi1, yi1 = max(face_box[0], comp_box[0]), max(face_box[1], comp_box[1])
                        xi2, yi2 = min(face_box[2], comp_box[2]), min(face_box[3], comp_box[3])
                        inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
                        union = (face_box[2]-face_box[0])*(face_box[3]-face_box[1]) + bw*bh - inter
                        iou = inter / union if union > 0 else 0.0
                        if iou > best_iou:
                            best_iou = iou
                            best_label = lab
                    mask = np.where(labels == best_label, 255, 0).astype(np.uint8)

        # 挖空五官（如果可用MediaPipe）
        try:
            feature_polys = self._get_feature_polygons_mediapipe(image)
            if feature_polys:
                for pts in feature_polys.values():
                    try:
                        cv2.fillConvexPoly(mask, pts, 0)
                    except Exception:
                        pass
        except Exception:
            pass

        return mask

    def _get_feature_polygons_mediapipe(self, image):
        """
        返回需挖空的五官多边形字典
        
        Returns:
            dict: {'left_eye': Nx2, 'right_eye':..., 'mouth':..., etc.}
        """
        if not hasattr(self, 'face_mesh') or self.face_mesh is None:
            return {}
            
        ih, iw = image.shape[:2]
        polys = {}
        
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_image)
        
        if not results.multi_face_landmarks:
            return {}
            
        lm = results.multi_face_landmarks[0].landmark

        def to_points(indices):
            pts = []
            for idx in indices:
                x = int(lm[idx].x * iw + 0.5)
                y = int(lm[idx].y * ih + 0.5)
                x = np.clip(x, 0, iw - 1)
                y = np.clip(y, 0, ih - 1)
                pts.append([x, y])
            if len(pts) >= 3:
                pts = np.array(pts, dtype=np.int32)
                pts = cv2.convexHull(pts)
            else:
                pts = None
            return pts

        # 关键点索引集合
        left_eye_idx  = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        right_eye_idx = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        left_brow_idx  = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
        right_brow_idx = [336, 296, 334, 293, 300, 383, 276, 283, 282, 295]
        mouth_outer_idx = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317]
        nose_idx = [6, 197, 195, 5, 4, 1, 275, 440, 344, 278, 48, 64, 98, 97, 2, 326, 327, 358, 327]

        polys['left_eye']  = to_points(left_eye_idx)
        polys['right_eye'] = to_points(right_eye_idx)
        polys['left_brow'] = to_points(left_brow_idx)
        polys['right_brow']= to_points(right_brow_idx)
        polys['mouth']     = to_points(mouth_outer_idx)
        polys['nose']      = to_points(nose_idx)

        # 去除 None
        polys = {k:v for k,v in polys.items() if v is not None}
        return polys

    def create_skin_analysis_masks(self, image, landmarks=None):
        """
        创建特定皮肤分析区域的掩码（额头、鼻梁、脸颊、下巴）
        
        Args:
            image: 输入图像
            landmarks: 可选的预检测特征点
            
        Returns:
            analysis_masks: 包含皮肤分析区域掩码的字典
        """
        # 如果没有提供特征点，则检测
        if landmarks is None:
            landmarks = self.detect_mediapipe_landmarks(image)
        
        if landmarks is None:
            print("未检测到MediaPipe面部特征点")
            return None
        
        analysis_masks = {}
        h, w = image.shape[:2]
        
        print(f"创建皮肤分析区域掩码，图像尺寸: {h}x{w}")
        
        # 创建各个皮肤分析区域的掩码
        for region_name, indices in self.skin_analysis_regions.items():
            mask = self.create_region_mask(image, landmarks, indices)
            analysis_masks[region_name] = mask
            print(f"创建 {region_name} 区域掩码完成")
        
        # 创建眼部和嘴部掩码（需要排除的区域）
        eye_mouth_mask = np.zeros((h, w), dtype=np.uint8)
        
        # 添加眼部区域
        left_eye_mask = self.create_region_mask(image, landmarks, self.face_regions['left_eye'])
        right_eye_mask = self.create_region_mask(image, landmarks, self.face_regions['right_eye'])
        mouth_mask = self.create_region_mask(image, landmarks, self.face_regions['mouth'])
        eyebrows_left_mask = self.create_region_mask(image, landmarks, self.face_regions['eyebrows_left'])
        eyebrows_right_mask = self.create_region_mask(image, landmarks, self.face_regions['eyebrows_right'])
        
        # 合并需要排除的区域
        eye_mouth_mask = cv2.bitwise_or(eye_mouth_mask, left_eye_mask)
        eye_mouth_mask = cv2.bitwise_or(eye_mouth_mask, right_eye_mask)
        eye_mouth_mask = cv2.bitwise_or(eye_mouth_mask, mouth_mask)
        eye_mouth_mask = cv2.bitwise_or(eye_mouth_mask, eyebrows_left_mask)
        eye_mouth_mask = cv2.bitwise_or(eye_mouth_mask, eyebrows_right_mask)
        
        # 从每个分析区域中排除眼部和嘴部
        for region_name in analysis_masks.keys():
            # 排除眼部和嘴部区域
            analysis_masks[region_name] = cv2.bitwise_and(
                analysis_masks[region_name], 
                cv2.bitwise_not(eye_mouth_mask)
            )
            
            # 形态学操作清理掩码
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            analysis_masks[region_name] = cv2.morphologyEx(
                analysis_masks[region_name], cv2.MORPH_CLOSE, kernel
            )
            analysis_masks[region_name] = cv2.morphologyEx(
                analysis_masks[region_name], cv2.MORPH_OPEN, kernel
            )
        
        # 创建所有分析区域的联合掩码
        combined_analysis_mask = np.zeros((h, w), dtype=np.uint8)
        for region_mask in analysis_masks.values():
            combined_analysis_mask = cv2.bitwise_or(combined_analysis_mask, region_mask)
        
        analysis_masks['combined_analysis_regions'] = combined_analysis_mask
        analysis_masks['excluded_regions'] = eye_mouth_mask
        
        print("皮肤分析区域掩码创建完成")
        return analysis_masks

    def visualize_masks(self, image, masks, output_dir):
        """
        可视化所有掩码（用于调试）
        
        Args:
            image: 原始图像
            masks: 掩码字典
            output_dir: 输出目录
        """
        Path(output_dir).mkdir(exist_ok=True)
        
        for mask_name, mask in masks.items():
            # 保存掩码
            mask_path = os.path.join(output_dir, f"{mask_name}_mask.jpg")
            cv2.imwrite(mask_path, mask)
            
            # 保存应用掩码的结果
            masked_result = cv2.bitwise_and(image, image, mask=mask)
            result_path = os.path.join(output_dir, f"{mask_name}_result.jpg")
            cv2.imwrite(result_path, masked_result)
