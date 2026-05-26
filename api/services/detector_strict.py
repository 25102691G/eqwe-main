import os
import json
import argparse
import cv2
import numpy as np
from pathlib import Path

# 依赖本仓库中的增强人脸分析器
import sys
CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from skimage.filters import gaussian, median, threshold_otsu
from skimage.measure import label, regionprops
from skimage.morphology import disk, opening, closing
from skimage.feature import local_binary_pattern
from typing import List, Tuple

def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def save_image(path: str, image: np.ndarray) -> None:
    cv2.imwrite(path, image)


# 严格版本的皮肤问题检测器 - 减少误检

class StrictSkinProblemDetector:
    def __init__(self):
        # 使用调试结果中效果最好的参数 (result_1)
        self.acne_types = {
            'acne_marks': {
                'color_ranges': {
                    # 使用result_1的红色参数 - 效果最好
                    'red_marks': {'h': (150, 180), 's': (90, 255), 'v': (50, 255)}
                },
                'size_range': (10, 100),  # 大范围适应各种炎症区域
                'shape_criteria': {'circularity': (0.1, 1.0), 'solidity': (0.1, 1.0)}  # 极宽松
            },
            'papules': {
                'color_ranges': {
                    # 红色丘疹 - 更严格的红色范围
                    'red_papules': {'h': (0, 8), 's': (100, 255), 'v': (120, 255)},
                    # 粉红色丘疹 - 更严格的粉红色范围
                    'pink_papules': {'h': (160, 175), 's': (120, 255), 'v': (150, 255)}
                },
                'size_range': (15, 200),  # 更严格的尺寸范围
                'shape_criteria': {'circularity': (0.5, 1.0), 'solidity': (0.6, 1.0)},  # 更严格的形状要求
                'elevation_required': True  # 需要高程要求，确保是凸起的
            },
            'comedones': {
                'name': '粉刺',
                'color_ranges': {
                    # 极严格的白头检测 - 减少误检
                    'whiteheads': {'h': (0, 180), 's': (0, 15), 'v': (240, 255)},
                    # 极严格的黑头检测
                    'blackheads': {'h': (0, 180), 's': (0, 50), 'v': (0, 40)}
                },
                'size_range': (3, 15),  # 大幅缩小尺寸
                'shape_criteria': {'circularity': (0.8, 1.0), 'solidity': (0.9, 1.0)}  # 极严格
            },
            'nodules': {
                'color_ranges': {
                    # 极严格的结节检测 - 基本不检测
                    'red_nodules': {'h': (0, 5), 's': (150, 255), 'v': (50, 120)}
                },
                'size_range': (150, 250),  # 极小范围
                'shape_criteria': {'circularity': (0.7, 1.0), 'solidity': (0.8, 1.0)},
                'elevation_required': True
            }
        }

        # 极严格的黑头检测 - 减少误检
        self.blackhead_criteria = {
            'color_range': {'h': (0, 180), 's': (0, 30), 'v': (0, 30)},  # 极严格
            'size_range': (2, 8),  # 极小尺寸
            'shape_criteria': {'circularity': (0.9, 1.0)}  # 极严格
        }

        # 极严格的毛孔检测 - 减少误检
        self.pore_criteria = {
            'color_range': {'h': (0, 180), 's': (0, 20), 'v': (40, 100)},  # 极严格
            'size_range': (1, 5),  # 极小尺寸
            'shape_criteria': {'circularity': (0.9, 1.0)}  # 极严格
        }

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        # 针对炎症皮肤优化的预处理
        denoised = cv2.bilateralFilter(image, 7, 60, 60)  # 适中的去噪
        lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # 适度增强对比度，突出红色区域
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(6, 6))
        l = clahe.apply(l)
        
        # 增强a通道（红绿对比）来更好地检测红色炎症
        a = cv2.equalizeHist(a)
        
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        return enhanced

    def detect_skin_region(self, image: np.ndarray) -> np.ndarray:
        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        lower_skin = np.array([0, 133, 77])
        upper_skin = np.array([255, 173, 127])
        skin_mask = cv2.inRange(ycrcb, lower_skin, upper_skin)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
        return skin_mask

    def calculate_elevation_map(self, image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
        elevation_map = gaussian(gradient_magnitude, sigma=1)
        return elevation_map

    def _extract_candidates(self, image_hsv: np.ndarray, skin_mask: np.ndarray, color_range: dict) -> np.ndarray:
        lower = np.array([color_range['h'][0], color_range['s'][0], color_range['v'][0]])
        upper = np.array([color_range['h'][1], color_range['s'][1], color_range['v'][1]])
        color_mask = cv2.inRange(image_hsv, lower, upper)

        color_mask = cv2.bitwise_and(color_mask, skin_mask)
        # cv2.imwrite('color_mask_1.jpg', color_mask)  # 保存第一个color_mask

        
        # 更严格的形态学操作
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))  # 更小的核
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel)
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel)
        # cv2.imwrite('color_mask_2.jpg', color_mask)  # 保存第二个color_mask
        return color_mask

    def _centroids_from_mask(self, mask: np.ndarray, size_range: Tuple[int, int]) -> List[Tuple[int, int]]:
        labels = label(mask > 0)
        points: List[Tuple[int, int]] = []
        for region in regionprops(labels):
            area = region.area
            if area < size_range[0] or area > size_range[1]:
                continue
            cy, cx = map(int, region.centroid)
            points.append((cx, cy))
        return points

    def detect_acne_marks(self, image: np.ndarray, skin_mask: np.ndarray) -> List[Tuple[int, int]]:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        criteria = self.acne_types['acne_marks']
        points: List[Tuple[int, int]] = []
        for _, color_range in criteria['color_ranges'].items():
            color_mask = self._extract_candidates(hsv, skin_mask, color_range)
            points.extend(_centroids_from_mask_with_shape(color_mask, criteria['size_range'], criteria.get('shape_criteria')))
        return points

    def detect_papules(self, image: np.ndarray, skin_mask: np.ndarray, elevation_map: np.ndarray) -> List[Tuple[int, int]]:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        criteria = self.acne_types['papules']
        points: List[Tuple[int, int]] = []
        
        # 处理所有颜色范围的丘疹
        for _, color_range in criteria['color_ranges'].items():
            color_mask = self._extract_candidates(hsv, skin_mask, color_range)
            
            # 降低高程阈值以适应炎症性丘疹
            elev_thr = np.percentile(elevation_map[skin_mask > 0], 85) if np.any(skin_mask > 0) else 0.0
            elev_mask = (elevation_map > elev_thr).astype(np.uint8) * 255
            combined = cv2.bitwise_and(color_mask, elev_mask)
            points.extend(_centroids_from_mask_with_shape(combined, criteria['size_range'], criteria.get('shape_criteria')))
        
        return points

    def detect_whiteheads(self, image: np.ndarray, skin_mask: np.ndarray) -> List[Tuple[int, int]]:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        criteria = self.acne_types['comedones']
        color_range = criteria['color_ranges']['whiteheads']
        color_mask = self._extract_candidates(hsv, skin_mask, color_range)
        return _centroids_from_mask_with_shape(color_mask, criteria['size_range'], criteria.get('shape_criteria'))

    def detect_blackheads(self, image: np.ndarray, skin_mask: np.ndarray) -> List[Tuple[int, int]]:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        color_range = self.blackhead_criteria['color_range']
        color_mask = self._extract_candidates(hsv, skin_mask, color_range)
        return _centroids_from_mask_with_shape(color_mask, self.blackhead_criteria['size_range'], self.blackhead_criteria.get('shape_criteria'))

    def detect_nodules(self, image: np.ndarray, skin_mask: np.ndarray, elevation_map: np.ndarray) -> List[Tuple[int, int]]:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        criteria = self.acne_types['nodules']
        color_range = list(criteria['color_ranges'].values())[0]
        color_mask = self._extract_candidates(hsv, skin_mask, color_range)
        
        # 更严格的高程阈值
        elev_thr = np.percentile(elevation_map[skin_mask > 0], 95) if np.any(skin_mask > 0) else 0.0
        elev_mask = (elevation_map > elev_thr).astype(np.uint8) * 255
        combined = cv2.bitwise_and(color_mask, elev_mask)
        return _centroids_from_mask_with_shape(combined, criteria['size_range'], criteria.get('shape_criteria'))

    def detect_pores(self, image: np.ndarray, skin_mask: np.ndarray) -> List[Tuple[int, int]]:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        color_range = self.pore_criteria['color_range']
        color_mask = self._extract_candidates(hsv, skin_mask, color_range)
        return self._centroids_from_mask(color_mask, self.pore_criteria['size_range'])


def _centroids_from_mask_with_shape(mask: np.ndarray, size_range: Tuple[int, int], shape_criteria: dict | None = None) -> List[Tuple[int, int]]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    points: List[Tuple[int, int]] = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < size_range[0] or area > size_range[1]:
            continue
        perimeter = cv2.arcLength(cnt, True)
        circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0.0
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity = (area / hull_area) if hull_area > 0 else 0.0
        
        if shape_criteria is not None:
            if 'circularity' in shape_criteria:
                lo, hi = shape_criteria['circularity']
                if not (lo <= circularity <= hi):
                    continue
            if 'solidity' in shape_criteria:
                lo, hi = shape_criteria['solidity']
                if not (lo <= solidity <= hi):
                    continue
        
        M = cv2.moments(cnt)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            points.append((cx, cy))
    return points


def draw_points_on_image(image: np.ndarray, points: List[Tuple[int, int]], color=(0, 0, 255), radius=3, thickness=-1) -> np.ndarray:
    result = image.copy()
    for (x, y) in points:
        cv2.circle(result, (int(x), int(y)), radius, color, thickness)
    return result

def run_pipeline(image, task_dir, masks) -> dict:
    # 读取图像
    # image = cv2.imread(image_path)
    # if image is None:
    #     raise FileNotFoundError(f"无法读取图像: {image_path}")

    # ensure_dir(output_dir)

    skin_mask = masks['skin_only']  # 仅皮肤区域（非皮肤掩码已排除）
    nose_mask = masks['nose_only']  # 鼻子区域掩码（可选使用）
    # 2) 皮肤问题检测（使用严格版本）
    detector = StrictSkinProblemDetector()

    # 预处理（增强对比、去噪）
    preprocessed = detector.preprocess_image(image)

    # 丘疹/痘印/白头粉刺/结节（合并一张）
    elevation_map = detector.calculate_elevation_map(preprocessed)
    acne_marks_pts = detector.detect_acne_marks(preprocessed, skin_mask)
    papules_pts = detector.detect_papules(preprocessed, skin_mask, elevation_map)
    whiteheads_pts = detector.detect_whiteheads(preprocessed, skin_mask)
    nodules_pts = detector.detect_nodules(preprocessed, skin_mask, elevation_map)

    # 黑头（单独一张）
    blackheads_pts = detector.detect_blackheads(preprocessed, nose_mask)

    # 毛孔（单独一张）
    pores_pts = detector.detect_pores(preprocessed, skin_mask)

    # 3) 在同一张图上使用不同颜色标记四类点，并输出三张图
    acne_image = image.copy()
    # 颜色使用 BGR：
    # 痘印=红色, 丘疹=绿色, 白头粉刺=黄色, 结节=蓝色
    color_map = {
        'acne_marks': (0, 0, 255),   # 红色
        'papules': (0, 255, 0),      # 绿色
        'whiteheads': (0, 255, 255), # 黄色
        'nodules': (255, 0, 0)       # 蓝色
    }

    for (x, y) in acne_marks_pts:
        cv2.circle(acne_image, (int(x), int(y)), 3, color_map['acne_marks'], -1)
    for (x, y) in papules_pts:
        cv2.circle(acne_image, (int(x), int(y)), 3, color_map['papules'], -1)
    for (x, y) in whiteheads_pts:
        cv2.circle(acne_image, (int(x), int(y)), 3, color_map['whiteheads'], -1)
    for (x, y) in nodules_pts:
        cv2.circle(acne_image, (int(x), int(y)), 3, color_map['nodules'], -1)

    blackheads_image = draw_points_on_image(image, blackheads_pts, color=(0, 0, 255), radius=3, thickness=-1)
    pores_image = draw_points_on_image(image, pores_pts, color=(0, 0, 255), radius=2, thickness=-1)

    # 生成图片字节流（总是生成，用于上传到MinIO）
    _, acne_encoded = cv2.imencode('.jpg', acne_image)
    acne_bytes = acne_encoded.tobytes()
    
    _, blackheads_encoded = cv2.imencode('.jpg', blackheads_image)
    blackheads_bytes = blackheads_encoded.tobytes()
    
    _, pores_encoded = cv2.imencode('.jpg', pores_image)
    pores_bytes = pores_encoded.tobytes()

    # 如果需要本地保存（从环境变量读取）
    save_local = os.getenv('SAVE_LOCAL_IMAGES', 'false').lower() == 'true'
    files_dict = {}
    
    if save_local and task_dir:  # 只在开关开启且有目录时保存
        acne_path = os.path.join(task_dir, "acne.jpg")
        cv2.imwrite(acne_path, acne_image)
        
        blackheads_path = os.path.join(task_dir, "blackheads.jpg")
        cv2.imwrite(blackheads_path, blackheads_image)
        
        pores_path = os.path.join(task_dir, "pores.jpg")
        cv2.imwrite(pores_path, pores_image)
        
        files_dict = {
            'acne.jpg': acne_path,
            'blackheads.jpg': blackheads_path,
            'pores.jpg': pores_path,
        }


    # 4) 输出 JSON 计数
    acne_group_points = acne_marks_pts + papules_pts + whiteheads_pts + nodules_pts
    result = {
        'counts': {
            "doudou":{
                'acne_marks': {
                    "name": "痘印",
                    "counts":len(acne_marks_pts),
                    "color":"(0, 0, 255)"
                    },
                'papules': { 
                    "name": "丘疹",
                    "counts":len(papules_pts),
                    "color":"(0, 255, 0)"},
                'whiteheads': { 
                    "name": "白头粉刺",
                    "counts":len(whiteheads_pts),
                    "color":"(0, 255, 255)"},
                'nodules': { 
                    "name": "结节",
                    "counts":len(nodules_pts),
                    "color":"(255, 0, 0)"},
                "description":"图片点位代表对应瑕疵",
                "suggestion": "白头粉刺、痘印、结节、丘疹等瑕疵较多，请注意清洁和祛痘"
            },
            'blackheads': {
                'name': '黑头',
                'counts': len(blackheads_pts),
                'color': '(0, 0, 255)',
                'suggestion':'皮肤光滑无黑头,注意做好清洁工作'
            },
            'pores': {
                'name': '毛孔',
                'counts': len(pores_pts),
                'color': '(0, 0, 255)',
                'suggestion':'无明显毛孔,平时注意防晒,做好保湿'
            },
            'acne_group_total': len(acne_group_points)
        },
        'files': files_dict,  # 本地文件路径（如果保存了）
        'image_bytes': {  # 新增：图片字节流
            'acne.jpg': acne_bytes,
            'blackheads.jpg': blackheads_bytes,
            'pores.jpg': pores_bytes,
        }
    }

    return result


