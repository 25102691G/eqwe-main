import math
from skimage.filters import frangi  # 需安装scikit-image：pip install scikit-image
import os
import numpy as np
import onnxruntime
import cv2
import json
from pathlib import Path
from api.services.mediapipe_compat import create_face_mesh, landmarks_to_mediapipe_result, synthesize_face_landmarks

class DarkCircleDetector:
    def __init__(self):
        # 使用预加载的MediaPipe FaceMesh实例
        from api.services.model_manager import get_model_manager
        model_manager = get_model_manager()
        self.face_mesh = model_manager.get_mediapipe_face_mesh()
        
        if self.face_mesh is None:
            # 如果模型管理器中没有，则创建新实例（兜底）
            self.face_mesh = create_face_mesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5
            )

        # 指定左右眼的关键点索引
        self.T_left_eye_landmarks = [464, 265, 443, 347]  # 左眼指定索引
        self.T_right_eye_landmarks = [244, 143, 223, 118]  # 右眼指定索引
    def get_points(self,landmark_indices,face_landmarks,w,h):        # 提取指定索引的关键点坐标
        points = []
        for idx in landmark_indices:
            # 将归一化坐标转换为像素坐标
            x = int(face_landmarks.landmark[idx].x * w)
            y = int(face_landmarks.landmark[idx].y * h)
            points.append((x, y))
        return points
    def get_bounding_box(self,points,w,h ,expand_ratio=0):
        x_coords = [p[0] for p in points]
        y_coords = [p[1] for p in points]

        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)

        # 计算扩展量（宽度和高度）
        width = x_max - x_min
        height = y_max - y_min

        # 扩展边界框（下方多扩展，适应黑眼圈分布）
        x_min = max(0, int(x_min - width * expand_ratio))
        x_max = min(w, int(x_max + width * expand_ratio))
        y_min = max(0, int(y_min - height * expand_ratio * 0.2))  # 上方少扩展
        y_max = min(h, int(y_max + height * expand_ratio * 0.6))  # 下方多扩展

        return (x_min, y_min, x_max - x_min, y_max - y_min)
    def get_eye_regions(self, image):
        """基于指定索引提取左右眼区域的边界框"""
        if self.face_mesh is None:
            points = synthesize_face_landmarks(image)
            if points is None:
                return None, None
            results = landmarks_to_mediapipe_result(points, image.shape)
        else:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_image)

        if not results.multi_face_landmarks:
            return None, None  # 未检测到人脸

        face_landmarks = results.multi_face_landmarks[0]
        h, w, _ = image.shape  # 图像高度和宽度
        # 获取左右眼的关键点
        left_eye_points =self.get_points(self.T_left_eye_landmarks,face_landmarks,w,h)
        right_eye_points = self.get_points(self.T_right_eye_landmarks,face_landmarks,w,h)

        # 根据关键点计算边界框（适当扩展以包含黑眼圈区域）

        left_bbox = self.get_bounding_box(left_eye_points,w,h)
        right_bbox = self.get_bounding_box(right_eye_points,w,h)

        return left_bbox, right_bbox

    def calculate_dark_score(self, image, bbox):
        """计算黑眼圈分数（0-10分，分数越高越严重）"""
        x, y, w, h = bbox
        eye_region = image[y:y + h, x:x + w]  # 提取眼眶区域
        if eye_region.size == 0:
            return 0.0

        # 转换到LAB色彩空间，使用亮度通道（更符合人眼感知）
        lab = cv2.cvtColor(eye_region, cv2.COLOR_BGR2LAB)
        l_channel = lab[:, :, 0]  # 亮度通道

        # 计算亮度特征
        avg_brightness = np.mean(l_channel)  # 平均亮度（0-255）
        dark_threshold = avg_brightness * 0.75  # 暗区域阈值（低于平均亮度65%）

        # 计算暗区域比例
        dark_pixels = np.sum(l_channel < dark_threshold)
        total_pixels = l_channel.size
        dark_ratio = dark_pixels / total_pixels if total_pixels > 0 else 0


        # 计算分数（综合亮度和暗区比例）
        brightness_score = max(0, min(10, (120 - avg_brightness) / 10))  # 亮度越低分数越高
        ratio_score = max(0, min(10, dark_ratio * 15))  # 暗区比例越高分数越高

        if ratio_score>4.5 and brightness_score>1:
            final_score=int(100-round((brightness_score * 0.5 + ratio_score * 0.5)/7, 2)*100)
        elif brightness_score==0:
            final_score=int(100-round(ratio_score/5,2)*100)
        else:
            final_score=int(100-round(ratio_score/7,2)*100)


        return final_score
        # 绘制矩形框和分数
    def draw_eye_info(self,bbox, result_image):
        x, y, w, h = bbox
        # 绘制红色矩形框
        cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 0, 255), 2)

    def process_image(self, image):
        """处理图像并返回结果"""
        # 读取图像
        if image is None:
            raise ValueError(f"图像为空: ")

        result_image = image.copy()

        # 获取指定索引的眼眶区域
        left_bbox, right_bbox = self.get_eye_regions(image)
        if not left_bbox or not right_bbox:
            print("未检测到有效的眼眶区域")
            return None

        # 计算分数
        left_score = self.calculate_dark_score(image, left_bbox)
        right_score = self.calculate_dark_score(image, right_bbox)

        # 绘制左右眼
        self.draw_eye_info(left_bbox,result_image)
        self.draw_eye_info(right_bbox, result_image)
        final_score=((left_score+right_score)*0.5)
        if final_score >= 70:
            suggest = "您几乎没有和眼圈，但要坚持注意休息保持良好习惯，做好护肤。"

        elif 50 < final_score < 70:
            # print("评价: 轻微黑眼圈")
            suggest = "您存在一些眼部问题，黑眼圈较为明显，可能是由于不良的生活习惯或年龄增长造成的，日常要坚持涂抹眼霜，保证睡眠时间。"

        else:
            # print("评价: 明显黑眼圈")
            suggest = "当前图像提示眼周暗沉、眼袋或细纹较明显，可能与睡眠、用眼、干燥或年龄变化有关。建议优先保证休息，做好眼周保湿、防晒和温和护理；如持续明显困扰，可咨询皮肤科或正规护理机构获取个性化建议。"

        return result_image, final_score,suggest


def softmax(x):
    return x / np.sum(np.exp(x), axis=-1, keepdims=True)  # , axis=-1


def Prenasolabial(img_list):
    """使用预加载的ONNX模型进行法令纹检测"""
    from api.services.model_manager import get_model_manager
    
    # 获取模型管理器
    model_manager = get_model_manager()
    model_info = model_manager.get_nasolabial_detector()
    
    if model_info is None:
        raise RuntimeError("法令纹检测模型未加载")
    
    sess1 = model_info['session']
    input_name = model_info['input_name']
    
    detect_list=[]

    for i in range(len(img_list)):
        height = np.array(img_list[i]).shape[0]
        weight = np.array(img_list[i]).shape[1]
        img = cv2.cvtColor(img_list[i], cv2.COLOR_BGR2RGB)
        w, h = 512, 512

        img = cv2.resize(img, (w, h)) / 255.  # 此处不太理解Image处理方式所以直接使用resize替换
        img = img.astype('float32')

        img = np.expand_dims(np.transpose(np.array(img, np.float32), (2, 0, 1)), 0)

        img = img.reshape([1, 3, 512, 512])
        output = sess1.run(None, {input_name: img})[0]

        pr = np.squeeze(output, axis=0)
        soft_max = softmax(pr.transpose((1, 2, 0)))

        pr = cv2.resize(soft_max, (weight, height), interpolation=cv2.INTER_LINEAR)
        pr = pr.argmax(axis=-1)

        set_image = (np.expand_dims(pr != 0, -1) * np.array(255, np.uint8)).astype('uint8')
        detect_list.append(set_image)
        b_img = cv2.bitwise_and(img_list[i], img_list[i], mask=set_image)

    return detect_list



def preprocess_face(image):
    """预处理：去噪和对比度增强"""
    # 1. 转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 2. 高斯模糊去噪（保留边缘的同时抑制噪声）
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    # 3. 对比度增强（限制对比度自适应直方图均衡化，避免过度增强噪声）
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_contrast = clahe.apply(blurred)

    return enhanced_contrast


def enhance_wrinkles(preprocessed):
    """增强皱纹：使用Frangi滤波器突出线性结构"""
    # Frangi滤波器对管状/线性结构敏感（适合皱纹、血管等）
    # 注意：输入需为float类型，且背景为暗、目标为亮
    frangi_img = frangi(
        preprocessed.astype(np.float32),
        sigmas=(1, 3),  # 皱纹可能的尺度范围（小到中等）
        scale_step=0.5,
        alpha=0.5,  # 控制对结构连续性的敏感度
        beta=15  # 控制对结构对比度的敏感度
    )

    # 归一化到0-255（便于后续处理）
    frangi_norm = cv2.normalize(frangi_img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)

    return frangi_norm



def detect_wrinkles(enhanced,block= None):
    """检测皱纹：边缘检测+形态学操作"""
    # 1. 边缘检测（Canny算子提取皱纹边缘）
    edges = cv2.Canny(enhanced, 50, 150)
    score=0
    thick_line=0
    fine_line=0
    area_now=enhanced.shape[0]*enhanced.shape[1]

    # 2. 形态学操作：连接断裂的皱纹并去除小噪点
    kernel = np.ones((2, 2), np.uint8)
    kernel_eroded = np.ones((3, 3), np.uint8)
    # 膨胀：连接邻近边缘
    # 腐蚀：细化边缘，去除毛刺

    dilated = cv2.dilate(edges, kernel, iterations=2)
    eroded = cv2.erode(dilated, kernel, iterations=3)

    white_pixels = np.sum(eroded == 255)

    contour,_=cv2.findContours(eroded, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contour_line_list=[]
    if block=="forehead":

        area_sum=0
        for cnt in contour:
            area=cv2.contourArea(cnt)
            area_sum+=area
            if area>area_now*0.0005 and area<area_now*0.0025:
                thick_line+=1

                contour_line_list.append(cnt)

            elif area<area_now*0.0001:
                continue
            elif area<area_now*0.0005 and area>area_now*0.0001:
                fine_line+=1

                contour_line_list.append(cnt)
        score = (area_sum / (eroded.shape[0] * eroded.shape[1])) / 0.4


    else:
        area_sum=0
        for cnt in contour:
            area=cv2.contourArea(cnt)
            area_sum+=area
            if area>area_now*0.003 and area<area_now*0.015:
                thick_line+=1
                contour_line_list.append(cnt)

            elif area<area_now*0.001:
                continue
            elif area<area_now*0.003  and area>area_now*0.001:
                fine_line+=1

                contour_line_list.append(cnt)
        score=(area_sum/(eroded.shape[0]*eroded.shape[1]))/0.5

    return eroded,score,fine_line,thick_line



def find_specific_coords(contours):
    """
    在所有轮廓中找到两个坐标：
    1. 最高y（y最小）对应的最大x坐标
    2. 最小x对应的最大y坐标
    :param contours: 所有轮廓的列表
    :return: (coord1, coord2)，格式为((x1,y1), (x2,y2))
    """
    if not contours:
        return None, None  # 无轮廓时返回空

    # 收集所有轮廓的点（转换为N×2数组）
    all_points = []
    for cnt in contours:
        points = cnt.reshape(-1, 2)  # 轮廓点从(N,1,2)转为(N,2)
        all_points.append(points)
    all_points = np.vstack(all_points)  # 合并为总点数×2的数组
    xs = all_points[:, 0]
    ys = all_points[:, 1]
    # 1. 最高y（y最小）对应的最大x坐标
    max_x = np.max(xs)  # 全局最小y值（最高处）
    min_x = np.min(xs)  # 全局最小x值（最左侧）
    cente_x=int((max_x+min_x)*0.5)
    candidates1 = all_points[(xs > cente_x-max_x*0.3) & (xs<=cente_x+max_x*0.3)]  # 所有y=min_y的点
    if len(candidates1) == 0:
        candidates1 = all_points
    if len(candidates1) == 0:
        return None, None
    min_y_idx = np.argmin(candidates1[:, 1])  # 这些点中x最大的索引

    coord1 = tuple(candidates1[min_y_idx])  # (max_x_at_min_y, min_y)

    candidates2 = all_points[(xs < min_x+int(max_x*0.2)) & (xs>=min_x)]  # 所有x=min_x的点
    if len(candidates2) == 0:
        candidates2 = all_points[xs < min_x]  # 所有x=min_x的点
    if len(candidates2) == 0:
        candidates2 = all_points
    if len(candidates2) == 0:
        return coord1, None
    max_y_idx = np.argmax(candidates2[:, 1])  # 这些点中y最大的索引
    coord2 = tuple(candidates2[max_y_idx])  # (min_x, max_y_at_min_x)

    return coord1, coord2


def find_specific_coords_r(contours):
    """
    在所有轮廓中找到两个坐标：
    1. 最高y（y最小）对应的最大x坐标
    2. 最小x对应的最大y坐标
    :param contours: 所有轮廓的列表
    :return: (coord1, coord2)，格式为((x1,y1), (x2,y2))
    """
    if not contours:
        return None, None  # 无轮廓时返回空

    # 收集所有轮廓的点（转换为N×2数组）
    all_points = []
    for cnt in contours:
        points = cnt.reshape(-1, 2)  # 轮廓点从(N,1,2)转为(N,2)
        all_points.append(points)
    all_points = np.vstack(all_points)  # 合并为总点数×2的数组
    xs = all_points[:, 0]
    ys = all_points[:, 1]
    min_x = np.min(xs)  # 全局最小x值
    max_x = np.max(xs)  # 全局最小x值
    min_y=np.min(ys)
    max_y=np.max(ys)
    center_x=(max_x+min_x)*0.5
    center_y=int(min_y+max_y)/2
    candidates1 = all_points[(xs > (min_x)) & (xs<=center_x)]  # 所有y=min_y的点
    if len(candidates1) == 0:
        candidates1 = all_points[(xs > (min_x)) & (xs <= max_x)]  # 所有y=min_y的点
    if len(candidates1) == 0:
        candidates1 = all_points
    if len(candidates1) == 0:
        return None, None

    min_y_idx = np.argmin(candidates1[:, 1])  # 这些点中x最大的索引

    coord1 = tuple(candidates1[min_y_idx])  # (max_x_at_min_y, min_y)

    # 2. 最大x对应的最大y坐标
    candidates2 = all_points[(xs >= (int(coord1[0]+max_x*0.1))) & (xs<=max_x)]  # 所有x=min_x的点
    if len(candidates2) == 0:
        candidates2 = all_points[(xs >= int(coord1[0]))]  # 所有x=min_x的点
    if len(candidates2) == 0:
        candidates2 = all_points
    if len(candidates2) == 0:
        return coord1, None

    max_y_idx = np.argmax(candidates2[:, 1])  # 这些点中y最大的索引
    coord2 = tuple(candidates2[max_y_idx])  # (min_x, max_y_at_min_x)

    return coord1, coord2

# def noise_mask_left(mask,img):
#     mask_copy=mask.copy()
#     mask_copy[:mask_copy.shape[0],int(mask_copy.shape[1]*0.9):]=0
#     contours, _ = cv2.findContours(mask_copy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     result = np.zeros_like(img)
#     all_area=0
#     for cnt in contours:
#         all_area+=cv2.contourArea(cnt)
#     score_line=(all_area+mask.shape[0]*mask_copy.shape[1]*0.1)/(mask_copy.shape[0]*mask_copy.shape[1])
#
#     highest_pt, lowest_pt = find_specific_coords(contours)
#
#     thresh_state=None
#     if score_line > 0.13 and score_line < 0.2:
#         cv2.line(result, highest_pt, lowest_pt, (255, 0, 0), 1)
#         thresh_state = "middle"
#     elif score_line < 0.13:
#         cv2.line(result, highest_pt, lowest_pt, (255, 0, 0), 1)
#         thresh_state = "lower"
#     else:
#         cv2.line(result, highest_pt, lowest_pt, (255, 0, 0), 1)
#         thresh_state = "seriousness"
#
#     result=cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
#
#
#
#     return result,thresh_state


# def noise_mask_right(mask,img):
#     mask_copy=mask.copy()
#
#     mask_copy[:mask_copy.shape[0],0:int(mask_copy.shape[1]*0.2)]=0
#     contours, _ = cv2.findContours(mask_copy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     result = np.zeros_like(img)
#     all_area = 0
#     for cnt in contours:
#         all_area += cv2.contourArea(cnt)
#     score_line = (all_area + mask_copy.shape[0] * mask_copy.shape[1] * 0.1) / (mask_copy.shape[0] * mask_copy.shape[1])
#     #
#     highest_pt, lowest_pt = find_specific_coords_r(contours)
#     thresh_state = None
#
#
#     if score_line > 0.13 and score_line < 0.2:
#         cv2.line(result, highest_pt, lowest_pt, (255, 0, 0), 1)
#         thresh_state = "medium"
#     elif score_line < 0.13:
#         cv2.line(result, highest_pt, lowest_pt, (255, 0, 0), 1)
#         thresh_state = "mild"
#     else:
#         cv2.line(result, highest_pt, lowest_pt, (255, 0, 0), 1)
#         thresh_state = "seriousness"
#     result=cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
#
#
#     return result,thresh_state

# def detect_single_face(img: np.ndarray):
#     scale = 1.1
#     neighbors = 5
#     min_size = 60
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
#     face_cascade = cv2.CascadeClassifier(cascade_path)
#     if face_cascade.empty():
#         raise RuntimeError(f"无法加载人脸分类器: {cascade_path}")
#     faces = face_cascade.detectMultiScale(
#         gray,
#         scaleFactor=scale,
#         minNeighbors=neighbors,
#         minSize=(min_size, min_size)
#     )
#     if len(faces) == 0:
#         return None, gray
#     largest = max(faces, key=lambda r: r[2] * r[3])
#     return largest, gray


def grabcut_face_mask(bgr_roi: np.ndarray) -> np.ndarray:
    h, w = bgr_roi.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    rect = (1, 1, max(1, w - 2), max(1, h - 2))
    cv2.grabCut(bgr_roi, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    mask_out = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype('uint8')
    return mask_out


def detect_facemesh_points(bgr_roi: np.ndarray):
    """使用预加载的MediaPipe FaceMesh检测面部关键点"""
    from api.services.model_manager import get_model_manager
    
    model_manager = get_model_manager()
    face_mesh = model_manager.get_mediapipe_face_mesh()
    
    if face_mesh is None:
        # 兜底：如果模型管理器中没有，则创建临时实例
        face_mesh = create_face_mesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)
        if face_mesh is None:
            points = synthesize_face_landmarks(bgr_roi)
            return None if points is None else np.array(points, dtype=np.int32)
        with face_mesh:
            rgb = cv2.cvtColor(bgr_roi, cv2.COLOR_BGR2RGB)
            res = face_mesh.process(rgb)
            if not res.multi_face_landmarks:
                return None
            h, w = bgr_roi.shape[:2]
            landmarks = []
            for lm in res.multi_face_landmarks[0].landmark:
                x = int(lm.x * w)
                y = int(lm.y * h)
                landmarks.append((x, y))
            return np.array(landmarks, dtype=np.int32)
    
    # 使用共享实例
    rgb = cv2.cvtColor(bgr_roi, cv2.COLOR_BGR2RGB)
    res = face_mesh.process(rgb)
    if not res.multi_face_landmarks:
        return None
    h, w = bgr_roi.shape[:2]
    landmarks = []
    for lm in res.multi_face_landmarks[0].landmark:
        x = int(lm.x * w)
        y = int(lm.y * h)
        landmarks.append((x, y))
    return np.array(landmarks, dtype=np.int32)





def _clamp_rect(x0, y0, x1, y1, w, h):
    x0 = max(0, min(w - 1, int(x0)))
    y0 = max(0, min(h - 1, int(y0)))
    x1 = max(0, min(w - 1, int(x1)))
    y1 = max(0, min(h - 1, int(y1)))
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0
    return x0, y0, x1, y1


def _rect_from_center(cx, cy, rw, rh, w, h):
    x0 = cx - rw // 2
    y0 = cy - rh // 2
    x1 = cx + rw // 2
    y1 = cy + rh // 2
    return _clamp_rect(x0, y0, x1, y1, w, h)


def _forehead_from_top_rows(points: np.ndarray, h: int, w: int):
    ys = points[:, 1]
    order = np.argsort(ys)
    row_h = max(2, int(0.025 * h))
    min_y = int(ys.min())
    mask_top = (ys <= min_y + 3 * row_h)
    top_pts = points[mask_top]
    if top_pts.shape[0] < 10:
        N = min(60, points.shape[0])
        top_pts = points[order[:N]]
    x0 = int(top_pts[:, 0].min())
    y0 = int(top_pts[:, 1].min())
    x1 = int(top_pts[:, 0].max())
    y1 = int(top_pts[:, 1].max())
    y0 = max(0, y0 - int(0.02 * h))
    return _clamp_rect(x0, y0, x1, y1, w, h)



def get_face_regions(points: np.ndarray, roi_shape):
    h, w = roi_shape
    left_eye_outer, left_eye_inner = 33, 133
    right_eye_outer, right_eye_inner = 362, 263
    left_lower_eyelid = [145, 159]
    right_lower_eyelid = [374, 386]
    left_cheek_center = 234
    right_cheek_center = 454
    left_alar = 64
    right_alar = 294
    nose_tip = 1
    left_mouth_corner = 57#61
    right_mouth_corner = 287#291

    def pts(idx_list):
        arr = points[np.array(idx_list)]
        return arr

    forehead_rect = _forehead_from_top_rows(points, h, w)

    L_outer = points[left_eye_outer]
    L_inner = points[left_eye_inner]
    L_lid_y = int(pts(left_lower_eyelid)[:, 1].mean())
    eye_pad = int(0.05 * w)
    left_eye_bag_h = int(0.12 * h)
    left_eye_bag_rect = _clamp_rect(min(L_outer[0], L_inner[0]) - eye_pad,
                                    L_lid_y + int(0.01 * h),
                                    max(L_outer[0], L_inner[0]) + eye_pad,
                                    L_lid_y + left_eye_bag_h,
                                    w, h)

    R_outer = points[right_eye_outer]
    R_inner = points[right_eye_inner]
    R_lid_y = int(pts(right_lower_eyelid)[:, 1].mean())
    right_eye_bag_h = int(0.12 * h)
    right_eye_bag_rect = _clamp_rect(min(R_outer[0], R_inner[0]) - eye_pad,
                                     R_lid_y + int(0.01 * h),
                                     max(R_outer[0], R_inner[0]) + eye_pad,
                                     R_lid_y + right_eye_bag_h,
                                     w, h)


    crow_rw = int(0.10 * w)
    crow_rh = int(0.10 * h)
    dx_pad = int(0.05 * w)  # 水平向外偏移，确保不进入眼睛区域

    left_crow_anchor = points[33]
    # 使用下眼睑平均 y，使矩形更靠近外眼角下方，减少进入眼睛的可能
    L_lid_y = int(pts(left_lower_eyelid)[:, 1].mean())
    l_center_y = L_lid_y + int(0.01 * h)
    # 仅向左扩展：右边界为锚点x - dx_pad，左边界为右边界 - crow_rw
    lcx1 = int(left_crow_anchor[0] - dx_pad)
    lcx0 = lcx1 - crow_rw
    lcy0 = int(l_center_y - crow_rh // 2)
    lcy1 = int(l_center_y + crow_rh // 2)
    left_crow_foot_rect = _clamp_rect(lcx0, lcy0, lcx1, lcy1, w, h)

    right_crow_anchor = points[263]
    R_lid_y = int(pts(right_lower_eyelid)[:, 1].mean())
    r_center_y = R_lid_y + int(0.01 * h)
    # 仅向右扩展：左边界为锚点x + dx_pad，右边界为左边界 + crow_rw
    rcx0 = int(right_crow_anchor[0] + dx_pad)
    rcx1 = rcx0 + crow_rw
    rcy0 = int(r_center_y - crow_rh // 2)
    rcy1 = int(r_center_y + crow_rh // 2)
    right_crow_foot_rect = _clamp_rect(rcx0, rcy0, rcx1, rcy1, w, h)

    LC = points[left_cheek_center]
    cheek_rw = int(0.20 * w)
    cheek_rh = int(0.16 * h)
    left_cheek_rect = _rect_from_center(LC[0], LC[1] + int(0.04 * h), cheek_rw, cheek_rh, w, h)

    RC = points[right_cheek_center]
    right_cheek_rect = _rect_from_center(RC[0], RC[1] + int(0.04 * h), cheek_rw, cheek_rh, w, h)

    # 嘴角区域：以嘴角关键点为中心的小矩形（略向下偏移）
    LM = points[left_mouth_corner]
    RM = points[right_mouth_corner]
    mouth_rw = int(0.08 * w)
    mouth_rh = int(0.08 * h)
    left_mouth_corner_rect = _rect_from_center(LM[0], LM[1] + int(0.03 * h), mouth_rw, mouth_rh, w, h)
    right_mouth_corner_rect = _rect_from_center(RM[0], RM[1] + int(0.03 * h), mouth_rw, mouth_rh, w, h)

    LA = points[left_alar]
    RA = points[right_alar]
    NT = points[nose_tip]
    nose_mid_y = int((NT[1] + LA[1] + RA[1]) / 3)
    base_rw = int(0.05 * w)
    expand = int(0.06 * w)
    up_expand = int(0.1 * h)
    down_expand = int(0.08 * h)
    nose_top = min(LA[1], RA[1]) - up_expand
    nose_bottom = nose_mid_y + down_expand
    lx0 = LA[0] - base_rw // 2 - expand
    ly0 = nose_top
    lx1 = LA[0] + base_rw // 2
    ly1 = nose_bottom
    left_nose_wing_rect = _clamp_rect(lx0, ly0, lx1, ly1, w, h)
    rx0 = RA[0] - base_rw // 2
    ry0 = nose_top
    rx1 = RA[0] + base_rw // 2 + expand
    ry1 = nose_bottom
    right_nose_wing_rect = _clamp_rect(rx0, ry0, rx1, ry1, w, h)

    return {
        'forehead': forehead_rect,
        'left_eye_bag': left_eye_bag_rect,#眼袋
        'right_eye_bag': right_eye_bag_rect,
        'left_crow_foot': left_crow_foot_rect,
        'right_crow_foot': right_crow_foot_rect,
        'left_cheek': left_cheek_rect,
        'right_cheek': right_cheek_rect,
        'left_mouth_corner': left_mouth_corner_rect,
        'right_mouth_corner': right_mouth_corner_rect,
        'left_nose_wing_exp': left_nose_wing_rect,
        'right_nose_wing_exp': right_nose_wing_rect,
    }


# ====== 纹理检测工具函数 ======

# def _edges_from_sobel(gray: np.ndarray):
#     gx = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=3)
#     gy = cv2.Sobel(gray, cv2.CV_16S, 0, 1, ksize=3)
#     abs_gx = cv2.convertScaleAbs(gx)
#     abs_gy = cv2.convertScaleAbs(gy)
#     grad = cv2.addWeighted(abs_gx, 0.5, abs_gy, 0.5, 0)
#     sobel_bin = ((grad > 8) & (grad<20)).astype(np.uint8) * 255
#
#     edges = cv2.morphologyEx(sobel_bin, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
#
#     grad = cv2.GaussianBlur(edges, (3, 3), 0.8)
#
#     edges = cv2.Canny(grad, 8, 20, L2gradient=True)
#     return edges
#
#
# def _horizontal_lines(edges: np.ndarray):
#     gy = cv2.Sobel(edges, cv2.CV_32F, 0, 1, ksize=3)
#     gx = cv2.Sobel(edges, cv2.CV_32F, 1, 0, ksize=3)
#     mag = cv2.magnitude(gx, gy)
#     eps = 1e-6
#     mask = (np.abs(gx) < 0.5 * (np.abs(gy) + eps)) & (mag > 10)
#     out = np.zeros_like(edges)
#     out[mask] = 255
#     out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1)))
#     return out
#
#
# def _vertical_lines(edges: np.ndarray):
#     gy = cv2.Sobel(edges, cv2.CV_32F, 0, 1, ksize=3)
#     gx = cv2.Sobel(edges, cv2.CV_32F, 1, 0, ksize=3)
#     mag = cv2.magnitude(gx, gy)
#
#     eps = 1e-6
#     mask = (np.abs(gy) < 0.5 * (np.abs(gx) + eps)) & (mag > 10)
#     out = np.zeros_like(edges)
#     out[mask] = 255
#     out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 9)))
#     return out
#
#
# def _wavy_lines(edges: np.ndarray):
#     work = edges.copy()
#     lines = cv2.HoughLinesP(work, 1, np.pi / 180, threshold=60, minLineLength=30, maxLineGap=5)
#     straight_mask = np.zeros_like(work)
#     if lines is not None:
#         for l in lines:
#             x1, y1, x2, y2 = l[0]
#             cv2.line(straight_mask, (x1, y1), (x2, y2), 255, 2)
#     straight_mask = cv2.dilate(straight_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
#     wavy = cv2.bitwise_and(work, cv2.bitwise_not(straight_mask))
#
#     wavy = cv2.morphologyEx(wavy, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
#     return wavy
#
#
# def _overlay_edges_on_region(region_bgr: np.ndarray, edges: np.ndarray):
#     color_edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
#     horiz = _horizontal_lines(edges)
#     vert = _vertical_lines(edges)
#     wavy = _wavy_lines(edges)
#     color_edges[:, :, 1:] = 0            # 红
#     color_horiz = cv2.cvtColor(horiz, cv2.COLOR_GRAY2BGR); color_horiz[:, :, 0] = 0; color_horiz[:, :, 2] = 0  # 绿
#     color_vert = cv2.cvtColor(vert, cv2.COLOR_GRAY2BGR); color_vert[:, :, 0] = 0  # 黄(红+绿)
#     color_wavy = cv2.cvtColor(wavy, cv2.COLOR_GRAY2BGR); color_wavy[:, :, :2] = 0  # 蓝
#     overlay = cv2.addWeighted(color_edges, 1.0, color_horiz, 1.0, 0)
#     overlay = cv2.addWeighted(overlay, 1.0, color_vert, 1.0, 0)
#     overlay = cv2.addWeighted(overlay, 1.0, color_wavy, 1.0, 0)
#     mix = cv2.addWeighted(region_bgr, 1.0, overlay, 0.8, 0)
#     return mix


def split_three_sections(image, landmarks):
    pt10 = landmarks[10]
    pt151 = landmarks[151]
    pt234 = landmarks[234]
    pt454 = landmarks[454]
    pt2 = landmarks[2]
    pt195 = landmarks[195]
    pt152 = landmarks[152]
    pt5=landmarks[5]
    length=np.linalg.norm(np.array(pt10) + np.array(pt151))*(7/3)
    direction = np.array(pt10) - np.array(pt151)
    direction = direction / np.linalg.norm(direction)
    forehead_point = (np.array(pt10) + direction * length).astype(int)

    rects = [
        ("top", forehead_point[1], landmarks[9][1]),
        ("middle", landmarks[9][1], pt5[1]),
        ("bottom", pt5[1], pt152[1])
    ]

    outputs = []
    for name, y1, y2 in rects:
        section_img= image[y1:y2, :,:]
        # 仅保留该分割区域的最小外接矩形，尽量减少黑色背景
        outputs.append(section_img)
    outputs.append(image)
    return outputs,rects

def connect_three_sections(bgr_roi_img,img_list, rects):
    # img_bgr=bgr_roi_img.copy()
    # i=0

    img_bgr = cv2.vconcat(img_list[:3])
    if img_bgr.shape != img_list[3].shape:
        img_bgr=cv2.resize(img_bgr,(img_list[3].shape[1],img_list[3].shape[0]))
    img_bgr=cv2.add(img_bgr, img_list[3])

        # 仅保留该分割区域的最小外接矩形，尽量减少黑色背景
    return img_bgr


def zhang_suen_thinning(binary_img):
    """
    Zhang-Suen细化算法提取骨架（中线）- 优化版本
    :param binary_img: 二值图像（0为背景，255为前景）
    :return: 骨架图像（单像素宽度）
    """
    # 转换为二值图像（0和1）
    img = (binary_img > 0).astype(np.uint8)
    h, w = img.shape
    
    # 预先计算8邻域的偏移
    # P2, P3, P4, P5, P6, P7, P8, P9 (顺时针从上方开始)
    neighbors_offset = [
        (-1, 0), (-1, 1), (0, 1), (1, 1),
        (1, 0), (1, -1), (0, -1), (-1, -1)
    ]
    
    changed = True
    iteration = 0
    max_iterations = 100  # 防止无限循环
    
    while changed and iteration < max_iterations:
        changed = False
        iteration += 1
        
        # 第一步和第二步
        for step in [1, 2]:
            # 找到所有前景像素的位置
            foreground = np.argwhere(img[1:-1, 1:-1] == 1)
            if len(foreground) == 0:
                break
            
            # 调整坐标（因为我们从[1:-1, 1:-1]开始）
            foreground = foreground + 1
            
            to_delete = []
            
            for i, j in foreground:
                # 获取8邻域
                p2 = img[i-1, j]
                p3 = img[i-1, j+1]
                p4 = img[i, j+1]
                p5 = img[i+1, j+1]
                p6 = img[i+1, j]
                p7 = img[i+1, j-1]
                p8 = img[i, j-1]
                p9 = img[i-1, j-1]
                
                neighbors = [p2, p3, p4, p5, p6, p7, p8, p9]
                
                # 条件A: 2 <= B(P1) <= 6 (邻域中前景像素数)
                b_p1 = sum(neighbors)
                if not (2 <= b_p1 <= 6):
                    continue
                
                # 条件B: A(P1) = 1 (0->1跳变次数)
                a_p1 = 0
                for k in range(8):
                    if neighbors[k] == 0 and neighbors[(k+1) % 8] == 1:
                        a_p1 += 1
                if a_p1 != 1:
                    continue
                
                # 条件C和D（根据步骤不同）
                if step == 1:
                    # P2 * P4 * P6 = 0 and P4 * P6 * P8 = 0
                    if (p2 * p4 * p6 == 0) and (p4 * p6 * p8 == 0):
                        to_delete.append((i, j))
                        changed = True
                else:  # step == 2
                    # P2 * P4 * P8 = 0 and P2 * P6 * P8 = 0
                    if (p2 * p4 * p8 == 0) and (p2 * p6 * p8 == 0):
                        to_delete.append((i, j))
                        changed = True
            
            # 批量删除
            for i, j in to_delete:
                img[i, j] = 0
    
    # 转换回255
    return (img * 255).astype(np.uint8)
def create_region_mask(image, landmarks, region_indices):
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
        cv2.fillPoly(image, [region_points], 0)

    return image

def nasolabial_process(img):
    dict_score = {}
    # img=img[600:1800,:,:],save_path

    if img is None:
        raise FileNotFoundError(f"无法读取图片:")

    img_Change=img.copy()
    # face, gray = detect_single_face(img)
    #
    # k_size = (5, 5)
    # # 标准差（控制模糊程度，0表示自动计算）
    # sigma = 0
    # blurred = cv2.GaussianBlur(gray, k_size, sigma)
    # thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 5, 2)
    #
    # if face is None:
    #     print('未检测到人脸。')
    #     return
    #
    # x, y, w, h = face
#鼻子处掩膜，相较赵晨掩膜点有更改
    nose_mask_point= [ 98, 97, 2, 326, 327, 358, 429, 437, 399, 419,351,122 ,114,126, 49, 129, 98]
    #将人脸检测直接更为关键点，拿关键点去框人脸再进行检测
    points = detect_facemesh_points(img_Change)
    gray = cv2.cvtColor(img_Change, cv2.COLOR_BGR2GRAY)
    if points[10][1] > points[8][1] - points[10][1]:
        y = points[10][1] - (points[8][1] - points[10][1])
        h = points[152][1] - points[10][1] + (points[8][1] - points[10][1])
        x = points[127][0]
        w = points[356][0] - points[127][0]
    else:
        y = 0
        h = points[152][1] - points[10][1] + points[10][1]
        x = points[127][0]
        w = points[356][0] - points[127][0]

    face_roi_bgr = img_Change[y:y + h, x:x + w]
    face_black_eye_bgr = face_roi_bgr.copy()
    face_roi_gray = gray[y:y + h, x:x + w]
    points = detect_facemesh_points(face_roi_bgr)
    Three_Courts, rect_tmb = split_three_sections(face_roi_bgr, points)
    thresh_list = Prenasolabial(Three_Courts)

    # 当前用于做面部皱纹细线直接检测
    face_pre_binary = connect_three_sections(face_roi_bgr, thresh_list, rect_tmb)
    #鼻子掩膜去除鼻子皱纹
    face_pre_binary=create_region_mask(face_pre_binary, points, nose_mask_point)

    # 备份计算区域分数，皱纹数量
    face_pre_binary_score = face_pre_binary.copy()

    ####更新 增加了整个面部的皱纹画线
    kernel = np.ones((3, 3), np.uint8)
    face_pre_binary = cv2.erode(face_pre_binary, kernel, iterations=1)

    contour, _ = cv2.findContours(face_pre_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    black_img = np.zeros_like(face_pre_binary, np.uint8)
    for cnt in contour:
        if cv2.contourArea(cnt) > (black_img.shape[0] * black_img.shape[1] * 0.0001):
            cv2.drawContours(black_img, [cnt], 0, 255, cv2.FILLED)
    vis = img.copy()
    face_pre_connect = cv2.bitwise_and(face_roi_bgr, face_roi_bgr, mask=face_pre_binary)

    preprocessed = preprocess_face(face_pre_connect)
    # 步骤2：皱纹增强
    enhanced = enhance_wrinkles(preprocessed)
    # 步骤3：皱纹检测
    line_binary = zhang_suen_thinning(black_img)
    result = face_roi_bgr.copy()
    result[line_binary == 255] = [148, 29, 67]  # 用红色标记皱纹

    ###更新代码 result
    thresh_state = None
    line_num_thick = 0
    line_num_fine = 0
    line_num_all = 0

    if points is not None:

        regions = get_face_regions(points, (h, w))
        overlay_final = vis.copy()
        for name, (rx0, ry0, rx1, ry1) in regions.items():
            # 取区域裁剪
            region_bgr = face_roi_bgr[ry0:ry1, rx0:rx1]
            region_gray = face_roi_gray[ry0:ry1, rx0:rx1]
            region_band = face_pre_connect[ry0:ry1, rx0:rx1]
            # region_mask = mask[ry0:ry1, rx0:rx1]
            region_pre = face_pre_binary_score[ry0:ry1, rx0:rx1]
            if region_bgr.size == 0:
                continue
            if name == "left_nose_wing_exp" and len((cv2.findContours(region_pre, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0])) == 0:

                cv2.line(face_roi_bgr, (points[209][0], points[209][1]), (points[206][0], points[206][1]), (255, 0, 0),
                         2)
                result1 = face_roi_bgr[ry0:ry1, rx0:rx1]
                line_num_fine = line_num_fine + 1
                dict_score[name] = 0

            elif name == "left_nose_wing_exp" and len(
                    (cv2.findContours(region_pre, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0])) > 0:

                white_pixels = np.sum(region_pre == 255)
                score = (white_pixels / (region_pre.shape[0] * region_pre.shape[1])) / 0.5

                contours, _ = cv2.findContours(region_pre, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                dict_score[name] = score
                line_num_thick = line_num_thick + 1

                point1, point2 = find_specific_coords(contours)
                result1 = region_bgr.copy()
                if point1 is None or point2 is None:
                    continue
                if score <= 0.3 and score >= 0:

                    cv2.line(result1, point1, point2, (255, 0, 0), 3)
                elif score > 0.3 and score <= 0.6:
                    cv2.line(result1, point1, point2, (255, 0, 0), 6)

                else:
                    cv2.line(result1, point1, point2, (255, 0, 0), 9)


            elif name == "right_nose_wing_exp" and len(
                    (cv2.findContours(region_pre, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0])) == 0:


                cv2.line(face_roi_bgr, (points[429][0], points[429][1]), (points[426][0], points[426][1]), (255, 0, 0),
                         2)
                result1 = face_roi_bgr[ry0:ry1, rx0:rx1]
                line_num_fine = line_num_fine + 1
                dict_score[name] = 0
            elif name == "right_nose_wing_exp" and len(
                    (cv2.findContours(region_pre, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0])) > 0:
                white_pixels = np.sum(region_pre == 255)
                score = (white_pixels / (region_pre.shape[0] * region_pre.shape[1])) / 0.6
                contours, _ = cv2.findContours(region_pre, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                result1 = region_bgr.copy()
                dict_score[name] = score
                line_num_thick = line_num_thick + 1

                point1, point2 = find_specific_coords_r(contours)
                if point1 is None or point2 is None:
                    continue
                if score <= 0.3 and score >= 0:

                    cv2.line(result1, point1, point2, (255, 0, 0), 3)
                elif score > 0.3 and score <= 0.6:
                    cv2.line(result1, point1, point2, (255, 0, 0), 6)

                else:
                    cv2.line(result1, point1, point2, (255, 0, 0), 9)

            else:
                contour_thick = 0
                contour_fine = 0

                preprocessed = preprocess_face(region_band)
                # 步骤2：皱纹增强
                enhanced = enhance_wrinkles(preprocessed)
                # 步骤3：皱纹检测
                line_binary = zhang_suen_thinning(region_pre)
                result1 = region_bgr.copy()
                result1[line_binary == 255] = [148, 29, 67]  # 用红色标记皱纹
                contours, _ = cv2.findContours(region_pre, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                wrinkles_mask, score_get, af_line, at_line = detect_wrinkles(enhanced, name)
                picture_area = region_pre.shape[0] * region_pre.shape[1]
                for cnt in contours:
                    area = cv2.contourArea(cnt)
                    if area > picture_area * 0.01:
                        contour_thick += 1
                    elif area < picture_area * 0.01 and area > picture_area * 0.005:
                        contour_fine += 1
                dict_score[name] = score_get
                line_num_thick += contour_thick
                line_num_fine += contour_fine
            # 写回到原图对应位置
            if name == "right_nose_wing_exp" or name == "left_nose_wing_exp":
                result[ry0:ry1, rx0:rx1] = result1
                X0, Y0, X1, Y1 = rx0 + x, ry0 + y, rx1 + x, ry1 + y
                overlay_final[Y0:Y1, X0:X1] = result1
        line_num_all = line_num_thick + line_num_fine
        overlay_final = overlay_final[y:y + h, x:x + w]

        # 使用全局DarkCircleDetector实例（避免重复创建）
        detector = get_dark_circle_detector()
        black_eye_img, final_score, suggest = detector.process_image(face_black_eye_bgr)

        dic={"wrinkles":{
                "name":"皱纹",
                "description":"图片线条代表皱纹位置",
                "Statistics_and_Color":{
                    "coarse_texture":line_num_thick,
                    "microgroove":line_num_fine,
                    "coarse_texture_color":(148, 29,67),
                    "microgroove_color":(148, 29,67)},
                "suggest":"您属于年轻肌，面部几乎没有皱纹、下垂、凹陷等衰老迹象。您的皮肤胶原蛋白饱满，但是随着年龄增长，胶原蛋白和皮下脂肪流失，细纹和皱纹会随之增加",
                "Histogram_matrix":{
                    "color":{"l_color":(255,0,0),
                        "r_color":(236,123,101)},
                    "Severity classification":{"轻度":"<0.33","中度":"<0.66","重度":">0.66"},
                    "l_score_color":{
                        "cheekline_wrinkle_score_L":dict_score["left_mouth_corner"],
                        "eye_bag_lines_L":dict_score["left_eye_bag"],
                        "nasolabial_folds_L":dict_score["left_nose_wing_exp"],
                        "crow_feet_L":dict_score["left_crow_foot"]},
                    "r_score_color": {
                        "cheekline_wrinkle_score_R":dict_score["right_mouth_corner"],
                        "eye_bag_lines_R":dict_score["right_eye_bag"],
                        "nasolabial_folds_R":dict_score["right_nose_wing_exp"],
                        "crow_feet_R":dict_score["right_crow_foot"]}},
                "wrinkles_score":"",
            },
            "black_eye":{
                "description":"图片颜色代表对应区域",
                "position_score":{"color":(0,0,255),"talk":"黑眼圈得分","score":final_score},
                "suggest":{"talk":"建议：","talk_suggest":suggest},
                },
            }
        # with open(save_path, "w", encoding='utf-8') as file:
        # # 将字典转换为JSON格式并写入文件
        #     json.dump(dic, file, indent=4,ensure_ascii=False)
        return result,black_eye_img,dic

    else:
        print('未检测到人脸关键点。')
        return None


# ============================================================
# 服务实例复用优化：创建全局DarkCircleDetector实例
# ============================================================
_dark_circle_detector = None

def get_dark_circle_detector():
    """获取全局DarkCircleDetector实例（单例模式）"""
    global _dark_circle_detector
    if _dark_circle_detector is None:
        _dark_circle_detector = DarkCircleDetector()
    return _dark_circle_detector
# ============================================================
