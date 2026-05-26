import time
import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops
import torch
from torchvision import transforms
from PIL import Image
import torch.nn as nn
from pathlib import Path
#使用方法：
#调用process_image（）函数 传入图片 返回标注好痣的图片
# 常用版本的面部外轮廓（face oval）索引

def _make_divisible(ch, divisor=8, min_ch=None):
    """
    :param ch: 输入特征矩阵的channel
    :param divisor: 基数
    :param min_ch: 最小通道数
    """
    if min_ch is None:
        min_ch = divisor
    #   将ch调整到距离8最近的整数倍
    #   int(ch + divisor / 2) // divisor 向上取整
    new_ch = max(min_ch, int(ch + divisor / 2) // divisor * divisor)
    #   确保向下取整时不会减少超过10%
    if new_ch < 0.9 * ch:
        new_ch += divisor
    return new_ch


#   定义 卷积-BN-ReLU6 联合操作
class ConvBNReLU(nn.Sequential):
    #   PyTorch中DW卷积通过调用 nn.Conv2d() 来实现
    #   参数 (groups=1) 为普通卷积，参数 (groups=输入特征矩阵的深度) 为DW卷积
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU6(inplace=True)
        )


#   倒残差结构
class InvertedResidual(nn.Module):
    #   expand_ratio:扩展因子(t)
    def __init__(self, in_channel, out_channel, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        #   定义隐层，对应第一层的输出通道数 (tk)
        hidden_channel = in_channel * expand_ratio
        #   当stride=1且输入特征矩阵与输出特征矩阵shape相同是才有shortcut
        self.use_shotcut = stride == 1 and in_channel == out_channel

        layers = []
        if expand_ratio != 1:
            #   1x1 pointwise conv
            layers.append(ConvBNReLU(in_channel, hidden_channel, kernel_size=1))
        layers.extend([
            #   3x3 depthwise conv
            ConvBNReLU(hidden_channel, hidden_channel, stride=stride, groups=hidden_channel),
            #   1x1 pointwise conv(linear)  linear:不添加激活函数就等于线性函数
            nn.Conv2d(hidden_channel, out_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channel),
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_shotcut:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    #   alpha:用来控制卷积层中所使用卷积核个数的参数
    def __init__(self, num_classes=1000, alpha=1.0, round_nearest=8):
        super(MobileNetV2, self).__init__()
        #   初始化倒残差模块
        block =InvertedResidual
        #   通过_make_divisible将卷积核个数调整为8的整数倍
        input_channel = _make_divisible(32 * alpha, round_nearest)
        last_channel = _make_divisible(1280 * alpha, round_nearest)

        #   创建参数列表
        inverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        features = []
        features.append(ConvBNReLU(3, input_channel, stride=2))
        #   定义一系列block结构
        for t, c, n, s in inverted_residual_setting:
            #   调整输出通道数
            output_channel = _make_divisible(c * alpha, round_nearest)
            #   重复倒残差结构
            #   第一层：stride=n  其它层：stride=1
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel
        #   定义最后一个卷积层
        features.append(ConvBNReLU(input_channel, last_channel, 1))
        #   特征提取层
        self.features = nn.Sequential(*features)

        #   分类器部分
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(last_channel, num_classes)
        )

        #   初始化权重
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def predict_main(img):
    """使用预加载的模型进行预测"""
    from api.services.model_manager import get_model_manager
    
    # 获取模型管理器
    model_manager = get_model_manager()
    model_info = model_manager.get_nevus_classifier()
    
    if model_info is None:
        raise RuntimeError("痣分类模型未加载")
    
    model = model_info['model']
    device = model_info['device']
    class_indict = model_info['class_dict']
    
    # 数据预处理
    data_transform = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # [N, C, H, W]
    img = data_transform(img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)
    
    # prediction (模型已经在eval模式)
    with torch.no_grad():
        # predict class
        output = torch.squeeze(model(img.to(device))).cpu()
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()

    return class_indict[str(predict_cla)]



# 多层特征信息融合，根据比例融合比对求出得分
def score_map_from_roi(bgr_roi):
    """
    基于 ROI 计算像素级干燥分数图 (0~1)。
    指标（细化纹理）:
    - 梯度幅值（Scharr）
    - 多尺度 Laplacian 高频
    - 局部标准差
    - 颜色: HSV 的 S、V（越低越干）
    """
    hsv = cv2.cvtColor(bgr_roi, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1].astype(np.float32) / 255.0
    v = hsv[:, :, 2].astype(np.float32) / 255.0

    gray = cv2.cvtColor(bgr_roi, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

    # 纹理多特征
    gx = cv2.Scharr(gray, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(gray, cv2.CV_32F, 0, 1)
    grad = cv2.magnitude(gx, gy)

    lap3 = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    lap5 = cv2.Laplacian(gray, cv2.CV_32F, ksize=5)
    hf = 0.5 * np.abs(lap3) + 0.5 * np.abs(lap5)

    mu = cv2.GaussianBlur(gray, (0, 0), 1.0)
    mu2 = cv2.GaussianBlur(gray * gray, (0, 0), 1.0)
    local_std = np.sqrt(np.clip(mu2 - mu * mu, 0.0, 1.0))

    # 分位归一化（逐像素图）
    def rnorm_map(x, q=95):
        p = float(np.percentile(x, q)) + 1e-6
        return np.clip(x / p, 0.0, 1.0)

    grad_n = rnorm_map(grad)
    hf_n = rnorm_map(hf)
    std_n = rnorm_map(local_std)

    rough = 0.45 * grad_n + 0.4 * hf_n + 0.15 * std_n

    score = 0.6 * rough + 0.25 * (1.0 - s) + 0.15 * (1.0 - v)
    score = cv2.GaussianBlur(score, (0, 0), 1.0)
    score = np.clip(score, 0.0, 1.0)
    return score


# 色彩变换  肤色得分对应指定颜色
def colormap_from_score_map(score_map):
    """
    将分数图映射到 BGR 颜色（区间离散版）：
    - 0.0–0.3   -> [250, 152, 152]
    - 0.3–0.5   -> [203, 192, 255]
    - 0.5–0.65  -> [226, 43, 138]
    - 0.65–0.8  -> [235, 206, 135]
    - 0.8–1.0   -> [255, 0, 0]
    """
    s = np.clip(score_map.astype(np.float32), 0.0, 1.0)

    H, W = s.shape
    b = np.full((H, W), 255, dtype=np.float32)
    g = np.full((H, W), 255, dtype=np.float32)
    r = np.full((H, W), 255, dtype=np.float32)

    # 按区间设置颜色（BGR）
    m1 = (s >= 0.0) & (s < 0.4)
    b[m1], g[m1], r[m1] = 255, 255, 255

    m2 = (s >= 0.4) & (s < 0.55)
    b[m2], g[m2], r[m2] = 255, 255, 255

    m3 = (s >= 0.55) & (s < 0.65)
    b[m3], g[m3], r[m3] = 226, 43, 138

    m4 = (s >= 0.65) & (s < 0.75)
    b[m4], g[m4], r[m4] = 255, 255, 255

    m5 = (s >= 0.75)
    b[m5], g[m5], r[m5] = 255, 255, 255

    color = np.stack([b, g, r], axis=2).astype(np.uint8)
    return color


def build_skin_mask(frame, face_rect=None):
    """
    基于肤色的像素级掩膜（不使用固定形状、不分区域）。
    - 在 HSV 与 YCrCb 空间阈值取交集
    - 形态学清理 + 连通域筛选（若提供 face_rect，则保留与其重叠最大的皮肤区域）
    返回: mask(H,W) uint8, 值为 0/255
    """
    img = frame
    H, W = img.shape[:2]

    # HSV 肤色阈值（可按环境调节）
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_hsv1 = np.array([0, 30, 50], dtype=np.uint8)
    upper_hsv1 = np.array([25, 180, 255], dtype=np.uint8)
    mask_hsv1 = cv2.inRange(hsv, lower_hsv1, upper_hsv1)
    # 红色端
    lower_hsv2 = np.array([160, 30, 50], dtype=np.uint8)
    upper_hsv2 = np.array([179, 180, 255], dtype=np.uint8)
    mask_hsv2 = cv2.inRange(hsv, lower_hsv2, upper_hsv2)
    mask_hsv = cv2.bitwise_or(mask_hsv1, mask_hsv2)

    # YCrCb 肤色范围
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
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

    # 若提供人脸框，只保留与其重叠最大的连通域，避免背景皮肤干扰
    if face_rect is not None:
        x, y, w, h = face_rect
        x = max(0, x);
        y = max(0, y)
        w = min(w, W - x);
        h = min(h, H - y)
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
                    union = (face_box[2] - face_box[0]) * (face_box[3] - face_box[1]) + bw * bh - inter
                    iou = inter / union if union > 0 else 0.0
                    if iou > best_iou:
                        best_iou = iou
                        best_label = lab
                mask = np.where(labels == best_label, 255, 0).astype(np.uint8)

    # 挖空五官（若可用）
    # feature_polys = _feature_polygons_mediapipe(frame)
    # if feature_polys:
    #     for pts in feature_polys.values():
    #         try:
    #             cv2.fillConvexPoly(mask, pts, 0)
    #         except Exception:
    #             pass

    return mask


# 面孔特征滤波增强  边缘特征增强
def pretreatment(img, method='laplacian'):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 预处理：轻度高斯模糊抑制噪声，保留小特征
    blurred = cv2.GaussianBlur(gray, (3, 3), 0.8)  # 小核+低标准差，保留细节
    if method == 'multi-scale':
        # 方法1：多尺度Canny边缘检测
        # 小阈值突出弱边缘，高阈值保证边缘连续性
        edges1 = cv2.Canny(blurred, 10, 30)  # 低阈值组合，捕捉小特征
        edges2 = cv2.Canny(blurred, 30, 90)  # 中阈值组合
        # 融合多尺度结果，保留小特征同时增强边缘连续性
        edges = cv2.bitwise_or(edges1, edges2)


    elif method == 'laplacian':
        # 方法2：拉普拉斯增强+阈值处理
        # 使用小核拉普拉斯算子增强细微变化
        laplacian = cv2.Laplacian(blurred, cv2.CV_64F, ksize=3)
        laplacian_abs = cv2.convertScaleAbs(laplacian)
        # 自适应阈值提取小特征边缘
        edges = cv2.adaptiveThreshold(
            laplacian_abs, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 5, 2
        )

    elif method == 'scharr':
        # 方法3：Scharr算子（对细微边缘更敏感）
        # Scharr比Sobel有更高的边缘响应，适合小特征
        scharr_x = cv2.Scharr(blurred, cv2.CV_64F, 1, 0)
        scharr_y = cv2.Scharr(blurred, cv2.CV_64F, 0, 1)
        scharr_x_abs = cv2.convertScaleAbs(scharr_x)
        scharr_y_abs = cv2.convertScaleAbs(scharr_y)
        # 合并x和y方向边缘
        edges = cv2.addWeighted(scharr_x_abs, 0.5, scharr_y_abs, 0.5, 0)

    else:
        raise ValueError("方法不存在，请选择 'multi-scale', 'laplacian' 或 'scharr'")

    # 后处理：形态学操作增强小边缘的连通性
    kernel = np.ones((2, 2), np.uint8)  # 小结构元，避免破坏小特征
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)

    return img, gray, edges


# 近似圆
def is_approximately_circular(contour, min_circularity=0.68):
    """
    判断轮廓是否近似圆形
    :param contour: 轮廓数据
    :param min_circularity: 最小圆形度阈值（0-1，越大越接近正圆）
    :return: 是否为近似圆形
    """
    area = cv2.contourArea(contour)
    if area < 10:  # 过滤过小的轮廓（噪点）
        return False, 0.0

    perimeter = cv2.arcLength(contour, True)
    if perimeter == 0:  # 避免除以零
        return False, 0.0

    # 圆形度计算公式：4π×面积 / 周长²（完美圆形为1）
    circularity = (4 * np.pi * area) / (perimeter ** 2)

    # print(circularity)

    return circularity >= min_circularity, circularity


def extract_mole_roi(image, threshold=0.7):
    """提取图像中的痣区域（ROI）"""
    # 转为HSV色彩空间，便于肤色与痣的分离
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 肤色范围（可根据实际情况调整）
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)

    # 痣通常比周围皮肤深，用阈值分割提取深色区域
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mole_mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

    # 结合皮肤掩码，排除非皮肤区域的干扰
    mole_mask = cv2.bitwise_and(mole_mask, skin_mask)

    # 形态学操作去除噪点
    kernel = np.ones((3, 3), np.uint8)
    mole_mask = cv2.morphologyEx(mole_mask, cv2.MORPH_CLOSE, kernel)

    # 查找轮廓并提取最大的痣区域（假设单痣场景）
    contours, _ = cv2.findContours(mole_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # print(len(contours))
    if not contours:
        return None, None
    if len(contours) == 1:
        largest_contour = max(contours, key=cv2.contourArea)
        ret, _ = is_approximately_circular(largest_contour)
        return ret

    else:
        return None

def nms_without_scores(boxes, iou_threshold):
    """
    无置信度时的非极大值抑制（基于边界框面积排序）

    参数:
        boxes: 边界框坐标，形状为 [N, 4]，格式为 [x1, y1, x2, y2]
        iou_threshold: IOU阈值，超过此阈值的重叠框将被抑制

    返回:
        keep: 保留的边界框索引列表
    """
    if len(boxes) == 0:
        return []

    # 转换为numpy数组并确保为浮点型
    boxes = np.array(boxes, dtype=np.float32)
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]

    # 计算每个边界框的面积（作为排序依据）
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    # 按面积降序排序（面积越大越先被保留）
    order = areas.argsort()[::-1]  # [::-1] 表示降序
    keep = []
    while order.size > 0:
        # 取当前面积最大的框
        i = order[0]
        keep.append(i)

        # 计算当前框与剩余框的交叠区域
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        # 计算交叠区域的宽和高（确保非负）
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h

        # 计算IOU
        iou = inter / (areas[i] + areas[order[1:]] - inter)

        # 保留IOU小于阈值的框
        inds = np.where(iou <= iou_threshold)[0]

        # 更新排序（排除已处理和被抑制的框）
        order = order[inds + 1]  # +1是因为已排除order[0]
    return keep




def process_image(img, cascade_path="haarcascade_frontalface_default.xml"):
    """
    在纯白背景上显示人脸像素级热力图（不修改原图），并保存/显示。
    同时返回基于矩形区域的统计结果（便于数值查看）。
    """
    from api.services.model_manager import get_model_manager
    
    # 使用预加载的人脸检测器
    model_manager = get_model_manager()
    face_cascade = model_manager.get_face_cascade()
    
    if face_cascade is None:
        # 兜底：如果模型管理器中没有，则加载
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + cascade_path)
    
    # img=img[600:1800,:,:]
    if img is None:
        raise FileNotFoundError("无法读取图片，请检查路径。")

    H, W = img.shape[:2]
    gray_old = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_old, scaleFactor=1.1, minNeighbors=5, minSize=(120, 120))  # 人脸矩形框粗定位

    # 纯白画布（最终输出）
    canvas = np.full((H, W, 3), 255, dtype=np.uint8)
    # 小面积特征独立展示画布（白底）
    small_canvas = np.full((H, W, 3), 255, dtype=np.uint8)

    fx,fy,fh,fw=0,0,0,0
    summary = {}
    for (x, y, w, h) in faces:
        # 计算该人脸 ROI 的像素级分数与颜色
        roi = img[y:y + h, x:x + w]
        fx,fy,fh,fw=x,y,w,h
        if w<700:
            small_min_area_px = w * 0.155
            small_max_area_px = w * 0.45
        else:
            small_min_area_px = w * 0.1633
            small_max_area_px = w * 0.68

        if roi.size == 0:
            continue
        score = score_map_from_roi(roi)
        heat = colormap_from_score_map(score)

        #仅在肤色掩膜范围内绘制到白底画布
        full_mask = build_skin_mask(img, face_rect=(x, y, w, h))
        mask = full_mask[y:y + h, x:x + w]
        patch = canvas[y:y + h, x:x + w]  # 直接赋值，在canvas上操作
        patch[mask > 0] = heat[mask > 0]
        # full_mask[y:y + h, x:x + w][mask > 0] = heat[mask > 0]

    # canvas=canvas[600:1800,:,:]
    # 小面积特征提取与绘制（基于最终掩膜）
    img1, gray, edge = pretreatment(canvas)
    contour_rectangle = []
    contour_img_list = []
    contour_list_save = []
    contour_array = []
    contours, _ = cv2.findContours(edge, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    contour_num = 0
    for cnt in contours:  # 得到轮廓后进行循环，找到每个点的坐标位置单独扣除掩膜，再将每块儿进行识别检测打分
        a = cv2.contourArea(cnt)

        (cx, cy), (l, w), theta = cv2.minAreaRect(cnt)  # 最小外接矩形，倾斜

        x, y, w, h = cv2.boundingRect(cnt)  # 外接矩形，正
        NEVUS_img = img[y:y + h, x:x + w]
        # cv2.imwrite("nevus_zc_"+str(contour_num)+".jpg", NEVUS_img)
        if a >= float(small_min_area_px) and a <= float(small_max_area_px):
            contour_rectangle.append([x, y, w, h])
            contour_list_save.append(cnt)
            # ROI轮廓转为整图绝对坐标后画到白底小图
            cv2.drawContours(canvas, [cnt], -1, (0, 0, 255), 1)

        contour_num += 1

    for cx, cy, cw, ch in contour_rectangle:
        contour_img_list.append(img[cy:cy + ch, cx:cx + cw])

    index_z = []
    index_B = []
    kj = 0
    for j in contour_img_list:
        if j is not None and j.shape[0] > 0 and j.shape[1] > 0:
            ret = extract_mole_roi(j)
            if ret is True:
                index_z.append(kj)
        else:
            print(f"跳过空图像: contour_img{kj}")
        kj = kj + 1
    ####新增nms
    for ind in index_z:
        contour_array.append([contour_rectangle[ind][0], contour_rectangle[ind][1],
                              contour_rectangle[ind][0] + contour_rectangle[ind][2],
                              contour_rectangle[ind][1] + contour_rectangle[ind][3]])
    keep_rectangle = nms_without_scores(contour_array, 0.5)

    img_list_save = []
    zhi_count = 0
    se_ban = 0
    all_num_count = 0
    img1111 = img.copy()

    t0 = time.time()
    for k in keep_rectangle:

        block_img = img[contour_array[k][1]:contour_array[k][3], contour_array[k][0]: contour_array[k][2], :]
        gray_now_block = gray_old[contour_array[k][1]:contour_array[k][3], contour_array[k][0]: contour_array[k][2]]
        mole_mask = cv2.adaptiveThreshold(gray_now_block, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
        all_num_count += 1
        # 结合皮肤掩码，排除非皮肤区域的干扰
        pre_img = block_img.copy()
        pre_img = cv2.cvtColor(pre_img, cv2.COLOR_BGR2RGB)

        pre_img = Image.fromarray(pre_img)
        label = predict_main(pre_img)
        # "nevus", "stain"
        if label == "nevus":
            zhi_count += 1

        elif label == "stain":
            se_ban += 1
        color = (0, 0, 255) if label == "nevus" else (255, 0, 0) if label == "stain" else (0, 255, 0)

        # 绘制结果（痣：红色，色斑：蓝色）
        cv2.rectangle(img1111,   (contour_array[k][0], contour_array[k][1])
                      , (contour_array[k][2], contour_array[k][3]), color, 1)
    final_img=img1111[fy:fy + fh, fx:fx + fw,:]


    return final_img,zhi_count,se_ban








