"""
人脸测量绘制服务
提供人脸特征测量和可视化功能
"""

import os
import json
import math
from typing import Dict, List, Tuple, Optional
import uuid as uuid_lib

import cv2
import numpy as np



# 静默第三方库冗余日志/警告（需在导入 mediapipe 前设置）
import warnings as _warnings
# 抑制 TensorFlow/TFLite 的 INFO/WARNING
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")  # 0=ALL, 1=INFO, 2=WARNING, 3=ERROR
# 抑制 absl/glog 的 WARNING
os.environ.setdefault("GLOG_minloglevel", "2")      # 0=INFO, 1=WARNING, 2=ERROR, 3=FATAL
try:
    from absl import logging as _absl_logging
    _absl_logging.set_verbosity(_absl_logging.ERROR)
    try:
        _absl_logging.set_stderrthreshold("error")
    except Exception:
        pass
except Exception:
    pass

# 静默 protobuf 的特定弃用警告
try:
    import re as _re
    _warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        module=r"google\.protobuf\.symbol_database"
    )
except Exception:
    pass

# 彻底安静：在关键阶段重定向 stdout/stderr，屏蔽底层 C++ 日志
from contextlib import contextmanager, redirect_stderr, redirect_stdout
@contextmanager
def quiet_logs():
    import sys
    try:
        with open(os.devnull, "w") as _null, redirect_stderr(_null), redirect_stdout(_null):
            yield
    except Exception:
        # 兜底：即便重定向失败也不影响主流程
        yield

# 颜色与样式
WHITE = (255, 255, 255)
BLUE = (255, 0, 0)  # BGR 蓝色
RED = (0, 0, 255)
GREEN = (0, 255, 0)
YELLOW = (0, 255, 255)
CYAN = (255, 255, 0)
PURPLE = (128, 0, 128)  # BGR 紫色

# 单位与中文绘制辅助
MM_PER_PX = float(os.environ.get("MM_PER_PX", "0.264583"))  # 默认约为 96DPI 的像素到毫米换算

def px_to_mm(px: float) -> float:
    return float(px) * MM_PER_PX

def fmt_mm(px: float) -> str:
    return f"{px_to_mm(px):.1f}mm"

def draw_text_cn(img, text: str, pos: Tuple[int,int], color=(255,255,255), bg_color=None, font_px=20):
    """
    在图像上绘制中文文本。优先使用 Windows 常见中文字体；失败时回退至 cv2.putText。
    - img: OpenCV BGR 图像
    - text: 文本（支持中文）
    - pos: 左下角坐标 (x, y)
    - color: 文本颜色（BGR）
    - bg_color: 背景矩形颜色（BGR），为 None 则不绘制背景
    - font_px: 字体像素大小
    """
    try:
        from PIL import Image, ImageDraw, ImageFont
        font_paths = [
            "C:/Windows/Fonts/msyh.ttc",   # 微软雅黑
            "C:/Windows/Fonts/msyh.ttf",
            "C:/Windows/Fonts/simhei.ttf", # 黑体
            "C:/Windows/Fonts/simsun.ttc", # 宋体
        ]
        font = None
        for fp in font_paths:
            if os.path.exists(fp):
                try:
                    font = ImageFont.truetype(fp, font_px)
                    break
                except Exception:
                    continue
        if font is None:
            font = ImageFont.load_default()

        # OpenCV BGR -> PIL RGB
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        draw = ImageDraw.Draw(pil)
        x, y = pos

        if bg_color is not None:
            # 计算文本尺寸并绘制背景矩形
            try:
                bbox = draw.textbbox((x, y), text, font=font)
                tw = bbox[2] - bbox[0]
                th = bbox[3] - bbox[1]
            except Exception:
                tw, th = draw.textsize(text, font=font)
            pad = max(2, font_px // 6)
            # 使背景位于文本上方（cv2.putText 的习惯是 pos 为左下角）
            draw.rectangle([x - pad, y - th - pad, x + tw + pad, y + pad], fill=tuple(int(c) for c in bg_color))

        # 绘制文本（pos 作为左下角，故将 y 减去字体高度）
        draw.text((x, y - font_px), text, fill=tuple(int(c) for c in color), font=font)

        # PIL RGB -> OpenCV BGR（写回 img）
        img[:] = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    except Exception:
        # 回退：可能无法显示中文，但不致使程序崩溃
        cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, max(0.5, font_px / 24.0), color, 1, cv2.LINE_AA)

# 可视化参数（会做自适应缩放）
BASE_THICKNESS = 2
BASE_POINT_RADIUS = 3
FONT = cv2.FONT_HERSHEY_SIMPLEX

# 指标关键点（根据您的要求更新）
KP = {
    # 1) 脸部整体几何
    "temporal_width": (127, 356),           # 颞部宽度
    "zygion_width": (234, 454),             # 颧骨宽度
    "gonion_width": (58, 288),              # 下颌角宽度
    "face_length": (10, 151),               # 脸部长度（修改：使用10和151）
    "jaw_angle_triplet": (361, 288, 152),   # 下颌角度数
    
    # 2) 三庭
    "upper_third": (10, 9),                 # 上庭长度
    "mid_third": (9, 2),                    # 中庭长度
    "lower_third": (2, 152),                # 下庭长度
    
    # 3) 眼部
    "right_outer_eye_to_zygion": (263, 454),  # 右外眼角颧弓留白
    "inner_canthal": (133, 362),            # 内眼角间距
    "left_outer_eye_to_zygion": (33, 234),  # 左外眼角颧弓留白
    "left_eye_width": (33, 133),            # 左眼宽度
    "right_eye_width": (362, 263),          # 右眼宽度
    
    # 4) 下巴/黄金三角
    "golden_triangle_triplet": (468, 2, 473),  # 黄金三角度数
    "chin_length": (17, 152),               # 下巴长度
    "chin_width": (136, 365),               # 下巴宽度
    "chin_angle_triplet": (136, 152, 365),  # 下巴角度数
    
    # 5) 鼻翼与口唇
    "alar_width": (129, 358),               # 鼻翼宽度
    "lip_height_points": (37, 267, 17),     # 嘴唇高度
    "lip_width": (61, 291),                 # 嘴唇宽度
    "lip_thickness_upper": (0, 13),         # 上唇厚度
    "lip_thickness_lower": (14, 17),        # 下唇厚度
    "mouth_corner_angle_segments": (13, 291, 291, 358),  # 嘴角弯曲度
}

def calc_adaptive(img_shape: Tuple[int, int, int], base_thickness=BASE_THICKNESS, base_point_radius=BASE_POINT_RADIUS):
    h, w = img_shape[:2]
    diag = math.hypot(w, h)
    base_diag = math.hypot(1000, 1000)
    s = max(0.5, min(2.0, diag / base_diag))
    return max(1, int(base_thickness * s)), max(2, int(base_point_radius * s)), s

def to_int_pt(pt: Tuple[float, float]) -> Tuple[int, int]:
    return (int(round(pt[0])), int(round(pt[1])))

# 脸部虚线裁剪：全局矩形（由点索引 234, 10, 454, 152 定义）
CURRENT_FACE_RECT = None  # (left, top, right, bottom)

def set_face_rect_from_pts(pts: Dict[int, Tuple[int,int]]):
    global CURRENT_FACE_RECT
    try:
        left = int(pts[234][0])
        top = int(pts[10][1] - abs(pts[151][1] - pts[10][1]) * 2 / 3)
        right = int(pts[454][0])
        bottom = int(pts[152][1])
        # 规范化：确保 left<right, top<bottom
        l, r = sorted([left, right])
        t, b = sorted([top, bottom])
        CURRENT_FACE_RECT = (l, t, r, b)
    except Exception:
        CURRENT_FACE_RECT = None

def _clip_segment_to_rect(p1: Tuple[float,float], p2: Tuple[float,float], rect: Tuple[int,int,int,int]):
    """
    Liang-Barsky 算法裁剪线段到矩形，返回裁剪后的 (q1, q2)，无交则返回 None
    rect: (left, top, right, bottom)
    """
    x0, y0 = float(p1[0]), float(p1[1])
    x1, y1 = float(p2[0]), float(p2[1])
    xmin, ymin, xmax, ymax = float(rect[0]), float(rect[1]), float(rect[2]), float(rect[3])

    dx, dy = x1 - x0, y1 - y0
    p = [-dx, dx, -dy, dy]
    q = [x0 - xmin, xmax - x0, y0 - ymin, ymax - y0]

    u1, u2 = 0.0, 1.0
    for pi, qi in zip(p, q):
        if pi == 0:
            if qi < 0:
                return None
        else:
            t = qi / pi
            if pi < 0:
                if t > u2: return None
                if t > u1: u1 = t
            else:
                if t < u1: return None
                if t < u2: u2 = t

    qx0, qy0 = x0 + u1 * dx, y0 + u1 * dy
    qx1, qy1 = x0 + u2 * dx, y0 + u2 * dy
    return (qx0, qy0), (qx1, qy1)

def draw_dashed_line(img, p1, p2, color, thickness, dash_len=10, gap_len=6):
    """
    绘制虚线（可分割成均匀的小段），并在存在 CURRENT_FACE_RECT 时裁剪每段到脸部矩形内。
    """
    p1 = np.array(p1, dtype=float)
    p2 = np.array(p2, dtype=float)
    vec = p2 - p1
    L = np.linalg.norm(vec)
    if L == 0:
        return
    dirv = vec / L
    pos = 0.0
    while pos < L:
        start = p1 + dirv * pos
        end = p1 + dirv * min(L, pos + dash_len)
        seg = (tuple(start), tuple(end))
        if CURRENT_FACE_RECT is not None:
            clipped = _clip_segment_to_rect(seg[0], seg[1], CURRENT_FACE_RECT)
            if clipped is not None:
                s, e = clipped
                cv2.line(img, to_int_pt(s), to_int_pt(e), color, thickness, lineType=cv2.LINE_AA)
        else:
            cv2.line(img, to_int_pt(seg[0]), to_int_pt(seg[1]), color, thickness, lineType=cv2.LINE_AA)
        pos += dash_len + gap_len

def draw_horizontal_measurement(img, y, x1, x2, text, color=BLUE, thickness=2, font_scale=0.6, aux_guides=True, guide_span=None):
    """水平测量：根据脸部宽度来确定虚线分割的美观程度"""
    x1, x2 = int(round(x1)), int(round(x2))
    y = int(round(y))
    if x1 > x2:
        x1, x2 = x2, x1

    # 确保虚线的跨度与脸部宽度一致，分成若干个小段（优先限定在脸部矩形内）
    if CURRENT_FACE_RECT is not None:
        y1g, y2g = int(CURRENT_FACE_RECT[1]), int(CURRENT_FACE_RECT[3])
    elif guide_span is None:
        y1g, y2g = 0, img.shape[0] - 1
    else:
        y1g = max(0, y - int(guide_span))
        y2g = min(img.shape[0] - 1, y + int(guide_span))

    # 分段绘制虚线（仅当需要辅助线时）
    if aux_guides:
        draw_dashed_line(img, (x1, y1g), (x1, y2g), WHITE, max(1, thickness - 1))
        draw_dashed_line(img, (x2, y1g), (x2, y2g), WHITE, max(1, thickness - 1))

    # 紫色水平双向箭头 - 记录实际绘制的端点
    actual_start = (x1, y)
    actual_end = (x2, y)

    cv2.arrowedLine(img, (x1, y), (x2, y), PURPLE, thickness, cv2.LINE_AA, 0, 0.02)
    cv2.arrowedLine(img, (x2, y), (x1, y), PURPLE, thickness, cv2.LINE_AA, 0, 0.02)

    # 文本 - 计算文本宽度以实现中心对齐
    mid_x = (x1 + x2) // 2
    font_px = max(14, int(18 * font_scale))
    
    # 估算文本宽度（中文字符按font_px计算，英文字符按font_px*0.6计算）
    text_width = 0
    for char in text:
        if ord(char) > 127:  # 中文字符
            text_width += font_px
        else:  # 英文字符和数字
            text_width += int(font_px * 0.6)
    
    # 文本起始位置向左偏移文本宽度的一半，实现中心对齐
    text_x = mid_x - text_width // 2
    draw_text_cn(img, text, (text_x, y - 6), color=WHITE, bg_color=(0, 0, 0), font_px=font_px)

     # 返回实际绘制的线段端点坐标
    return {
        'start': [actual_start[0], actual_start[1]],
        'end': [actual_end[0], actual_end[1]]
    }

def draw_vertical_measurement(img, x, y1, y2, text, color=BLUE, thickness=2, font_scale=0.6, aux_guides=True, guide_span=None):
    """垂直测量：根据脸部高度来确定虚线分割的美观程度"""
    x = int(round(x))
    y1, y2 = int(round(y1)), int(round(y2))
    if y1 > y2:
        y1, y2 = y2, y1

    # 确保虚线的跨度与脸部高度一致，分成若干个小段（优先限定在脸部矩形内）
    if CURRENT_FACE_RECT is not None:
        x1g, x2g = int(CURRENT_FACE_RECT[0]), int(CURRENT_FACE_RECT[2])
    elif guide_span is None:
        x1g, x2g = 0, img.shape[1] - 1
    else:
        x1g = max(0, x - int(guide_span))
        x2g = min(img.shape[1] - 1, x + int(guide_span))

    # 分段绘制虚线（仅当需要辅助线时）
    if aux_guides:
        draw_dashed_line(img, (x1g, y1), (x2g, y1), WHITE, max(1, thickness - 1))
        draw_dashed_line(img, (x1g, y2), (x2g, y2), WHITE, max(1, thickness - 1))

    # 紫色竖直双向箭头 - 记录实际绘制的端点
    actual_start = (x, y1)
    actual_end = (x, y2)

    cv2.arrowedLine(img, (x, y1), (x, y2), PURPLE, thickness, cv2.LINE_AA, 0, 0.02)
    cv2.arrowedLine(img, (x, y2), (x, y1), PURPLE, thickness, cv2.LINE_AA, 0, 0.02)

    # 文本
    mid_y = (y1 + y2) // 2
    font_px = max(14, int(18 * font_scale))
    draw_text_cn(img, text, (x + 6, mid_y - 12), color=WHITE, bg_color=(0, 0, 0), font_px=font_px)

    # 返回实际绘制的线段端点坐标
    return {
        'start': [actual_start[0], actual_start[1]],
        'end': [actual_end[0], actual_end[1]]
    }

def draw_angle_dashed(img, A, O, B, color=WHITE, thickness=2, radius=40, font_scale=0.6, label="angle"):
    """绘制角度（使用虚线圆弧）"""
    A = np.array(A, dtype=float)
    O = np.array(O, dtype=float)
    B = np.array(B, dtype=float)
    v1 = A - O
    v2 = B - O
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return None
    cosang = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
    ang_deg = math.degrees(math.acos(cosang))

    cv2.line(img, to_int_pt(tuple(O)), to_int_pt(tuple(A)), color, thickness, cv2.LINE_AA)
    cv2.line(img, to_int_pt(tuple(O)), to_int_pt(tuple(B)), color, thickness, cv2.LINE_AA)

    a1 = math.degrees(math.atan2(-(A[1]-O[1]), A[0]-O[0]))
    a2 = math.degrees(math.atan2(-(B[1]-O[1]), B[0]-O[0]))
    a1 = (a1 + 360) % 360
    a2 = (a2 + 360) % 360
    start_angle = a1
    end_angle = a2
    delta = (end_angle - start_angle + 360) % 360
    if delta > 180:
        start_angle, end_angle = end_angle, start_angle

    # 使用虚线绘制圆弧
    center = to_int_pt(tuple(O))
    steps = int(abs(end_angle - start_angle) / 5)  # 每5度一个段
    if steps > 0:
        for i in range(steps):
            angle1 = start_angle + i * (end_angle - start_angle) / steps
            angle2 = start_angle + (i + 0.5) * (end_angle - start_angle) / steps
            if i % 2 == 0:  # 绘制虚线
                pt1 = (int(center[0] + radius * math.cos(math.radians(angle1))),
                       int(center[1] - radius * math.sin(math.radians(angle1))))
                pt2 = (int(center[0] + radius * math.cos(math.radians(angle2))),
                       int(center[1] - radius * math.sin(math.radians(angle2))))
                cv2.line(img, pt1, pt2, color, thickness, cv2.LINE_AA)

    font_px = max(14, int(18 * font_scale))
    draw_text_cn(img, f"{label}:{ang_deg:.2f}°", (center[0]+8, center[1]-8), color=WHITE, bg_color=(0,0,0), font_px=font_px)
    return ang_deg

def line_point_distance(p1, p2, vertical=False):
    if vertical:
        return abs(p2[1] - p1[1])
    return abs(p2[0] - p1[0])

def dist(p1, p2):
    return float(np.linalg.norm(np.array(p1, dtype=float) - np.array(p2, dtype=float)))

def normalized_to_pixel(lm, shape):
    h, w = shape[:2]
    x = min(max(int(round(lm.x * w)), 0), w - 1)
    y = min(max(int(round(lm.y * h)), 0), h - 1)
    return (x, y)

def put_point_with_id(img, pt, idx, color, r):
    pass
    # cv2.circle(img, to_int_pt(pt), r, color, -1, cv2.LINE_AA)
    # cv2.putText(img, str(idx), (to_int_pt(pt)[0]+4, to_int_pt(pt)[1]-4), FONT, 0.45, color, 1, cv2.LINE_AA)

def get_angle_line_endpoints(a, o, b, radius=50):
    """
    计算角度测量中两条射线的端点坐标
    """
    # 计算向量
    vec_oa = (a[0] - o[0], a[1] - o[1])
    vec_ob = (b[0] - o[0], b[1] - o[1])
    
    # 归一化向量
    len_oa = max(1, math.sqrt(vec_oa[0]**2 + vec_oa[1]**2))
    len_ob = max(1, math.sqrt(vec_ob[0]**2 + vec_ob[1]**2))
    
    vec_oa_norm = (vec_oa[0]/len_oa, vec_oa[1]/len_oa)
    vec_ob_norm = (vec_ob[0]/len_ob, vec_ob[1]/len_ob)
    
    # 计算射线端点
    line1_end = (int(o[0] + vec_oa_norm[0] * radius), int(o[1] + vec_oa_norm[1] * radius))
    line2_end = (int(o[0] + vec_ob_norm[0] * radius), int(o[1] + vec_ob_norm[1] * radius))
    
    return {
        'line1_start': [int(o[0]), int(o[1])],
        'line1_end': [line1_end[0], line1_end[1]],
        'line2_start': [int(o[0]), int(o[1])],
        'line2_end': [line2_end[0], line2_end[1]]
    }

def compute_forehead_point(pt10: Tuple[int,int], pt151: Tuple[int,int], length: float) -> Tuple[int,int]:
    """
    估计发际线顶点：从 151->10 的方向，在 10 点继续延长 length 距离。
    """
    v = (pt10[0] - pt151[0], pt10[1] - pt151[1])
    norm = math.hypot(v[0], v[1])
    if norm == 0:
        return pt10
    ux, uy = v[0] / norm, v[1] / norm
    hx = int(round(pt10[0] + ux * length))
    hy = int(round(pt10[1] + uy * length))
    return (hx, hy)

def compute_and_draw_group1(img, pts, thick, pr, scale) -> Dict:
    """脸部整体几何"""
    res = {}
    
    # 颞部宽度 - 水平测量
    pL, pR = pts[KP["temporal_width"][0]], pts[KP["temporal_width"][1]]
    eye_top = min(pts[33][1], pts[133][1], pts[362][1], pts[263][1])
    left_eyebrow_bottom = max(pts[70][1], pts[63][1], pts[105][1], pts[66][1], pts[107][1])
    right_eyebrow_bottom = max(pts[300][1], pts[293][1], pts[334][1], pts[296][1], pts[336][1])
    eyebrow_bottom = max(left_eyebrow_bottom, right_eyebrow_bottom)
    
    if eyebrow_bottom < eye_top:
        mid_y = (eyebrow_bottom + eye_top) // 2
    else:
        mid_y = max(0, eye_top - int(15 * scale))
    
    y = mid_y
    tw = dist(pL, pR)
    hair_top = compute_forehead_point(pts[10], pts[151], length=dist(pts[10], pts[151]))
    y_top, y_bottom = hair_top[1], pts[152][1]
    face_len = abs(y_bottom - y_top)
    guide_span_v = int(face_len // 2 + int(12 * scale))
    temporal_line_points = draw_horizontal_measurement(img, y, pL[0], pR[0], f"颞宽:{fmt_mm(tw)}", PURPLE, thick, aux_guides=True, guide_span=guide_span_v)
    
    # 颧骨宽度
    zL, zR = pts[KP["zygion_width"][0]], pts[KP["zygion_width"][1]]
    y = (zL[1] + zR[1]) // 2
    zw = dist(zL, zR)
    zygion_line_points = draw_horizontal_measurement(img, y, zL[0], zR[0], f"颧宽:{fmt_mm(zw)}", PURPLE, thick, aux_guides=True, guide_span=guide_span_v)
    
    # 下颌角宽度
    gL, gR = pts[KP["gonion_width"][0]], pts[KP["gonion_width"][1]]
    y = (gL[1] + gR[1]) // 2
    gw = dist(gL, gR)
    gonion_line_points = draw_horizontal_measurement(img, y, gL[0], gR[0], f"颌宽:{fmt_mm(gw)}", PURPLE, thick, aux_guides=True, guide_span=guide_span_v)

    # 脸部长度
    pt10, pt151, pt152 = pts[10], pts[151], pts[152]
    hair_top = compute_forehead_point(pt10, pt151, length=dist(pt10, pt151) * 2/3)
    y_top, y_bottom = hair_top[1], pt152[1]
    left_x = max(5, min(p[0] for p in [pL, pR, zL, zR, gL, gR, pt10, pt152]) - int(30 * scale))
    fl = abs(y_bottom - y_top)
    face_width = abs(zR[0] - zL[0])
    guide_span_h = int(face_width // 2 + int(12 * scale))
    face_length_line_points = draw_vertical_measurement(img, left_x, y_top, y_bottom, f"脸长:{fmt_mm(fl)}", PURPLE, thick, aux_guides=True, guide_span=guide_span_h)

    # 下颌角度
    a, o, b = KP["jaw_angle_triplet"]
    ang = draw_angle_dashed(img, pts[a], pts[o], pts[b], WHITE, thick, radius=int(50*scale), label="下颌角")
    jaw_angle_line_points = get_angle_line_endpoints(pts[a], pts[o], pts[b], radius=int(50*scale))
    
    res.update({
        # "temporal_width_detailed": ["颞部宽度", round(px_to_mm(tw), 2),
        #                            temporal_line_points['start'], temporal_line_points['end']],
        # "zygion_width_detailed": ["颧骨宽度", round(px_to_mm(zw), 2), 
        #                          zygion_line_points['start'], zygion_line_points['end']],
        # "gonion_width_detailed": ["下颌角宽度", round(px_to_mm(gw), 2),
        #                          gonion_line_points['start'], gonion_line_points['end']],
        # "face_length_detailed": ["脸部长度", round(px_to_mm(fl), 2),
        #                         face_length_line_points['start'], face_length_line_points['end']],
        # "jaw_angle_detailed": ["下颌角度", round(ang, 2) if ang is not None else None, 
        #                       jaw_angle_line_points['line1_start'], jaw_angle_line_points['line1_end'],
        #                       jaw_angle_line_points['line2_start'], jaw_angle_line_points['line2_end']],
        
        "face_shape":"长脸型",
        "jaw_angle":{
            "name":"下颌角宽度",
            "angle":f"{round(ang, 2)}"}
    })

    # 标注关键点
    for idx in [*KP["temporal_width"], *KP["zygion_width"], *KP["gonion_width"], *KP["face_length"], *KP["jaw_angle_triplet"]]:
        if idx in (10, 151):
            continue
        put_point_with_id(img, pts[idx], idx, WHITE, pr)
    
    # draw_text_cn(img, "脸部整体几何测量", (20, 30), color=WHITE, bg_color=(50,50,50), font_px=22)
    return res

def compute_and_draw_group2(img, pts, thick, pr, scale) -> Dict:
    """三庭测量"""
    res = {}

    pt10, pt151, pt9, pt2, pt152 = pts[10], pts[151], pts[9], pts[2], pts[152]
    hair_top = compute_forehead_point(pt10, pt151, length=dist(pt10, pt151) * 2/3)
    face_left = min(pts[KP["temporal_width"][0]][0], pts[KP["gonion_width"][0]][0])
    left_x = max(5, face_left - int(20 * scale))
    mid_x = pts[9][0]
    y_top, y_bottom = hair_top[1], pt152[1]
    y1g = max(0, min(y_top, y_bottom) - int(12 * scale))
    y2g = min(img.shape[0]-1, max(y_top, y_bottom) + int(12 * scale))
    draw_dashed_line(img, (mid_x, y1g), (mid_x, y2g), WHITE, max(1, thick-1))

    # 上庭
    y1_u, y2_u = hair_top[1], pt9[1]
    up = abs(y2_u - y1_u)
    face_width = abs(pts[KP["zygion_width"][1]][0] - pts[KP["zygion_width"][0]][0])
    face_hspan = int(face_width // 2 + int(12 * scale))
    upper_third_line_points = draw_vertical_measurement(img, left_x, y1_u, y2_u, f"上庭:{fmt_mm(up)}", PURPLE, thick, aux_guides=True, guide_span=face_hspan)

    # 中庭
    y1_m, y2_m = pt9[1], pt2[1]
    mid = abs(y2_m - y1_m)
    mid_third_line_points = draw_vertical_measurement(img, left_x, y1_m, y2_m, f"中庭:{fmt_mm(mid)}", PURPLE, thick, aux_guides=True, guide_span=face_hspan)

    # 下庭
    y1_l, y2_l = pt2[1], pt152[1]
    low = abs(y2_l - y1_l)
    lower_third_line_points = draw_vertical_measurement(img, left_x, y1_l, y2_l, f"下庭:{fmt_mm(low)}", PURPLE, thick, aux_guides=True, guide_span=face_hspan)

    total = up + mid + low if (up + mid + low) > 0 else 1.0

    res.update({
        # "upper_third_detailed": ["上庭", round(px_to_mm(up), 2), 
        #                         upper_third_line_points['start'], upper_third_line_points['end']],
        # "mid_third_detailed": ["中庭", round(px_to_mm(mid), 2), 
        #                       mid_third_line_points['start'], mid_third_line_points['end']],
        # "lower_third_detailed": ["下庭", round(px_to_mm(low), 2), 
        #                        lower_third_line_points['start'], lower_third_line_points['end']],
        "top":"上庭标准",
        "middle":"中庭标准",
        "bottom":"下庭偏大",
        "proportion":
            {
            "name":"三庭比例",
            "ratio":f"{round(up/total, 2)}:{round(mid/total, 2)}:{round(low/total, 2)}"
            }
        })

    for idx in [*KP["upper_third"], *KP["mid_third"], *KP["lower_third"]]:
        put_point_with_id(img, pts[idx], idx, WHITE, pr)

    # draw_text_cn(img, "三庭测量", (20, 30), color=WHITE, bg_color=(50,50,50), font_px=22)
    return res

def compute_and_draw_group3(img, pts, thick, pr, scale) -> Dict:
    """眼部测量"""
    res = {}
    
    # 右外眼角-右颧弓留白
    p1, p2 = pts[KP["right_outer_eye_to_zygion"][0]], pts[KP["right_outer_eye_to_zygion"][1]]
    hgap_r = line_point_distance(p1, p2, vertical=False)
    y = p1[1]
    right_gap_line_points = draw_horizontal_measurement(img, y, p1[0], p2[0], f"右留白:{fmt_mm(hgap_r)}", PURPLE, thick)

    # 左外眼角-左颧弓留白
    lp1, lp2 = pts[KP["left_outer_eye_to_zygion"][0]], pts[KP["left_outer_eye_to_zygion"][1]]
    hgap_l = line_point_distance(lp1, lp2, vertical=False)
    y = lp1[1]
    left_gap_line_points = draw_horizontal_measurement(img, y, lp1[0], lp2[0], f"左留白:{fmt_mm(hgap_l)}", PURPLE, thick)

    # 内眼角间距
    cL, cR = pts[KP["inner_canthal"][0]], pts[KP["inner_canthal"][1]]
    y = (cL[1] + cR[1]) // 2
    inner_gap = dist(cL, cR)
    inner_canthal_line_points = draw_horizontal_measurement(img, y, cL[0], cR[0], f"内眼角:{fmt_mm(inner_gap)}", PURPLE, thick)

    # 眼宽
    y_alar = (pts[KP["alar_width"][0]][1] + pts[KP["alar_width"][1]][1]) // 2

    l1, l2 = pts[KP["left_eye_width"][0]], pts[KP["left_eye_width"][1]]
    lw = dist(l1, l2)
    left_eye_line_points = draw_horizontal_measurement(img, y_alar, l1[0], l2[0], f"左眼宽:{fmt_mm(lw)}", PURPLE, thick, aux_guides=False)
    
    r1, r2 = pts[KP["right_eye_width"][0]], pts[KP["right_eye_width"][1]]
    rw = dist(r1, r2)
    right_eye_line_points = draw_horizontal_measurement(img, y_alar, r1[0], r2[0], f"右眼宽:{fmt_mm(rw)}", PURPLE, thick, aux_guides=False)

    res.update({
        # "right_outer_eye_zygion_gap_detailed": ["右外眼角颧弓留白", round(px_to_mm(hgap_r), 2), 
        #                                        right_gap_line_points['start'], right_gap_line_points['end']],
        # "left_outer_eye_zygion_gap_detailed": ["左外眼角颧弓留白", round(px_to_mm(hgap_l), 2), 
        #                                       left_gap_line_points['start'], left_gap_line_points['end']],
        # "inner_canthal_distance_detailed": ["内眼角间距", round(px_to_mm(inner_gap), 2), 
        #                                    inner_canthal_line_points['start'], inner_canthal_line_points['end']],
        # "left_eye_width_detailed": ["左眼宽度", round(px_to_mm(lw), 2), 
        #                            left_eye_line_points['start'], left_eye_line_points['end']],
        # "right_eye_width_detailed": ["右眼宽度", round(px_to_mm(rw), 2), 
        #                             right_eye_line_points['start'], right_eye_line_points['end']],
        "right_outer_eye":"五眼右侧偏窄",
        "inner_canthal":"内角间距偏宽",
        "left_outer_eye":"五眼左侧偏窄",
        "five_eye_proportion":
            {
            "name":"五眼比例",
            "ratio":f" {round(px_to_mm(hgap_r), 2)/round(px_to_mm(lw), 2):.2f}:{round(px_to_mm(rw), 2)/round(px_to_mm(lw), 2):.2f}:{round(px_to_mm(inner_gap), 2)/round(px_to_mm(lw), 2):.2f}:1:{round(px_to_mm(hgap_l), 2)/round(px_to_mm(lw), 2):.2f}"
            },
        "best_proportion":{
            "name":"最佳比例",
            "ratio": "0.8:1:1.2:1:0.8"
            }
    })

    for idx in [*KP["right_outer_eye_to_zygion"], *KP["left_outer_eye_to_zygion"], *KP["inner_canthal"], *KP["left_eye_width"], *KP["right_eye_width"]]:
        put_point_with_id(img, pts[idx], idx, WHITE, pr)
    
    # draw_text_cn(img, "眼部测量", (20, 30), color=WHITE, bg_color=(50,50,50), font_px=22)
    return res

def compute_and_draw_group4(img, pts, thick, pr, scale) -> Dict:
    """下巴/黄金三角测量"""
    res = {}

    # 黄金三角
    a, o, b = KP["golden_triangle_triplet"]
    cv2.line(img, to_int_pt(pts[o]), to_int_pt(pts[a]), WHITE, max(1, thick-0), cv2.LINE_AA)
    cv2.line(img, to_int_pt(pts[o]), to_int_pt(pts[b]), WHITE, max(1, thick-0), cv2.LINE_AA)
    cv2.line(img, to_int_pt(pts[a]), to_int_pt(pts[b]), WHITE, max(1, thick-0), cv2.LINE_AA)
    ang = draw_angle_dashed(img, pts[a], pts[o], pts[b], WHITE, thick, radius=int(50*scale), label="黄金三角")

    # 下巴白色虚线矩形
    x1, x2 = sorted([pts[136][0], pts[365][0]])
    y1, y2 = sorted([pts[17][1], pts[152][1]])
    y1 = max(0, y1)  
    y2 = min(img.shape[0] - 1, y2)  

    chin_rect = (x1, y1, x2, y2)
    
    def draw_dashed_line_within_chin(p1, p2, color, thickness, dash_len=10, gap_len=6):
        clipped = _clip_segment_to_rect(p1, p2, chin_rect)
        if clipped is not None:
            draw_dashed_line(img, clipped[0], clipped[1], color, thickness, dash_len, gap_len)
    
    draw_dashed_line_within_chin((x1, y1), (x2, y1), WHITE, thick)
    draw_dashed_line_within_chin((x1, y2), (x2, y2), WHITE, thick)
    draw_dashed_line_within_chin((x1, y1), (x1, y2), WHITE, thick)
    draw_dashed_line_within_chin((x2, y1), (x2, y2), WHITE, thick)

    cl = abs(y2 - y1)
    cw = abs(x2 - x1)

    y_below = min(img.shape[0]-5, y2 + int(15 * scale))
    chin_width_line_points = draw_horizontal_measurement(img, y_below+5, x1, x2, f"下巴宽:{fmt_mm(cw)}", PURPLE, thick, aux_guides=False, guide_span=int(25*scale))

    x_left = max(5, x1 - int(15 * scale))
    chin_length_line_points = draw_vertical_measurement(img, x_left, y1, y2, f"下巴长:{fmt_mm(cl)}", PURPLE, thick, aux_guides=False, guide_span=int(25*scale))

    # 下巴角度
    a2, o2, b2 = KP["chin_angle_triplet"]
    ang2 = draw_angle_dashed(img, pts[a2], pts[o2], pts[b2], WHITE, thick, radius=int(50*scale), label="下巴角")

    golden_triangle_line_points = get_angle_line_endpoints(pts[a], pts[o], pts[b], radius=int(50*scale))
    chin_angle_line_points = get_angle_line_endpoints(pts[a2], pts[o2], pts[b2], radius=int(50*scale))

    res.update({
        # "golden_triangle_angle_detailed": ["黄金三角角度", round(ang, 2) if ang is not None else None,
        #                                    golden_triangle_line_points['line1_start'], golden_triangle_line_points['line1_end'],
        #                                    golden_triangle_line_points['line2_start'], golden_triangle_line_points['line2_end']],
        # "chin_length_detailed": ["下巴长度", round(px_to_mm(cl), 2),
        #                          chin_length_line_points['start'], chin_length_line_points['end']],
        # "chin_width_detailed": ["下巴宽度", round(px_to_mm(cw), 2),
        #                         chin_width_line_points['start'], chin_width_line_points['end']],
        # "chin_angle_detailed": ["下巴角度", round(ang2, 2) if ang2 is not None else None,
        #                         chin_angle_line_points['line1_start'], chin_angle_line_points['line1_end'],
        #                         chin_angle_line_points['line2_start'], chin_angle_line_points['line2_end']]
        "golden_angle":{
            "name":"黄金三角角度",
            "angle":f"{round(ang, 2)}"
        },
        "chin_shape":"方下巴",
        "chin_angle":{
            "name":"下巴角度",
            "angle":f"{round(ang2, 2)}"
        }
    })

    for idx in [*KP["chin_length"], *KP["chin_width"], *KP["golden_triangle_triplet"], *KP["chin_angle_triplet"]]:
        put_point_with_id(img, pts[idx], idx, WHITE, pr)
    
    # draw_text_cn(img, "下巴/黄金三角测量", (20, 30), color=WHITE, bg_color=(50,50,50), font_px=22)
    return res


# 虚线矩形
def rect_from_indices(idxs, pts):
    xs = [pts[i][0] for i in idxs]
    ys = [pts[i][1] for i in idxs]
    return min(xs), min(ys), max(xs), max(ys)

def dashed_rect(img, x1, y1, x2, y2, color, thickness):
    draw_dashed_line(img, (x1, y1), (x2, y1), color, thickness)
    draw_dashed_line(img, (x1, y2), (x2, y2), color, thickness)
    draw_dashed_line(img, (x1, y1), (x1, y2), color, thickness)
    draw_dashed_line(img, (x2, y1), (x2, y2), color, thickness)

def compute_and_draw_group5(img, pts, thick, pr, scale) -> Dict:
    """鼻翼与口唇测量"""
    res = {}

    # 关键点连线
    nose_pairs = [(6, 122), (122, 188), (188, 174), (174, 236), (236, 198), (198, 209), (209, 129), (129, 98), (98, 97), (97, 2), (2, 326), (326, 327), (327, 358), (358, 429), (429, 420), (420, 456), (456, 399), (399, 412), (412, 351)]
    mouth_pairs = [(0, 37), (37, 40), (40, 61), (61, 91), (91, 84), (84, 314), (314, 321), (321, 291), (61, 191), (191, 80), (80, 13), (13, 310), (310, 415), (415, 291), (291, 270), (270, 267)]

    nose_pts_idx = set([i for a, b in nose_pairs for i in (a, b)])
    mouth_pts_idx = set([i for a, b in mouth_pairs for i in (a, b)])
    for i in nose_pts_idx:
        put_point_with_id(img, pts[i], i, WHITE, pr)
    for i in mouth_pts_idx:
        put_point_with_id(img, pts[i], i, WHITE, pr)

    # 画连线
    for a, b in nose_pairs:
        cv2.line(img, to_int_pt(pts[a]), to_int_pt(pts[b]), WHITE, max(1, thick-1), cv2.LINE_AA)
    if nose_pairs:
        cv2.line(img, to_int_pt(pts[nose_pairs[-1][1]]), to_int_pt(pts[nose_pairs[0][0]]), WHITE, max(1, thick-1), cv2.LINE_AA)

    for a, b in mouth_pairs:
        cv2.line(img, to_int_pt(pts[a]), to_int_pt(pts[b]), WHITE, max(1, thick-1), cv2.LINE_AA)
    if mouth_pairs:
        cv2.line(img, to_int_pt(pts[mouth_pairs[-1][1]]), to_int_pt(pts[mouth_pairs[0][0]]), WHITE, max(1, thick-1), cv2.LINE_AA)

    nx1, ny1, nx2, ny2 = rect_from_indices(nose_pts_idx, pts)
    mx1, my1, mx2, my2 = rect_from_indices(mouth_pts_idx, pts)

    pad = int(6 * scale)
    nx1, ny1 = max(0, nx1 - pad), max(0, ny1 - pad)
    nx2, ny2 = min(img.shape[1]-1, nx2 + pad), min(img.shape[0]-1, ny2 + pad)

    mx1, my1 = max(0, mx1 - pad), max(0, my1 - pad)
    mx2, my2 = min(img.shape[1]-1, mx2 + pad), min(img.shape[0]-1, my2 + pad)

    dashed_rect(img, nx1, ny1, nx2, ny2, WHITE, thick)
    dashed_rect(img, mx1, my1, mx2, my2, WHITE, thick)

    # 尺寸标注
    nose_w, nose_h = abs(nx2 - nx1), abs(ny2 - ny1)
    mouth_w, mouth_h = abs(mx2 - mx1), abs(my2 - my1)
    short_span = int(24 * scale)

    nose_y_below = min(img.shape[0]-5, ny2 + int(14 * scale))
    draw_horizontal_measurement(img, nose_y_below+5, nx1, nx2, f"鼻宽:{fmt_mm(nose_w)}", PURPLE, thick, aux_guides=False, guide_span=short_span)
    nose_x_left = max(5, nx1 - int(14 * scale))
    draw_vertical_measurement(img, nose_x_left, ny1, ny2, f"鼻长:{fmt_mm(nose_h)}", PURPLE, thick, aux_guides=False, guide_span=short_span)

    mouth_y_below = min(img.shape[0]-5, my2 + int(14 * scale))
    draw_horizontal_measurement(img, mouth_y_below+5, mx1, mx2, f"唇宽:{fmt_mm(mouth_w)}", PURPLE, thick, aux_guides=False, guide_span=short_span)
    mouth_x_left = max(5, mx1 - int(14 * scale))
    draw_vertical_measurement(img, mouth_x_left, my1, my2, f"唇高:{fmt_mm(mouth_h)}", PURPLE, thick, aux_guides=False, guide_span=short_span)

    # 鼻翼宽度
    aw = dist(pts[KP["alar_width"][0]], pts[KP["alar_width"][1]])

    # 嘴唇厚度
    upper_thickness = dist(pts[KP["lip_thickness_upper"][0]], pts[KP["lip_thickness_upper"][1]])
    lower_thickness = dist(pts[KP["lip_thickness_lower"][0]], pts[KP["lip_thickness_lower"][1]])

    # 嘴角弯曲度
    a = pts[13]
    o = pts[291]
    b = (o[0], o[1] - 100)
    mouth_corner_angle_deg = draw_angle_dashed(img, a, o, b, WHITE, thick, radius=int(40*scale), label="嘴角弯曲度")

    res.update({
        # "nose_width_detailed": ["鼻宽", round(px_to_mm(nose_w), 2), [int(nx1), int(nose_y_below)], [int(nx2), int(nose_y_below)]],
        # "nose_height_detailed": ["鼻长", round(px_to_mm(nose_h), 2), [int(nose_x_left), int(ny1)], [int(nose_x_left), int(ny2)]],
        # "lip_width_detailed": ["唇宽", round(px_to_mm(mouth_w), 2), [int(mx1), int(mouth_y_below)], [int(mx2), int(mouth_y_below)]],
        # "lip_height_detailed": ["唇高", round(px_to_mm(mouth_h), 2), [int(mouth_x_left), int(my1)], [int(mouth_x_left), int(my2)]],
        # "alar_width_detailed": ["鼻翼宽度", round(px_to_mm(aw), 2), [int(pts[KP["alar_width"][0]][0]), int(pts[KP["alar_width"][0]][1])], [int(pts[KP["alar_width"][1]][0]), int(pts[KP["alar_width"][1]][1])]],
        # "lip_thickness_upper_detailed": ["上唇厚度", round(px_to_mm(upper_thickness), 2), [int(pts[KP["lip_thickness_upper"][0]][0]), int(pts[KP["lip_thickness_upper"][0]][1])], [int(pts[KP["lip_thickness_upper"][1]][0]), int(pts[KP["lip_thickness_upper"][1]][1])]],
        # "lip_thickness_lower_detailed": ["下唇厚度", round(px_to_mm(lower_thickness), 2), [int(pts[KP["lip_thickness_lower"][0]][0]), int(pts[KP["lip_thickness_lower"][0]][1])], [int(pts[KP["lip_thickness_lower"][1]][0]), int(pts[KP["lip_thickness_lower"][1]][1])]],
        # "mouth_corner_angle_detailed": ["嘴角弯曲度", round(mouth_corner_angle_deg, 2) if mouth_corner_angle_deg is not None else None, [int(a[0]), int(a[1])], [int(o[0]), int(o[1])], [int(b[0]), int(b[1])]]
        "nose":{
            "name":"标准鼻",
            "alar_width_detailed":{
                "name":"鼻翼宽度",
                "angle":f"{round(px_to_mm(aw), 2)}"
            }
        },
        "lip":{
            "name":"厚唇",
            "lip_thickness":{
                "name":"嘴唇厚度",
                "angle":f"{round(px_to_mm(upper_thickness), 2)}"
            },
            "mouth_corner_angle":{
                "name":"嘴角弯曲度",
                "angle":f"{round(mouth_corner_angle_deg, 2) if mouth_corner_angle_deg is not None else None}"
            }
        }
    })

    # draw_text_cn(img, "鼻翼与口唇测量", (20, 30), color=WHITE, bg_color=(50,50,50), font_px=22)
    return res

def _rect_from_indices_local(idxs, pts, img):
    xs = [pts[i][0] for i in idxs if i in pts]
    ys = [pts[i][1] for i in idxs if i in pts]
    if not xs or not ys:
        return None
    x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img.shape[1] - 1, x2), min(img.shape[0] - 1, y2)
    return x1, y1, x2, y2

def _norm(a):
    while a <= -math.pi: a += 2*math.pi
    while a > math.pi: a -= 2*math.pi
    return a

def _draw_inner_arc(img, O, A, B, radius, color, thickness=2, steps=60):
    O = np.array(O, dtype=float); A = np.array(A, dtype=float); B = np.array(B, dtype=float)
    v1 = A - O; v2 = B - O
    if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
        return 0.0
    a1 = math.atan2(-(v1[1]), v1[0]); a2 = math.atan2(-(v2[1]), v2[0])

    a1 = _norm(a1); a2 = _norm(a2)
    d = _norm(a2 - a1)
    if abs(d) > math.pi:
        d = -math.copysign(2*math.pi - abs(d), d)
    n = max(8, int(steps * abs(d) / math.pi))
    cx, cy = int(round(O[0])), int(round(O[1]))
    for i in range(n):
        t1 = a1 + d * (i / n); t2 = a1 + d * ((i + 0.5) / n)
        if i % 2 == 0:
            p1 = (int(cx + radius * math.cos(t1)), int(cy - radius * math.sin(t1)))
            p2 = (int(cx + radius * math.cos(t2)), int(cy - radius * math.sin(t2)))
            cv2.line(img, p1, p2, color, max(1, thickness-1), cv2.LINE_AA)
    return math.degrees(d)

def _draw_closed_poly(img, pts, idxs, color, thickness):
    pts_list = [to_int_pt(pts[i]) for i in idxs if i in pts]
    if len(pts_list) >= 3:
        cnt = np.array(pts_list, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(img, [cnt], isClosed=True, color=color, thickness=thickness, lineType=cv2.LINE_AA)

def compute_and_draw_group6(img, pts, thick, pr, scale) -> Dict:
    """右眼和右眉毛测量（定制规则：矩形轴按105/53与70/55，眉宽77-右端，眉高105-53，挑度107–(105,66)，弯度55–46 vs 水平）"""
    res = {}

    PURPLE = (128, 0, 128)  # 实线统一紫色（BGR）



    # 关键点集合
    eyebrow_points = [107, 55, 105, 53, 52, 66, 70, 46]
    eye_points = [33, 133, 159, 145, 157]

    # ===== 底层：虚线矩形（先画） =====
    # 眉毛矩形：水平轴=105/53 的 y，竖直轴=70/55 的 x
    if all(i in pts for i in [105, 53, 70, 55]):
        y_top = min(pts[105][1], pts[53][1]); y_bot = max(pts[105][1], pts[53][1])
        x_left = min(pts[70][0], pts[55][0]); x_right = max(pts[70][0], pts[55][0])
        # 四条边（仅在矩形范围内，不做延长）
        draw_dashed_line(img, (x_left, y_top), (x_right, y_top), WHITE, thick)
        draw_dashed_line(img, (x_left, y_bot), (x_right, y_bot), WHITE, thick)
        draw_dashed_line(img, (x_left, y_top), (x_left, y_bot), WHITE, thick)
        draw_dashed_line(img, (x_right, y_top), (x_right, y_bot), WHITE, thick)
        # 55 的水平轴虚线（底层），长度与眉毛矩形水平线一致
        if 55 in pts:
            y55 = pts[55][1]
            draw_dashed_line(img, (x_left, y55), (x_right, y55), WHITE, max(1, thick-1))
        brow_rect = (x_left, y_top, x_right, y_bot)
    else:
        brow_rect = _rect_from_indices_local(eyebrow_points, pts, img)
        if brow_rect is not None:
            bx1, by1, bx2, by2 = brow_rect
            draw_dashed_line(img, (bx1, by1), (bx2, by1), WHITE, thick)
            draw_dashed_line(img, (bx1, by2), (bx2, by2), WHITE, thick)
            draw_dashed_line(img, (bx1, by1), (bx1, by2), WHITE, thick)
            draw_dashed_line(img, (bx2, by1), (bx2, by2), WHITE, thick)

    # 眼睛矩形：边线“画到关键点上”——用眼部点的 min/max 边界
    eye_rect = _rect_from_indices_local(eye_points, pts, img)
    if eye_rect is not None:
        ex1, ey1, ex2, ey2 = eye_rect
        draw_dashed_line(img, (ex1, ey1), (ex2, ey1), WHITE, thick)
        draw_dashed_line(img, (ex1, ey2), (ex2, ey2), WHITE, thick)
        draw_dashed_line(img, (ex1, ey1), (ex1, ey2), WHITE, thick)
        draw_dashed_line(img, (ex2, ey1), (ex2, ey2), WHITE, thick)

    # 55 的虚线圆（底层）

    if 55 in pts and 65 in pts:
        center = pts[55]
        mid_5565 = ((pts[55][0] + pts[65][0]) // 2, (pts[55][1] + pts[65][1]) // 2)
        radius_c = int(dist(center, mid_5565))
        segs = 48
        for i in range(segs):
            if i % 2 == 0:
                ang1 = 2 * math.pi * i / segs
                ang2 = 2 * math.pi * (i + 1) / segs
                p1 = (int(center[0] + radius_c * math.cos(ang1)), int(center[1] + radius_c * math.sin(ang1)))
                p2 = (int(center[0] + radius_c * math.cos(ang2)), int(center[1] + radius_c * math.sin(ang2)))
                cv2.line(img, p1, p2, WHITE, max(1, thick - 1), cv2.LINE_AA)


    # ===== 中层：关键点与闭合连线（紫色） =====
    eyebrow_order = [46, 53, 52, 65, 55, 107, 66, 105, 63, 70]
    eye_order = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
 
    _draw_closed_poly(img, pts, eyebrow_order, WHITE, max(1, thick-1))
    _draw_closed_poly(img, pts, eye_order, WHITE, max(1, thick-1))
    # 关键点（白色细线风格，参考组5）
    for idx in set(eyebrow_points + eye_points):
        if idx in pts:
            put_point_with_id(img, pts[idx], idx, WHITE, pr)

    # ===== 顶层：箭头线/角度/文本 =====
    # 1) 眉毛宽度：70 到  55

    if 70 in pts and 55 in pts:
        pL = pts[70]
        pR = pts[55]
        val = abs(pR[0] - pL[0])
        y_out = (brow_rect[1] - int(12 * scale)) if brow_rect else max(0, min(pL[1], pR[1]) - int(12 * scale))
        info = draw_horizontal_measurement(img, y_out, pL[0], pR[0], "", WHITE, thick, aux_guides=False)
        cv2.arrowedLine(img, (info['start'][0], info['start'][1]), (info['end'][0], info['end'][1]), PURPLE, thick, cv2.LINE_AA, 0, 0.02)
        cv2.arrowedLine(img, (info['end'][0], info['end'][1]), (info['start'][0], info['start'][1]), PURPLE, thick, cv2.LINE_AA, 0, 0.02)
        # 文本水平位置取 [105,53] 中值
        font_px = max(14, int(18 * 0.6))
        tx = int(((pts[105][0] if 105 in pts else pL[0]) + (pts[53][0] if 53 in pts else pR[0])) / 2)
        draw_text_cn(img, f"眉宽:{fmt_mm(val)}", (tx, int(y_out - 6)), color=WHITE, font_px=font_px)
        res["eyebrow_width"] = {"name":"眉毛宽度", "angle":f"{round(px_to_mm(val), 2)}"}


    # 2) 眉毛高度：105 与 53

    if 105 in pts and 53 in pts:
        pT, pB = pts[105], pts[53]
        val = abs(pT[1] - pB[1])
        x_out = (brow_rect[0] - int(12 * scale)) if brow_rect else max(0, min(pT[0], pB[0]) - int(12 * scale))
        info = draw_vertical_measurement(img, x_out, pT[1], pB[1], "", WHITE, thick, aux_guides=False)
        cv2.arrowedLine(img, (info['start'][0], info['start'][1]), (info['end'][0], info['end'][1]), PURPLE, thick, cv2.LINE_AA, 0, 0.02)
        cv2.arrowedLine(img, (info['end'][0], info['end'][1]), (info['start'][0], info['start'][1]), PURPLE, thick, cv2.LINE_AA, 0, 0.02)
        # 固定文本在箭头线左侧
        mid_y = (pT[1] + pB[1]) // 2
        font_px = max(14, int(18 * 0.6))
        draw_text_cn(img, f"眉高:{fmt_mm(val)}", (int(x_out - 10 - font_px*5.5), int(mid_y + 6)), color=WHITE, font_px=font_px)
        res["eyebrow_height"] = {"name":"眉毛高度","angle":f"{round(px_to_mm(val), 2)}"}


    # 3) 眉毛挑度：107 顶点；连 107-105（白色），不再重复 107-66；弧在内侧（紫色虚线）

    if 107 in pts and 105 in pts and 66 in pts:
        A, O, B = pts[105], pts[107], pts[66]
        cv2.line(img, to_int_pt(O), to_int_pt(A), WHITE, thick, cv2.LINE_AA)
        ang_signed = _draw_inner_arc(img, O, A, B, radius=int(40*scale), color=PURPLE, thickness=thick)
        draw_text_cn(img, f"眉挑度:{abs(ang_signed):.2f}°", (to_int_pt(O)[0]+8, to_int_pt(O)[1]-8), color=WHITE, font_px=max(14, int(18*0.6)))
        lp = get_angle_line_endpoints(A, O, B, radius=int(40*scale))
        res["eyebrow_arch_angle_detailed"] = {"name":"眉毛挑度", "angle":f"{round(abs(ang_signed), 2)}"}


    # 4) 眉毛弯度：55 水平线 vs 55-46
    if 55 in pts and 46 in pts:
        O = pts[55]; A = pts[46]
        H = (O[0] + int(50*scale), O[1])
        # 去掉55附近的紫色虚线半圆，只保留白色边线可视化（顶层）
        # 先计算角度（不画弧）
        # 计算 55-46 与 水平轴的夹角（带符号），并规范到非钝角
        v1x, v1y = A[0]-O[0], A[1]-O[1]
        v2x, v2y = H[0]-O[0], H[1]-O[1]
        n1 = math.hypot(v1x, v1y); n2 = math.hypot(v2x, v2y)
        ang_signed = 0.0
        if n1 > 0 and n2 > 0:
            dot = v1x*v2x + v1y*v2y
            cosv = max(-1.0, min(1.0, dot/(n1*n2)))
            ang = math.degrees(math.acos(cosv))
            # 确定符号（用叉积判断朝向）
            cross = v1x*v2y - v1y*v2x
            ang_signed = ang if cross >= 0 else -ang
            if abs(ang_signed) > 90:
                ang_signed = math.copysign(180 - abs(ang_signed), ang_signed)
        # 文本
        draw_text_cn(img, f"眉弯度:{ang_signed:.2f}°", (to_int_pt(O)[0]+8, to_int_pt(O)[1]+22), color=WHITE, font_px=max(14, int(18*0.6)))
        # 顶层：画 55-46 的白色实线
        cv2.line(img, to_int_pt(O), to_int_pt(A), WHITE, thick, cv2.LINE_AA)
        lp2 = get_angle_line_endpoints(A, O, H, radius=int(40*scale))
        res["eyebrow_curve_angle"] = {"name":"眉毛弯度","angle":f"{round(ang_signed, 2)}"}

    # 5) 眼睛高度：外置（左侧）
    if 159 in pts and 145 in pts:
        pT, pB = pts[159], pts[145]
        val = abs(pT[1] - pB[1])
        x_out = (eye_rect[0] - int(12 * scale)) if eye_rect else max(0, max(pT[0], pB[0]) + int(12 * scale))
        info = draw_vertical_measurement(img, x_out, pT[1], pB[1], "", WHITE, thick, aux_guides=False)
        cv2.arrowedLine(img, (info['start'][0], info['start'][1]), (info['end'][0], info['end'][1]), PURPLE, thick, cv2.LINE_AA, 0, 0.02)
        cv2.arrowedLine(img, (info['end'][0], info['end'][1]), (info['start'][0], info['start'][1]), PURPLE, thick, cv2.LINE_AA, 0, 0.02)
        # 眼高标注：文本水平位置取[159,145]中值，固定在箭头线左侧
        mid_y = (pT[1] + pB[1]) // 2
        font_px = max(14, int(18 * 0.6))
        # 文本置于箭头线左侧
        tx = int(x_out - 10 - font_px * 5.5)
        draw_text_cn(img, f"眼高:{fmt_mm(val)}", (tx, int(mid_y + 6)), color=WHITE, font_px=font_px)
        res["eye_height"] = {"name":"眼睛高度", "angle":f"{round(px_to_mm(val), 2)}"}

    # 6) 眼睛宽度：外置（下方）

    if 33 in pts and 133 in pts:
        pL, pR = pts[33], pts[133]
        val = abs(pR[0] - pL[0])
        y_out = (eye_rect[3] + int(12 * scale)) if eye_rect else max(0, max(pL[1], pR[1]) + int(12 * scale))
        info = draw_horizontal_measurement(img, y_out, pL[0], pR[0], "", WHITE, thick, aux_guides=False)
        cv2.arrowedLine(img, (info['start'][0], info['start'][1]), (info['end'][0], info['end'][1]), PURPLE, thick, cv2.LINE_AA, 0, 0.02)
        cv2.arrowedLine(img, (info['end'][0], info['end'][1]), (info['start'][0], info['start'][1]), PURPLE, thick, cv2.LINE_AA, 0, 0.02)
        # 眼宽标注：参考组5“唇宽”，将文本放在箭头线正上方
        mid_x = (pL[0] + pR[0]) // 2
        ty = int(y_out - 6)
        font_px = max(14, int(18 * 0.6))
        draw_text_cn(img, f"眼宽:{fmt_mm(val)}", (int(mid_x), ty), color=WHITE, font_px=font_px)
        res["eye_width"] = {"name":"眼睛宽度","angle":f"{round(px_to_mm(val), 2)}"}


    # 7) 内眦角：白色边线 + 紫色虚线内角弧

    if 157 in pts and 133 in pts and 145 in pts:
        A, O, B = pts[157], pts[133], pts[145]
        cv2.line(img, to_int_pt(O), to_int_pt(A), WHITE, thick, cv2.LINE_AA)
        cv2.line(img, to_int_pt(O), to_int_pt(B), WHITE, thick, cv2.LINE_AA)
        ang_signed = _draw_inner_arc(img, O, A, B, radius=int(40*scale), color=WHITE, thickness=thick)
        draw_text_cn(img, f"内眦角:{abs(ang_signed):.2f}°", (to_int_pt(O)[0]+8, to_int_pt(O)[1]-24), color=WHITE, font_px=max(14, int(18*0.6)))
        lp3 = get_angle_line_endpoints(A, O, B, radius=int(40*scale))
        res["inner_canthus_angle"] = {"name":"内眦角度数","angle":f"{round(abs(ang_signed), 2)}"}


    # 标题
    draw_text_cn(img, "右眼与右眉毛测量", (20, 30), color=WHITE, bg_color=(50,50,50), font_px=22)

    return res

def safe_filename(name: str) -> str:
    return "".join((c if c not in '<>:"/\\|?*' else '_') for c in name)

class FaceMeasurementService:
    """人脸测量服务类"""
    
    def __init__(self):
        """初始化服务"""
        pass
    
    def process_face_measurement(self, image_path: str, uuid_folder: str) -> Dict:
        """
        处理单张图片的人脸测量
        :param image_path: 图片路径
        :param uuid_folder: UUID文件夹路径
        :return: 测量结果字典
        """
        # 读取图片
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"无法读取图片: {image_path}")

        thickness, point_radius, scale = calc_adaptive(img.shape)

        # 导入MediaPipe
        try:
            with quiet_logs():
                import mediapipe as mp
        except ImportError:
            raise ImportError("未安装 mediapipe，请先安装：pip install mediapipe")

        # 人脸检测
        mp_face_mesh = mp.solutions.face_mesh
        
        with mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.3
        ) as fm:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            res = fm.process(rgb)
        
        if not res.multi_face_landmarks:
            raise ValueError("未检测到人脸")

        # 转换关键点坐标
        lms = res.multi_face_landmarks[0].landmark
        pts = {}
        for i, lm in enumerate(lms):
            pts[i] = normalized_to_pixel(lm, img.shape)
        
        # 设置脸部虚线裁剪矩形
        set_face_rect_from_pts(pts)

        # 创建透明画布（与原图等大）
        h, w, c = img.shape
        
        # 五个测量组
        group_funcs = [
            ("1_脸部整体几何","Overall_face", compute_and_draw_group1),
            ("2_三庭","Column_Ratio", compute_and_draw_group2),
            ("3_眼部","Eye_area", compute_and_draw_group3),
            ("4_下巴与黄金三角","Golden_Triangle", compute_and_draw_group4),
            ("5_鼻翼口唇","Nostril", compute_and_draw_group5),
            ("6_右眼右眉","Right_Eye_Brow", compute_and_draw_group6),
        ]
        
        results = {}
        saved_files = []
        
        for title,e_name, func in group_funcs:
            try:
                # 创建透明画布
                transparent_canvas = np.zeros((h, w, 4), dtype=np.uint8)  # RGBA格式
                
                # 将透明画布转为BGR格式用于绘制
                bgr_canvas = cv2.cvtColor(transparent_canvas[:,:,:3], cv2.COLOR_RGB2BGR)
                
                # 执行测量和绘制
                measurement_data = func(bgr_canvas, pts, thickness, point_radius, scale)
                
                # 将BGR画布转回RGBA，设置透明度
                rgba_canvas = cv2.cvtColor(bgr_canvas, cv2.COLOR_BGR2RGBA)
                
                # 对于非黑色像素，设置为不透明
                mask = np.any(bgr_canvas != [0, 0, 0], axis=2)
                rgba_canvas[mask, 3] = 255  # 设置alpha通道为不透明
                rgba_canvas[~mask, 3] = 0   # 黑色像素保持透明
                
                # 保存文件
                safe_title = safe_filename(e_name)
                output_path = os.path.join(uuid_folder, f"measurement_{safe_title}.png")
                
                # 使用cv2保存PNG（支持透明通道）
                success = cv2.imwrite(output_path, rgba_canvas)
                if success:
                    saved_files.append(output_path)
                    results[e_name] = measurement_data
                else:
                    results[e_name] = {"error": f"保存文件失败: {output_path}"}
                    
            except Exception as e:
                results[e_name] = {"error": str(e)}
        
        # 保存测量数据JSON
        json_path = os.path.join(uuid_folder, "measurements.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        return {
            "success": True,
            "measurements": results,
            "saved_files": saved_files,
            "json_path": json_path
        }