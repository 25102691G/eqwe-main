"""
人脸对齐API控制器
接收图片，进行人脸检测、对齐和裁剪，返回处理后的图片
"""

import os
import cv2
import math
import time
import numpy as np
from flask import request, send_file
from pathlib import Path
import sys
from dotenv import load_dotenv

load_dotenv()

# 添加项目根目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from api.controllers.v1 import bp
from api.controllers.v1.image_decode import decode_image_bytes_to_bgr
from api.controllers.v1.mobile_contract import (
    build_mobile_asset_url,
    build_object_key,
    parse_browser_file_path,
)
from api.middleware.storage.cloud_storage_service import CloudStorageService
from api.middleware.storage.secure_token_service import SecureTokenService

# 配置上传文件夹 - 指向项目根目录
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
UPLOAD_BASE_DIR = os.path.join(project_root, "upload_files")
Path(UPLOAD_BASE_DIR).mkdir(exist_ok=True)
SAVE_LOCAL_IMAGES = os.getenv("SAVE_LOCAL_IMAGES", "false").lower() == "true"


class FaceAlignService:
    """人脸对齐服务类"""
    
    def __init__(self):
        self.face_cascade = None
        self.eye_cascade = None
        self._load_cascades()
    
    def _load_cascades(self):
        """加载Haar级联分类器"""
        try:
            haar_path = cv2.data.haarcascades
            face_model = os.path.join(haar_path, "haarcascade_frontalface_default.xml")
            eye_model = os.path.join(haar_path, "haarcascade_eye.xml")

            self.face_cascade = cv2.CascadeClassifier(face_model)
            self.eye_cascade = cv2.CascadeClassifier(eye_model)

            if self.face_cascade.empty():
                raise RuntimeError(f"无法加载人脸检测模型: {face_model}")
            if self.eye_cascade.empty():
                raise RuntimeError(f"无法加载眼睛检测模型: {eye_model}")
                
        except Exception as e:
            raise RuntimeError(f"模型加载失败: {str(e)}")
    
    def detect_faces(self, gray):
        """检测人脸"""
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=6,
            flags=cv2.CASCADE_SCALE_IMAGE,
            minSize=(80, 80)
        )
        return faces
    
    def pick_largest(self, rects):
        """选择最大的矩形区域"""
        if rects is None or len(rects) == 0:
            return None
        return max(rects, key=lambda r: r[2] * r[3])  # max by area
    
    def detect_eyes_in_face(self, gray, face_rect):
        """在人脸区域内检测眼睛"""
        x, y, w, h = face_rect
        roi_gray = gray[y:y + h, x:x + w]
        
        eyes = self.eye_cascade.detectMultiScale(
            roi_gray,
            scaleFactor=1.1,
            minNeighbors=6,
            flags=cv2.CASCADE_SCALE_IMAGE,
            minSize=(25, 25)
        )
        
        if len(eyes) < 2:
            return None

        # 改进的眼睛选择逻辑
        upper_cut = int(h * 0.6)
        lower_cut = int(h * 0.2)
        candidates = []
        
        for (ex, ey, ew, eh) in eyes:
            eye_center_y = ey + eh / 2
            if lower_cut <= eye_center_y <= upper_cut:
                candidates.append((ex, ey, ew, eh))
        
        if len(candidates) < 2:
            candidates = eyes
        
        candidates = sorted(candidates, key=lambda e: e[2] * e[3], reverse=True)[:2]

        if len(candidates) < 2:
            return None

        centers = []
        for (ex, ey, ew, eh) in candidates:
            cx = x + ex + ew / 2.0
            cy = y + ey + eh / 2.0
            centers.append((cx, cy, ew, eh))
        
        centers.sort(key=lambda c: c[0])
        
        left_eye, right_eye = centers[0], centers[1]
        eye_distance = abs(right_eye[0] - left_eye[0])
        face_width = w
        
        if eye_distance < face_width * 0.2 or eye_distance > face_width * 0.6:
            return None
            
        return (left_eye[0], left_eye[1]), (right_eye[0], right_eye[1])
    
    def rotate_to_align_eyes(self, image, left_eye, right_eye):
        """旋转图像使双眼水平对齐"""
        # 计算两眼连线的角度
        dx = right_eye[0] - left_eye[0]
        dy = right_eye[1] - left_eye[1]
        angle = math.degrees(math.atan2(dy, dx))
        
        # 计算旋转中心
        center = ((left_eye[0] + right_eye[0]) / 2.0, (left_eye[1] + right_eye[1]) / 2.0)
        
        # 获取旋转矩阵
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # 计算旋转后的图像边界
        cos_val = abs(M[0, 0])
        sin_val = abs(M[0, 1])
        new_w = int(image.shape[1] * cos_val + image.shape[0] * sin_val)
        new_h = int(image.shape[1] * sin_val + image.shape[0] * cos_val)
        
        # 调整旋转矩阵以考虑平移
        M[0, 2] += (new_w - image.shape[1]) / 2
        M[1, 2] += (new_h - image.shape[0]) / 2
        
        # 执行旋转
        rotated = cv2.warpAffine(
            image,
            M,
            (new_w, new_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0)
        )
        return rotated, M
    
    def crop_face_region(self, image, left_eye, right_eye, margin_ratio=0.3):
        """基于眼睛位置智能裁剪人脸区域"""
        # 计算两眼之间的距离
        eye_distance = math.sqrt((right_eye[0] - left_eye[0])**2 + (right_eye[1] - left_eye[1])**2)
        
        # 根据眼距计算人脸裁剪尺寸
        face_radius = eye_distance * 1.8
        
        # 计算裁剪中心（两眼中心稍微向下偏移）
        center_x = (left_eye[0] + right_eye[0]) / 2
        center_y = (left_eye[1] + right_eye[1]) / 2 + eye_distance * 0.4
        
        # 应用边距
        crop_size = int(face_radius * (1 + margin_ratio * 2))
        
        # 计算裁剪区域
        x1 = int(center_x - crop_size / 2)
        y1 = int(center_y - crop_size / 2)
        x2 = x1 + crop_size
        y2 = y1 + crop_size
        
        # 确保裁剪区域在图像范围内
        h, w = image.shape[:2]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)
        
        # 裁剪
        cropped = image[y1:y2, x1:x2]
        
        return cropped
    
    def process_image(self, image, margin_ratio=0.3):
        """处理单张图像：检测、对齐、裁剪"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 检测人脸
        faces = self.detect_faces(gray)
        face = self.pick_largest(faces)
        if face is None:
            return None, "未检测到人脸"

        # 检测眼睛
        eyes = self.detect_eyes_in_face(gray, face)
        
        if eyes is not None:
            # 对齐并裁剪
            left_eye, right_eye = eyes
            rotated, M = self.rotate_to_align_eyes(image, left_eye, right_eye)
            
            # 转换眼睛坐标到旋转后的坐标系
            left_eye_rot = cv2.transform(np.array([[[left_eye[0], left_eye[1]]]], dtype=np.float32), M).squeeze()
            right_eye_rot = cv2.transform(np.array([[[right_eye[0], right_eye[1]]]], dtype=np.float32), M).squeeze()
            
            # 基于眼睛位置裁剪，保持原始比例
            cropped = self.crop_face_region(rotated, left_eye_rot, right_eye_rot, margin_ratio)
            
            if cropped is None or cropped.size == 0:
                return None, "裁剪失败"
                
            return cropped, "成功"
        else:
            # 如果没有检测到眼睛，使用简单的人脸裁剪
            x, y, w, h = face
            
            # 计算扩展的裁剪区域
            expand_x = int(w * margin_ratio)
            expand_y = int(h * margin_ratio)
            x1 = max(0, x - expand_x)
            y1 = max(0, y - expand_y) 
            x2 = min(image.shape[1], x + w + expand_x)
            y2 = min(image.shape[0], y + h + expand_y)
            
            cropped = image[y1:y2, x1:x2]
            if cropped.size == 0:
                return None, "基础裁剪失败"
                
            return cropped, "未检测到双眼，使用基础裁剪"


# 从配置文件导入存储和令牌配置
from api.configs.storage_config import STORAGE_CONFIG, TOKEN_CONFIG

# 调试：打印环境变量


# 创建服务实例
face_align_service = FaceAlignService()
cloud_storage = CloudStorageService(STORAGE_CONFIG)
token_service = SecureTokenService(TOKEN_CONFIG)

@bp.route('/face-align', methods=['POST'])
def face_align_api():
    """
    人脸对齐API接口
    接收JSON格式的请求，包含MinIO中的文件信息
    请求格式: {
        "file_path": "http://192.168.3.193:9001/browser/aitest/4c20dff603e7f52f29c1297f0dc86c20",
        "file_name": "4c20dff603e7f52f29c1297f0dc86c20.jpg"
    }
    """
    try:
        # 检查请求是否为JSON格式
        if not request.is_json:
            return {
                'error': '请求格式错误',
                'message': '请使用JSON格式发送请求'
            }, 401
        
        data = request.get_json()
        
        # 验证必需参数
        if not data or 'file_path' not in data or 'file_name' not in data:
            return {
                'error': '参数缺失',
                'message': '请提供 file_path 和 file_name 参数'
            }, 401
        
        file_path = data['file_path'].strip()
        file_name = data['file_name'].strip()
        
        if not file_path or not file_name:
            return {
                'error': '参数不能为空',
                'message': 'file_path 和 file_name 都不能为空'
            }, 401
        
        # 获取参数
        margin_ratio = float(data.get('margin', 0.3))
        margin_ratio = max(0.1, min(1.0, margin_ratio))  # 限制范围
        
        # 解析MinIO路径，提取bucket和object_key
        # file_path格式: http://192.168.3.193:9001/browser/aitest/4c20dff603e7f52f29c1297f0dc86c20
        try:
            bucket_name, folder_path = parse_browser_file_path(file_path)
            object_key = build_object_key(folder_path, file_name)
            task_uuid = os.path.splitext(file_name)[0]
            
            # 创建临时目录用于处理
            temp_dir = os.path.join(UPLOAD_BASE_DIR, f"temp_{task_uuid}")
            Path(temp_dir).mkdir(exist_ok=True)
            
            # 下载图片字节并在内存中解码，避免本地路径读取失败。
            file_extension = os.path.splitext(file_name)[1] or '.jpg'
            temp_image_path = os.path.join(temp_dir, f"temp{file_extension}")

            download_success, image_bytes = cloud_storage.download_to_memory(object_key)
            if not download_success:
                return {
                    'error': '从MinIO下载图片失败',
                    'message': str(image_bytes)
                }, 400

            if SAVE_LOCAL_IMAGES:
                with open(temp_image_path, "wb") as file_handle:
                    file_handle.write(image_bytes)

            image = decode_image_bytes_to_bgr(image_bytes)
            if image is None:
                return {'error': '无法解码下载的图像数据'}, 400
                
        except Exception as e:
            return {
                'error': '处理图片时发生错误',
                'message': str(e)
            }, 400
        
        print(f"开始人脸对齐任务: {task_uuid}")
        print(f"图像尺寸: {image.shape}")
        
        # 处理图像
        cropped_image, message = face_align_service.process_image(image, margin_ratio)
        
        if cropped_image is None:
            return {
                'error': message,
                'uuid': task_uuid
            }, 400
        
        # 保存裁剪后的图片到临时文件
        aligned_path = os.path.join(temp_dir, f"aligned_{file_name}")
        success = cv2.imwrite(aligned_path, cropped_image)
        
        if not success:
            return {'error': '保存处理后的图片失败'}, 500
        
        # 上传对齐后的图片到MinIO，保存为 align.jpg
        aligned_object_key = f"{folder_path}/align.jpg"  # 固定文件名为 align.jpg
        
        # 上传对齐后的图片到MinIO
        upload_success, upload_result = cloud_storage.upload_file(
            aligned_path, 
            aligned_object_key,
            content_type="image/jpeg"
        )
        
        if not upload_success:
            return {'error': f'上传对齐后图片失败: {upload_result}'}, 500
        
        print(f"对齐后图片已保存到MinIO: {aligned_object_key}")
        
        # 返回结果
        result = {
            'uuid': task_uuid,
            'message': message,
            'bucket': bucket_name,
            'folder': folder_path,
            'file_name': file_name,
            'aligned_file_name': 'align.jpg',
            'aligned_object_key': aligned_object_key,
            'original_image_url': build_mobile_asset_url(folder_path, file_name),
            'aligned_image_url': build_mobile_asset_url(folder_path, 'align.jpg'),
            'original_size': {
                'width': image.shape[1],
                'height': image.shape[0]
            },
            'aligned_size': {
                'width': cropped_image.shape[1],
                'height': cropped_image.shape[0]
            },
            'margin_ratio': margin_ratio,
            'file_info': {
                'path': file_path,
                'name': file_name,
                'aligned_object_key': aligned_object_key
            },
            'image_urls': {
                'original_image_url': build_mobile_asset_url(folder_path, file_name),
                'aligned_image_url': build_mobile_asset_url(folder_path, 'align.jpg'),
            },
            'storage': {
                'type': 'cloud',
                'bucket': bucket_name,
                'folder': folder_path,
                'object_key': aligned_object_key
            },
            'status': 'success'
        }
            
        print(f"人脸对齐完成: {task_uuid}")
        print(f"原始尺寸: {image.shape[1]}x{image.shape[0]}")
        print(f"对齐后尺寸: {cropped_image.shape[1]}x{cropped_image.shape[0]}")
        
        return result
        
    except Exception as e:
        print(f"人脸对齐过程中发生错误: {str(e)}")
        return {'error': f'处理失败: {str(e)}'}, 500


@bp.route('/face-align/<task_uuid>/<filename>', methods=['GET'])
def get_face_align_file(task_uuid, filename):
    """
    获取处理结果文件 (本地存储，保持向后兼容)
    """
    try:
        file_path = os.path.join(UPLOAD_BASE_DIR, task_uuid, filename)
        if os.path.exists(file_path):
            return send_file(file_path)
        else:
            return {'error': '文件不存在'}, 404
    except Exception as e:
        return {'error': str(e)}, 500

@bp.route('/face-align/secure/<token>', methods=['GET'])
def get_secure_file(token):
    """
    通过JWT令牌安全访问文件
    支持云存储的预签名URL和访问控制
    """
    try:
        # 验证JWT令牌
        is_valid, result = token_service.validate_token(token)
        if not is_valid:
            return {'error': result}, 403
        
        task_uuid = result['task_uuid']
        filename = result['filename']
        file_type = result['file_type']
        
        # 如果有object_key，说明文件在云存储中
        object_key = result.get('object_key')
        if object_key:
            # 生成预签名URL并重定向
            presigned_url = cloud_storage.generate_presigned_url(object_key, expiration=3600)
            if presigned_url:
                from flask import redirect
                response = redirect(presigned_url, code=302)
                response.headers['X-Remaining-Access'] = str(result['remaining_access'])
                response.headers['X-Expires-At'] = str(result['expires_at'])
                return response
            else:
                # 云存储失败，尝试本地文件
                pass
        
        # 尝试从本地存储获取文件
        file_path = os.path.join(UPLOAD_BASE_DIR, task_uuid, filename)
        if os.path.exists(file_path):
            response = send_file(file_path)
            response.headers['X-Remaining-Access'] = str(result['remaining_access'])
            response.headers['X-Expires-At'] = str(result['expires_at'])
            response.headers['X-Storage-Type'] = 'local'
            return response
        else:
            return {'error': '文件不存在'}, 404
        
    except Exception as e:
        print(f"安全访问文件失败: {str(e)}")
        return {'error': '访问失败'}, 500

@bp.route('/face-align/token/<token>/info', methods=['GET'])
def get_token_info(token):
    """
    获取令牌信息（不消耗访问次数）
    """
    try:
        token_info = token_service.get_token_info(token)
        if token_info:
            return {
                'token_info': token_info,
                'status': 'success'
            }
        else:
            return {'error': '无效的令牌'}, 400
    except Exception as e:
        return {'error': str(e)}, 500

@bp.route('/face-align/cleanup', methods=['POST'])
def cleanup_expired_files():
    """
    清理过期文件和令牌 (建议设置定时任务调用)
    """
    try:
        import shutil
        current_time = time.time()
        expire_threshold = current_time - (7 * 24 * 3600)  # 7天前
        
        # 清理本地文件
        cleaned_local = 0
        if os.path.exists(UPLOAD_BASE_DIR):
            for task_dir in os.listdir(UPLOAD_BASE_DIR):
                task_path = os.path.join(UPLOAD_BASE_DIR, task_dir)
                if os.path.isdir(task_path):
                    # 检查目录修改时间
                    if os.path.getmtime(task_path) < expire_threshold:
                        shutil.rmtree(task_path)
                        cleaned_local += 1
        
        # 清理过期令牌
        cleaned_tokens = token_service.cleanup_expired_records()
        
        # 获取令牌统计
        stats = token_service.get_statistics()
        
        return {
            'message': '清理完成',
            'cleaned_local_dirs': cleaned_local,
            'cleaned_tokens': cleaned_tokens,
            'token_statistics': stats,
            'status': 'success'
        }
        
    except Exception as e:
        return {'error': str(e)}, 500
