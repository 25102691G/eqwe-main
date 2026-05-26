import os
import uuid as uuid_lib
import requests
import re
from flask import request, jsonify, url_for
from pathlib import Path
import sys

# 添加项目根目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from api.controllers.v1 import bp
from api.services.measure_draw import FaceMeasurementService
from api.middleware.storage.cloud_storage_service import CloudStorageService
from api.middleware.storage.secure_token_service import SecureTokenService
from api.configs.storage_config import STORAGE_CONFIG, TOKEN_CONFIG

# 配置
UPLOAD_BASE_DIR = os.path.join(os.getcwd(), "upload_files")

# ============================================================
# 服务实例复用优化：在模块级别创建服务实例（只创建一次）
# ============================================================
print("🔧 初始化人脸测量服务实例...")

# 初始化服务
cloud_storage = CloudStorageService(STORAGE_CONFIG)
token_service = SecureTokenService(TOKEN_CONFIG)
face_measurement_service = FaceMeasurementService()

print("✅ 人脸测量服务实例初始化完成")
# ============================================================

@bp.route('/face-measurement', methods=['POST'])
def face_measurement():
    """
    人脸测量API接口
    接收file_path和file_name，从MinIO下载图片，执行人脸测量并上传结果
    """
    try:
        # 检查请求格式
        if not request.is_json:
            return {'error': '请求必须是JSON格式'}, 400
        
        data = request.get_json()
        if not data:
            return {'error': '请求数据为空'}, 400
        
        # 获取参数
        file_path = data.get('file_path')
        file_name = data.get('file_name')
        
        if not file_path:
            return {'error': '缺少file_path参数'}, 400
        
        if not file_name:
            return {'error': '缺少file_name参数'}, 400
        
        # 解析MinIO路径，提取bucket和object_key
        # file_path格式: http://192.168.3.193:9001/browser/aitest/4c20dff603e7f52f29c1297f0dc86c20
        try:
            path_match = re.search(r'/browser/([^/]+)/(.+)$', file_path)
            if not path_match:
                return {'error': 'MinIO路径格式错误'}, 400
            
            bucket_name = path_match.group(1)  # aitest
            folder_path = path_match.group(2)  # 4c20dff603e7f52f29c1297f0dc86c20
            object_key = f"{folder_path}/{file_name}"  # 完整的对象键
            
            # 从文件名提取UUID作为task_uuid
            task_uuid = os.path.splitext(file_name)[0]
            
            # 创建临时目录用于处理
            temp_dir = os.path.join(UPLOAD_BASE_DIR, folder_path)
            Path(temp_dir).mkdir(exist_ok=True)
            
            # 使用cloud_storage_service下载图片
            file_extension = os.path.splitext(file_name)[1] or '.jpg'
            temp_image_path = os.path.join(temp_dir, f"align{file_extension}")
            
            download_success, download_result = cloud_storage.download_file(
                object_key, temp_image_path
            )
            
            if not download_success:
                return {
                    'error': '从MinIO下载图片失败',
                    'message': str(download_result)
                }, 400
            
        except Exception as e:
            return {
                'error': '处理图片时发生错误',
                'message': str(e)
            }, 400
        
        print(f"开始人脸测量任务: {task_uuid}")
        
        # 使用全局服务实例（已在模块级别初始化，无需重复创建）
        result = face_measurement_service.process_face_measurement(temp_image_path, temp_dir)
        
        if not result.get('success'):
            return {'error': '人脸测量失败'}, 400
        
        # 上传所有生成的文件到MinIO
        uploaded_files = []
        uploaded_urls = []
        
        # 上传测量图片文件
        for saved_file in result.get('saved_files', []):
            if os.path.exists(saved_file):
                # 获取文件名（保持原始名称）
                filename = os.path.basename(saved_file)
                # 构建MinIO对象键
                measure_object_key = f"{folder_path}/{filename}"
                
                # 上传到MinIO
                upload_success, upload_result = cloud_storage.upload_file(
                    saved_file,
                    measure_object_key,
                    content_type="image/png"
                )
                
                # if upload_success:
                #     uploaded_files.append({
                #         'filename': filename,
                #         'object_key': measure_object_key
                #     })
                    
                    # # 生成JWT访问令牌
                    # file_token = token_service.generate_token(
                    #     task_uuid,
                    #     filename,
                    #     file_type='measurement',
                    #     expire_hours=TOKEN_CONFIG['token_expire_hours'],
                    #     max_access=TOKEN_CONFIG['max_access_count'],
                    #     extra_data={'object_key': measure_object_key}
                    # )
                    
                    # # 生成安全访问URL
                    # file_url = url_for('v1.get_secure_file', token=file_token, _external=True)
                    # uploaded_urls.append({
                    #     'filename': filename,
                    #     'url': file_url
                    # })
        
        # 上传JSON文件
        json_path = result.get('json_path')
        if json_path and os.path.exists(json_path):
            json_filename = os.path.basename(json_path)
            json_object_key = f"{folder_path}/{json_filename}"
            
            upload_success, upload_result = cloud_storage.upload_file(
                json_path,
                json_object_key,
                content_type="application/json"
            )
            
            # if upload_success:
            #     uploaded_files.append({
            #         'filename': json_filename,
            #         'object_key': json_object_key
            #     })
                
                # 生成JWT访问令牌
                # json_token = token_service.generate_token(
                #     task_uuid,
                #     json_filename,
                #     file_type='measurement_data',
                #     expire_hours=TOKEN_CONFIG['token_expire_hours'],
                #     max_access=TOKEN_CONFIG['max_access_count'],
                #     extra_data={'object_key': json_object_key}
                # )
                
                # 生成安全访问URL
                # json_url = url_for('v1.get_secure_file', token=json_token, _external=True)
                # uploaded_urls.append({
                #     'filename': json_filename,
                #     'url': json_url
                # })
        
        # 返回结果
        return {
            # 'uuid': task_uuid,
            'success': True,
            # 'measurements': result.get('measurements', {}),
            'file_info': {
                'path': file_path,
                'name': file_name
            },
            # 'uploaded_files': uploaded_files,
            # 'urls': uploaded_urls,
            'storage': {
                'type': 'cloud',
                'bucket': bucket_name,
                'folder': folder_path
            },
            # 'access_info': {
            #     'expire_hours': TOKEN_CONFIG['token_expire_hours'],
            #     'max_access_count': TOKEN_CONFIG['max_access_count']
            # },
            'status': 'success'
        }
        
    except ValueError as ve:
        return {'error': str(ve)}, 400
    except ImportError as ie:
        return {'error': str(ie)}, 500
    except Exception as e:
        return {'error': f'处理失败: {str(e)}'}, 500