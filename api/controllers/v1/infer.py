"""
Flask人脸分析推理服务
接收MinIO路径和图片名称，进行人脸油性和缺水分析，返回结果
"""

import os
import cv2
import json
from datetime import datetime
from flask import request, send_file
from pathlib import Path
import numpy as np
import sys
from concurrent.futures import ThreadPoolExecutor
import threading
from dotenv import load_dotenv

# 加载 .env 文件
load_dotenv()

# 添加项目根目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from api.services.face_detection import FaceDetectionService
from api.services.skin_analysis import SkinAnalysisService
from api.services.skin_class import SkinClassificationService
from api.services.sensitivity_skin import SkinSensitivityService
from api.services.report_scores import (
    calculate_skin_tone_score,
    calculate_smoothness_score,
)
from api.agents import generate_skin_report
from api.controllers.v1 import bp
from api.controllers.v1.image_decode import decode_image_bytes_to_bgr
from api.controllers.v1.mobile_contract import (
    build_analysis_response,
    build_object_key,
    parse_browser_file_path,
)
from api.middleware.storage.cloud_storage_service import CloudStorageService
from api.configs.storage_config import STORAGE_CONFIG
from api.services.detector_strict import run_pipeline
from api.services.nevus import process_image
from api.services.nasolabial import nasolabial_process
# 配置上传文件夹 - 指向项目根目录
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
UPLOAD_BASE_DIR = os.path.join(project_root, "upload_files")
Path(UPLOAD_BASE_DIR).mkdir(exist_ok=True)

# ============================================================
# 本地保存开关：控制是否保存图片到本地磁盘
# ============================================================
# 从环境变量读取，默认为 False（不保存到本地）
SAVE_LOCAL_IMAGES = os.getenv('SAVE_LOCAL_IMAGES', 'false').lower() == 'true'
print(f"🔧 本地保存开关: {'开启' if SAVE_LOCAL_IMAGES else '关闭'}")
# ============================================================

# ============================================================
# 服务实例复用优化：在模块级别创建服务实例（只创建一次）
# ============================================================
print("🔧 初始化服务实例...")

# 创建云存储服务实例
cloud_storage = CloudStorageService(STORAGE_CONFIG)

# 创建人脸分析相关服务实例（全局复用）
face_service = FaceDetectionService()
skin_service = SkinAnalysisService()
skin_class_service = SkinClassificationService()
sensitivity_service = SkinSensitivityService()

print("✅ 服务实例初始化完成")

# ============================================================
# 并发控制优化：限制同时处理的请求数量
# ============================================================
# 信号量：最多允许 3 个请求同时进行分析（可根据服务器性能调整）
MAX_CONCURRENT_REQUESTS = 3
request_semaphore = threading.Semaphore(MAX_CONCURRENT_REQUESTS)

# 统计信息
active_requests = 0
total_requests = 0
request_lock = threading.Lock()

print(f"🔧 并发控制：最大并发请求数 = {MAX_CONCURRENT_REQUESTS}")
# ============================================================

@bp.route('/analyze-face', methods=['POST'])
def analyze_face():
    """
    人脸分析API接口
    接收JSON格式的请求，包含MinIO中的文件信息
    请求格式: {
        "file_path": "http://192.168.3.193:9001/browser/aitest/4c20dff603e7f52f29c1297f0dc86c20",
        "file_name": "4c20dff603e7f52f29c1297f0dc86c20.jpg"
    }
    """
    global active_requests, total_requests
    
    # 尝试获取信号量（如果超过最大并发数，会在这里等待）
    acquired = request_semaphore.acquire(blocking=True, timeout=10)# 超时十秒就抛出异常
    
    if not acquired:
        return {
            'error': '服务繁忙',
            'message': '当前请求过多，请稍后重试',
            'retry_after': 30
        }, 503
    
    try:
        # 更新统计信息
        with request_lock:
            active_requests += 1
            total_requests += 1
            current_active = active_requests
            current_total = total_requests
        
        print(f"📊 当前活跃请求: {current_active}/{MAX_CONCURRENT_REQUESTS}, 总请求数: {current_total}")
        
        # 检查请求是否为JSON格式
        if not request.is_json:
            return {
                'error': '请求格式错误',
                'message': '请使用JSON格式发送请求'
            }, 400
        
        data = request.get_json()
        
        # 验证必需参数
        if not data or 'file_path' not in data or 'file_name' not in data:
            return {
                'error': '参数缺失',
                'message': '请提供 file_path 和 file_name 参数'
            }, 400
        
        file_path = data['file_path'].strip()
        file_name = data['file_name'].strip()
        
        if not file_path or not file_name:
            return {
                'error': '参数不能为空',
                'message': 'file_path 和 file_name 都不能为空'
            }, 400
        
        # 解析MinIO路径，提取bucket和object_key
        # file_path格式: http://192.168.3.193:9001/browser/aitest/4c20dff603e7f52f29c1297f0dc86c20
        try:
            bucket_name, folder_path = parse_browser_file_path(file_path)
            object_key = build_object_key(folder_path, file_name)
            task_uuid = os.path.splitext(file_name)[0]
            
            task_dir = None
            if SAVE_LOCAL_IMAGES:
                task_dir = os.path.join(UPLOAD_BASE_DIR, folder_path)
                Path(task_dir).mkdir(parents=True, exist_ok=True)

            download_success, image_bytes = cloud_storage.download_to_memory(object_key)
            if not download_success:
                return {
                    'error': '从MinIO下载图片失败',
                    'message': str(image_bytes)
                }, 400

            if task_dir is not None:
                file_extension = os.path.splitext(file_name)[1] or '.jpg'
                temp_image_path = os.path.join(task_dir, f"original{file_extension}")
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
        
        print(f"开始分析任务: {task_uuid}")
        print(f"图像尺寸: {image.shape}")
        
        # 使用全局服务实例（已在模块级别初始化，无需重复创建）
        # face_service, skin_service, skin_class_service, sensitivity_service
        
        # 1. 人脸检测和特征点提取
        print("检测人脸特征点...")
        landmarks = face_service.detect_mediapipe_landmarks(image)
        if landmarks is None:
            return {'error': '未检测到人脸'}, 400

        # 2. 创建面部区域掩码
        print("创建面部区域掩码...")
        masks = face_service.create_comprehensive_masks(image, landmarks)
        if masks is None:
            return {'error': '面部区域分割失败'}, 400
        
        # 3. 创建特定皮肤分析区域掩码（额头、鼻梁、脸颊、下巴）
        print("创建皮肤分析区域掩码...")
        analysis_masks = face_service.create_skin_analysis_masks(image, landmarks)
        if analysis_masks is None:
            return {'error': '皮肤分析区域分割失败'}, 400
        
        # 3. 绘制特征点（可选，用于调试）
        landmarks_image = face_service.draw_landmarks(image.copy(), landmarks)
        # 直接编码为字节流，不保存到本地
        _, landmarks_encoded = cv2.imencode('.jpg', landmarks_image)
        landmarks_bytes = landmarks_encoded.tobytes()
        
        # ============================================================
        # 并行处理优化：使用线程池并行执行独立的分析任务
        # ============================================================
        print("🚀 开始并行分析任务...")
        
        # 定义各个独立的分析任务
        def task_oil_analysis():
            """油性检测任务"""
            print("  ⏳ 执行油性检测...")
            hsv_oil_mask, oil_score = skin_service.detect_oil_regions_hsv(
                image, analysis_masks['combined_analysis_regions']
            )
            # 生成可视化
            oil_mask_fixed = skin_service.apply_orange_mask(image, hsv_oil_mask, alpha=0.5)
            # 直接编码为字节流，不保存到本地
            _, oil_encoded = cv2.imencode('.jpg', oil_mask_fixed)
            oil_bytes = oil_encoded.tobytes()
            print("  ✅ 油性检测完成")
            return hsv_oil_mask, oil_score, oil_bytes
        
        def task_moisture_analysis():
            """水分分析任务"""
            print("  ⏳ 执行水分分析...")
            moisture_map, moisture_score = skin_service.calculate_moisture_enhanced(
                image, masks['face_contour']
            )
            # 生成可视化
            moisture_result = skin_service.create_moisture_visualization_enhanced(
                image, moisture_map, masks
            )
            # 直接编码为字节流，不保存到本地
            _, moisture_encoded = cv2.imencode('.jpg', moisture_result)
            moisture_bytes = moisture_encoded.tobytes()
            print("  ✅ 水分分析完成")
            return moisture_map, moisture_score, moisture_bytes
        
        def task_skin_tone_classification():
            """肤色分类任务"""
            print("  ⏳ 执行肤色分类...")
            # 只在需要本地保存时提供路径
            skin_tone_viz_path = os.path.join(task_dir, "skin_tone_classification.jpg") if task_dir else None
            ita_score, skin_tone_category, viz_image = skin_class_service.classify_skin_tone(
                image, 
                save_visualization=True, 
                save_path=skin_tone_viz_path
            )
            skin_tone_result = skin_class_service.create_classification_result(ita_score, skin_tone_category)
            print("  ✅ 肤色分类完成")
            return ita_score, skin_tone_category, skin_tone_result
        
        def task_hyperpigmentation():
            """色沉检测任务"""
            print("  ⏳ 执行色沉检测...")
            hyperpigmentation, zhi_count, se_ban = process_image(image)
            # 直接编码为字节流，不保存到本地
            _, hyper_encoded = cv2.imencode('.jpg', hyperpigmentation)
            hyper_bytes = hyper_encoded.tobytes()
            print("  ✅ 色沉检测完成")
            return hyper_bytes, zhi_count, se_ban
        
        def task_sensitivity_analysis():
            """敏感性分析任务"""
            print("  ⏳ 执行敏感性分析...")
            sensitivity_analysis = sensitivity_service.analyze_skin_sensitivity(
                image,
                alpha=0.65,
                wr=0.75,
                wt=0.25,
                smooth=11,
                use_mediapipe=True,
                save_debug_images=SAVE_LOCAL_IMAGES,  # 使用全局开关
                save_dir=task_dir,  # 如果 task_dir 是 None，服务内部会处理
                mask=masks['skin_only']
            )
            sensitivity_result = sensitivity_service.create_sensitivity_result(sensitivity_analysis)
            print("  ✅ 敏感性分析完成")
            return sensitivity_analysis, sensitivity_result
        
        def task_smoothness_analysis():
            """光滑度分析任务"""
            print("  ⏳ 执行光滑度分析...")
            smooth_json = run_pipeline(image, task_dir, masks)
            print("  ✅ 光滑度分析完成")
            return smooth_json
        
        def task_wrinkle_analysis():
            """皱纹黑眼圈分析任务"""
            print("  ⏳ 执行皱纹黑眼圈分析...")
            imgnasolabial_img, black_eye_img, dic = nasolabial_process(image)
            # 直接编码为字节流，不保存到本地
            _, nasolabial_encoded = cv2.imencode('.jpg', imgnasolabial_img)
            nasolabial_bytes = nasolabial_encoded.tobytes()
            _, black_eye_encoded = cv2.imencode('.jpg', black_eye_img)
            black_eye_bytes = black_eye_encoded.tobytes()
            print("  ✅ 皱纹黑眼圈分析完成")
            return nasolabial_bytes, black_eye_bytes, dic
        
        # 使用线程池并行执行所有任务
        with ThreadPoolExecutor(max_workers=7) as executor:
            # 提交所有任务
            future_oil = executor.submit(task_oil_analysis)
            future_moisture = executor.submit(task_moisture_analysis)
            future_skin_tone = executor.submit(task_skin_tone_classification)
            future_hyperpigmentation = executor.submit(task_hyperpigmentation)
            future_sensitivity = executor.submit(task_sensitivity_analysis)
            future_smoothness = executor.submit(task_smoothness_analysis)
            future_wrinkle = executor.submit(task_wrinkle_analysis)
            
            # 等待所有任务完成并获取结果（返回字节流）
            hsv_oil_mask, oil_score, oil_bytes = future_oil.result()
            moisture_map, moisture_score, moisture_bytes = future_moisture.result()
            ita_score, skin_tone_category, skin_tone_result = future_skin_tone.result()
            hyperpigmentation_bytes, zhi_count, se_ban = future_hyperpigmentation.result()
            sensitivity_analysis, sensitivity_result = future_sensitivity.result()
            smooth_json = future_smoothness.result()
            nasolabial_bytes, black_eye_bytes, dic = future_wrinkle.result()
        
        print("✅ 所有并行分析任务完成！")
        # ============================================================
        
        # 8. 分类皮肤类型（基于油性和水分）
        skin_type = _classify_skin_type(oil_score, moisture_score)
        skin_area_pixels = int(np.count_nonzero(masks['skin_only']))

        smoothness_counts = smooth_json.get("counts", {})
        acne_group_total = int(smoothness_counts.get("acne_group_total", 0))
        blackheads_count = int((smoothness_counts.get("blackheads") or {}).get("counts", 0))
        pores_count = int((smoothness_counts.get("pores") or {}).get("counts", 0))

        skin_tone_score = calculate_skin_tone_score(
            stain_count=int(se_ban),
            skin_area_pixels=skin_area_pixels,
        )
        smoothness_score = calculate_smoothness_score(
            acne_group_total=acne_group_total,
            blackheads_count=blackheads_count,
            pores_count=pores_count,
            skin_area_pixels=skin_area_pixels,
        )


        # 9. 上传结果图片到MinIO（直接从内存上传）
        uploaded_files = []
        
        # 准备需要上传的图片（文件名 -> 字节流）
        result_files_bytes = [
            ('landmarks.jpg', landmarks_bytes),
            ('oil_mask_fixed.jpg', oil_bytes),
            ('moisture_analysis.jpg', moisture_bytes),
            ('hyperpigmentation.jpg', hyperpigmentation_bytes),
            ('imgnasolabial.jpg', nasolabial_bytes),
            ('black_eye.jpg', black_eye_bytes)
        ]
        
        # 添加敏感性分析的overlay_on_white.jpg（从字节流获取）
        if 'image_bytes' in sensitivity_analysis and 'overlay_on_white' in sensitivity_analysis['image_bytes']:
            overlay_white_bytes = sensitivity_analysis['image_bytes']['overlay_on_white']
            result_files_bytes.append(('overlay_on_white.jpg', overlay_white_bytes))
        
        # 添加光滑度图片（从字节流获取）
        if 'image_bytes' in smooth_json:
            for result_filename, file_bytes in smooth_json['image_bytes'].items():
                result_files_bytes.append((result_filename, file_bytes))

        # 直接从内存上传到MinIO
        for filename, file_bytes in result_files_bytes:
            result_object_key = build_object_key(folder_path, filename)
             
            # 直接从内存上传到MinIO
            upload_success, upload_result = cloud_storage.upload_from_memory(
                file_bytes,
                result_object_key,
                content_type="image/jpeg"
            )
            
            if upload_success:
                uploaded_files.append({
                    'filename': filename,
                    'object_key': result_object_key
                })
                print(f"✅ 已上传结果图片到MinIO: {result_object_key}")
            else:
                print(f"❌ 上传失败 {filename}: {upload_result}")
        
        # 10. 创建并上传综合分析结果JSON
        comprehensive_analysis_data = {
            'analysis_results': {
                "oil_moi":{
                    'oil_analysis': {
                        'oil_score': round(oil_score, 2),
                        'description': skin_type["oil_sug"]
                    },
                    'moisture_analysis': {
                        'moisture_score': round(moisture_score, 2),
                        'description': skin_type["moi_sug"]
                    },
                    "score":round((oil_score + moisture_score)/2, 2),
                    "description":"皮肤油性区域和水分程度"
                },
                "skin_color":{
                    "skin_tone_classification": skin_tone_result,
                    "score":skin_tone_score,
                    "description":"根据色沉点位密度估算肤色均匀度得分",
                    "score_detail": {
                        "skin_area_pixels": skin_area_pixels,
                        "stain_count": se_ban,
                        "stain_density_per_10k": round((se_ban * 10000.0) / max(skin_area_pixels, 1), 2),
                    },
                    "hyperpigmentation":{
                        "zhi_count":zhi_count,
                        "se_ban":se_ban,
                        "description":"图片点位代表对应瑕疵",
                        "suggestion":"建议使用温和洗面奶"
                    }
                    },
                "sensitivity":{
                    'sensitivity_analysis': sensitivity_result,
                    "score":sensitivity_result["sensitivity_score"],
                    "description":"根据红度与纹理稳定度估算敏感度得分"},
                "smoothness":{
                    "smooth":smooth_json["counts"],
                    "score":smoothness_score,
                    "description":"根据痘印、粉刺、黑头和毛孔密度估算平滑度得分",
                    "score_detail": {
                        "skin_area_pixels": skin_area_pixels,
                        "acne_group_total": acne_group_total,
                        "blackheads_count": blackheads_count,
                        "pores_count": pores_count,
                    },
                },
                "wrinkles":dic,
            },
            'metadata': {
                'analysis_timestamp': datetime.now().isoformat(),
                'image_info': {
                    'original_size': {
                        'width': image.shape[1],
                        'height': image.shape[0]
                    },
                    'file_name': file_name
                }
            }
        }

        llm_report = generate_skin_report(comprehensive_analysis_data['analysis_results'])
        comprehensive_analysis_data['llm_report'] = llm_report
        
        # 将JSON转换为字节流
        json_str = json.dumps(comprehensive_analysis_data, ensure_ascii=False, indent=2)
        json_bytes = json_str.encode('utf-8')
        
        # 直接从内存上传JSON到MinIO
        json_object_key = f"{folder_path}/analysis_results.json"
        upload_success, upload_result = cloud_storage.upload_from_memory(
            json_bytes,
            json_object_key,
            content_type="application/json"
        )
        
        if upload_success:
            uploaded_files.append({
                'filename': 'analysis_results.json',
                'object_key': json_object_key
            })
            print(f"✅ 已上传综合分析结果JSON到MinIO: {json_object_key}")
        else:
            print(f"❌ 上传JSON失败: {upload_result}")
        
        # 如果需要本地保存JSON（调试模式）
        if SAVE_LOCAL_IMAGES:
            json_path = os.path.join(task_dir, "analysis_results.json")
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(comprehensive_analysis_data, f, ensure_ascii=False, indent=2)
            print(f"💾 已保存JSON到本地: {json_path}")
        
        # 11. 返回结果
        local_files = {
            'landmarks': 'landmarks.jpg',
            'oil_mask_fixed': 'oil_mask_fixed.jpg',
            'moisture_analysis': 'moisture_analysis.jpg',
            'skin_tone_classification': 'skin_tone_classification.jpg',
            'sensitivity_gray': 'sensitivity_gray.jpg',
            'overlay_on_white': 'overlay_on_white.jpg',
            'analysis_results': 'analysis_results.json',
        }
        result = build_analysis_response(
            task_uuid=task_uuid,
            file_path=file_path,
            input_file_name=file_name,
            source_object_key=object_key,
            bucket_name=bucket_name,
            folder_path=folder_path,
            analysis_results=comprehensive_analysis_data['analysis_results'],
            metadata=comprehensive_analysis_data['metadata'],
            uploaded_files=uploaded_files,
            report_object_key=json_object_key,
            llm_report=llm_report,
            local_files=local_files,
        )
        
        print(f"分析完成: {task_uuid}")
        print(f"HSV油性评分: {oil_score:.2f}, 水分评分: {moisture_score:.2f}")
        print(f"皮肤类型: {skin_type}")
        print(f"肤色分类: {skin_tone_category} (ITA: {ita_score:.2f})" if ita_score is not None else f"肤色分类: {skin_tone_category}")
        print(f"敏感性分析: {sensitivity_result['sensitivity_level']} (评分: {sensitivity_result['sensitivity_score']:.2f})")
        return result

    except Exception as e:
        print(f"分析过程中发生错误: {str(e)}")
        return {'error': f'分析失败: {str(e)}'}, 500
    
    finally:
        # 无论成功还是失败，都要释放信号量
        request_semaphore.release()
        with request_lock:
            active_requests -= 1
            print(f"📊 请求完成，当前活跃请求: {active_requests}/{MAX_CONCURRENT_REQUESTS}")

@bp.route('/get-result/<folder_path>/<filename>', methods=['GET'])
def get_result_file(folder_path, filename):
    """
    获取分析结果文件（本地存储，保持向后兼容）
    """
    try:
        file_path = os.path.join(UPLOAD_BASE_DIR, folder_path, filename)
        if os.path.exists(file_path):
            return send_file(file_path)
        else:
            return {'error': '文件不存在'}, 404
    except Exception as e:
        return {'error': str(e)}, 500

def _classify_skin_type(oiliness, moisture):
    """根据水油度分类皮肤类型"""
    if oiliness > 60:
        oil_sug = "当前出油面积较大,建议使用控油产品"
    elif oiliness <= 60 and oiliness > 40:
        oil_sug = "当前出油面积较大,建议使用控油产品"
    else:
        oil_sug = "当前面部控油较好,几乎没有太多出油"
    if moisture > 50:
        moi_sug = "当前皮肤状态很好,几乎不用保湿"
    else:
        moi_sug = "当前皮肤较为干燥,请做好保湿"
    return {"oil_sug": oil_sug, "moi_sug":moi_sug}

@bp.route('/face-analysis/health', methods=['GET'])
def health_check():
    """健康检查接口"""
    return {'status': 'healthy', 'service': 'face_analysis'}

@bp.route('/face-analysis/stats', methods=['GET'])
def get_stats():
    """获取服务统计信息"""
    with request_lock:
        return {
            'status': 'ok',
            'concurrent_control': {
                'max_concurrent_requests': MAX_CONCURRENT_REQUESTS,
                'active_requests': active_requests,
                'available_slots': MAX_CONCURRENT_REQUESTS - active_requests,
                'total_requests_processed': total_requests
            },
            'service_info': {
                'services_loaded': True,
                'parallel_tasks_per_request': 7
            }
        }
