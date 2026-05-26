"""
模型管理器 - 单例模式
在服务启动时预加载所有模型，避免每次请求重复加载
"""

import os
import torch
import onnxruntime
from pathlib import Path
import threading

from api.services.mediapipe_compat import create_face_mesh


class ModelManager:
    """模型管理器单例类 - 预加载并缓存所有模型"""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """初始化模型管理器，只执行一次"""
        if self._initialized:
            return
            
        print("🚀 初始化模型管理器...")
        self.models_dir = Path(__file__).parent.parent / 'models'
        
        # 存储所有模型
        self.models = {}
        
        # 加载所有模型
        self._load_all_models()
        
        self._initialized = True
        print("✅ 模型管理器初始化完成")
    
    def _load_all_models(self):
        """预加载所有模型到内存"""
        
        # 1. 加载PyTorch模型 - 痣分类模型
        print("📦 加载 MobileNetV2 模型...")
        try:
            from api.services.nevus import MobileNetV2
            
            weights_path = self.models_dir / 'mobilenetV2.pth'
            if not weights_path.exists():
                print(f"⚠️  警告: 模型文件不存在 {weights_path}")
            else:
                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                model = MobileNetV2(num_classes=2)
                state_dict = torch.load(str(weights_path), map_location=device, weights_only=True)
                model.load_state_dict(state_dict)
                model = model.to(device)
                model.eval()  # 设置为评估模式
                
                self.models['nevus_classifier'] = {
                    'model': model,
                    'device': device,
                    'class_dict': {"0": "nevus", "1": "stain"}
                }
                print(f"   ✓ MobileNetV2 已加载 (设备: {device})")
        except Exception as e:
            print(f"   ✗ MobileNetV2 加载失败: {e}")
        
        # 2. 加载ONNX模型 - 法令纹检测模型
        print("📦 加载 ONNX 法令纹检测模型...")
        try:
            onnx_path_6 = self.models_dir / 'best_epoch_weights6.onnx'
            if not onnx_path_6.exists():
                print(f"⚠️  警告: 模型文件不存在 {onnx_path_6}")
            else:
                providers = ['CPUExecutionProvider']
                if torch.cuda.is_available():
                    providers.insert(0, 'CUDAExecutionProvider')
                
                sess = onnxruntime.InferenceSession(str(onnx_path_6), providers=providers)
                self.models['nasolabial_detector_6'] = {
                    'session': sess,
                    'input_name': sess.get_inputs()[0].name
                }
                print(f"   ✓ ONNX模型(weights6) 已加载")
        except Exception as e:
            print(f"   ✗ ONNX模型(weights6) 加载失败: {e}")
        
        # 3. 加载ONNX模型 - 另一个检测模型
        # print("📦 加载 ONNX 检测模型 (20251028)...")
        # try:
        #     onnx_path_20251028 = self.models_dir / 'best_epoch_weights20251028.onnx'
        #     if not onnx_path_20251028.exists():
        #         print(f"⚠️  警告: 模型文件不存在 {onnx_path_20251028}")
        #     else:
        #         providers = ['CPUExecutionProvider']
        #         if torch.cuda.is_available():
        #             providers.insert(0, 'CUDAExecutionProvider')
                
        #         sess = onnxruntime.InferenceSession(str(onnx_path_20251028), providers=providers)
        #         self.models['detector_20251028'] = {
        #             'session': sess,
        #             'input_name': sess.get_inputs()[0].name
        #         }
        #         print(f"   ✓ ONNX模型(20251028) 已加载")
        # except Exception as e:
        #     print(f"   ✗ ONNX模型(20251028) 加载失败: {e}")
        
        # 4. 初始化MediaPipe FaceMesh (共享实例)
        print("📦 初始化 MediaPipe FaceMesh...")
        try:
            face_mesh = create_face_mesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5
            )
            if face_mesh is None:
                print("   ⚠ MediaPipe FaceMesh 不可用，后续使用Haar兜底")
            else:
                self.models['mediapipe_face_mesh'] = face_mesh
                print("   ✓ MediaPipe FaceMesh 已初始化")
        except Exception as e:
            print(f"   ✗ MediaPipe FaceMesh 初始化失败: {e}")
        
        # 5. 加载Haar Cascade人脸检测器 (共享实例)
        print("📦 加载 Haar Cascade 人脸检测器...")
        try:
            import cv2
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            face_cascade = cv2.CascadeClassifier(cascade_path)
            if face_cascade.empty():
                print(f"   ✗ 无法加载人脸分类器: {cascade_path}")
            else:
                self.models['face_cascade'] = face_cascade
                print("   ✓ Haar Cascade 人脸检测器已加载")
        except Exception as e:
            print(f"   ✗ Haar Cascade 加载失败: {e}")
    
    def get_model(self, model_name):
        """获取指定模型"""
        return self.models.get(model_name)
    
    def get_nevus_classifier(self):
        """获取痣分类模型"""
        return self.models.get('nevus_classifier')
    
    def get_nasolabial_detector(self):
        """获取法令纹检测模型"""
        return self.models.get('nasolabial_detector_6')
    
    def get_detector_20251028(self):
        """获取20251028检测模型"""
        return self.models.get('detector_20251028')
    
    def get_mediapipe_face_mesh(self):
        """获取MediaPipe FaceMesh实例"""
        return self.models.get('mediapipe_face_mesh')
    
    def get_face_cascade(self):
        """获取Haar Cascade人脸检测器"""
        return self.models.get('face_cascade')
    
    def list_models(self):
        """列出所有已加载的模型"""
        return list(self.models.keys())


# 创建全局单例实例
_model_manager = None

def get_model_manager():
    """获取模型管理器单例"""
    global _model_manager
    if _model_manager is None:
        _model_manager = ModelManager()
    return _model_manager
