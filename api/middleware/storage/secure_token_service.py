"""
安全令牌服务类
基于JWT实现的访问令牌管理，支持过期时间和访问次数限制
"""

import os
import jwt
import time
import uuid
from datetime import datetime, timedelta


class SecureTokenService:
    """安全令牌服务类"""
    
    def __init__(self, config=None):
        self.config = config or self._get_default_config()
        self.secret = self.config['jwt_secret']
        self.default_expire_hours = self.config['token_expire_hours']
        self.default_max_access = self.config['max_access_count']
        self.access_records = {}  # 记录访问次数 {token: count}
        self.last_cleanup = time.time()
    
    def _get_default_config(self):
        """获取默认配置"""
        return {
            'jwt_secret': os.getenv('JWT_SECRET', 'your-super-secret-key-change-in-production'),
            'token_expire_hours': int(os.getenv('TOKEN_EXPIRE_HOURS', '24')),
            'max_access_count': int(os.getenv('MAX_ACCESS_COUNT', '100')),
            'cleanup_interval': int(os.getenv('CLEANUP_INTERVAL', '3600')),  # 清理间隔（秒）
        }
    
    def generate_token(self, task_uuid, filename, file_type='aligned', expire_hours=None, max_access=None, extra_data=None):
        """
        生成JWT访问令牌
        
        Args:
            task_uuid: 任务UUID
            filename: 文件名
            file_type: 文件类型 ('original', 'aligned')
            expire_hours: 过期时间（小时）
            max_access: 最大访问次数
            extra_data: 额外数据
            
        Returns:
            str: JWT令牌
        """
        expire_hours = expire_hours or self.default_expire_hours
        max_access = max_access or self.default_max_access
        
        current_time = int(time.time())
        jti = str(uuid.uuid4())  # JWT ID，用于唯一标识
        
        payload = {
            'task_uuid': task_uuid,
            'filename': filename,
            'file_type': file_type,
            'max_access': max_access,
            'created_at': current_time,
            'exp': current_time + (expire_hours * 3600),
            'iat': current_time,  # issued at
            'jti': jti
        }
        
        # 添加额外数据
        if extra_data and isinstance(extra_data, dict):
            payload.update(extra_data)
        
        try:
            token = jwt.encode(payload, self.secret, algorithm='HS256')
            self.access_records[token] = 0  # 初始化访问计数
            
            print(f"生成令牌成功: {jti}, 过期时间: {expire_hours}小时, 最大访问: {max_access}次")
            return token
        except Exception as e:
            print(f"生成令牌失败: {str(e)}")
            return None
    
    def validate_token(self, token):
        """
        验证JWT令牌
        
        Args:
            token: JWT令牌
            
        Returns:
            tuple: (success: bool, result: dict/str)
        """
        try:
            # 先进行定期清理
            self._auto_cleanup()
            
            # 解码JWT令牌
            payload = jwt.decode(token, self.secret, algorithms=['HS256'])
            
            # 检查访问次数
            current_access = self.access_records.get(token, 0)
            max_access = payload.get('max_access', self.default_max_access)
            
            if current_access >= max_access:
                return False, "访问次数已达上限"
            
            # 增加访问计数
            self.access_records[token] = current_access + 1
            
            result = {
                'task_uuid': payload['task_uuid'],
                'filename': payload['filename'],
                'file_type': payload['file_type'],
                'remaining_access': max_access - self.access_records[token],
                'expires_at': payload['exp'],
                'created_at': payload['created_at'],
                'jti': payload['jti'],
                'current_access': self.access_records[token]
            }
            
            # 添加其他payload中的数据
            for key, value in payload.items():
                if key not in ['exp', 'iat', 'jti'] and key not in result:
                    result[key] = value
            
            return True, result
            
        except jwt.ExpiredSignatureError:
            # 令牌过期时清理访问记录
            if token in self.access_records:
                del self.access_records[token]
            return False, "令牌已过期"
        except jwt.InvalidTokenError as e:
            return False, f"无效的令牌: {str(e)}"
        except Exception as e:
            return False, f"令牌验证失败: {str(e)}"
    
    def revoke_token(self, token):
        """
        撤销令牌（将访问次数设为最大值）
        
        Args:
            token: JWT令牌
            
        Returns:
            bool: 撤销是否成功
        """
        try:
            payload = jwt.decode(token, self.secret, algorithms=['HS256'])
            max_access = payload.get('max_access', self.default_max_access)
            self.access_records[token] = max_access
            print(f"令牌已撤销: {payload.get('jti', 'unknown')}")
            return True
        except jwt.InvalidTokenError:
            return False
    
    def get_token_info(self, token):
        """
        获取令牌信息（不增加访问计数）
        
        Args:
            token: JWT令牌
            
        Returns:
            dict: 令牌信息，失败返回None
        """
        try:
            payload = jwt.decode(token, self.secret, algorithms=['HS256'])
            current_access = self.access_records.get(token, 0)
            
            return {
                'task_uuid': payload['task_uuid'],
                'filename': payload['filename'],
                'file_type': payload['file_type'],
                'max_access': payload.get('max_access', self.default_max_access),
                'current_access': current_access,
                'remaining_access': payload.get('max_access', self.default_max_access) - current_access,
                'expires_at': payload['exp'],
                'created_at': payload['created_at'],
                'jti': payload['jti'],
                'is_expired': payload['exp'] < int(time.time()),
                'is_exhausted': current_access >= payload.get('max_access', self.default_max_access)
            }
        except jwt.InvalidTokenError:
            return None
    
    def cleanup_expired_records(self):
        """手动清理过期的访问记录"""
        current_time = int(time.time())
        expired_tokens = []
        
        for token in list(self.access_records.keys()):
            try:
                payload = jwt.decode(token, self.secret, algorithms=['HS256'])
                if payload['exp'] < current_time:
                    expired_tokens.append(token)
            except jwt.InvalidTokenError:
                expired_tokens.append(token)
        
        # 删除过期记录
        for token in expired_tokens:
            if token in self.access_records:
                del self.access_records[token]
        
        self.last_cleanup = time.time()
        
        if expired_tokens:
            print(f"清理了 {len(expired_tokens)} 个过期令牌记录")
        
        return len(expired_tokens)
    
    def _auto_cleanup(self):
        """自动清理过期记录"""
        current_time = time.time()
        cleanup_interval = self.config.get('cleanup_interval', 3600)
        
        if current_time - self.last_cleanup > cleanup_interval:
            self.cleanup_expired_records()
    
    def get_statistics(self):
        """
        获取令牌使用统计
        
        Returns:
            dict: 统计信息
        """
        current_time = int(time.time())
        total_tokens = len(self.access_records)
        active_tokens = 0
        expired_tokens = 0
        exhausted_tokens = 0
        
        for token in self.access_records:
            try:
                payload = jwt.decode(token, self.secret, algorithms=['HS256'])
                if payload['exp'] < current_time:
                    expired_tokens += 1
                elif self.access_records[token] >= payload.get('max_access', self.default_max_access):
                    exhausted_tokens += 1
                else:
                    active_tokens += 1
            except jwt.InvalidTokenError:
                expired_tokens += 1
        
        return {
            'total_tokens': total_tokens,
            'active_tokens': active_tokens,
            'expired_tokens': expired_tokens,
            'exhausted_tokens': exhausted_tokens,
            'last_cleanup': datetime.fromtimestamp(self.last_cleanup).isoformat()
        }