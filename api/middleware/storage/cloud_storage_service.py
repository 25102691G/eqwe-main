"""
云存储服务类
支持 AWS S3、阿里云OSS 等对象存储服务
"""

import os

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

from api.configs.storage_config import get_storage_config


class CloudStorageService:
    """云存储服务类"""
    
    def __init__(self, config=None):
        self.config = config or self._get_default_config()
        self.s3_client = None
        self.bucket_name = self.config['bucket_name']
        self.cdn_domain = self.config.get('cdn_domain', '')
        self._init_s3_client()
    
    def _get_default_config(self):
        """获取默认配置"""
        return get_storage_config()
    
    def _init_s3_client(self):
        """初始化S3客户端"""
        try:
            # 获取端点URL
            endpoint_url = self.config.get('endpoint_url', '').strip()
            
            # 构建客户端参数
            client_params = {
                'aws_access_key_id': self.config['aws_access_key_id'],
                'aws_secret_access_key': self.config['aws_secret_access_key'],
                'region_name': self.config['region']
            }
            
            # 如果有自定义端点（如MinIO），添加端点URL和额外配置
            if endpoint_url:
                client_params['endpoint_url'] = endpoint_url
                # 对于 MinIO，使用路径样式访问
                client_params['config'] = Config(
                    signature_version='s3v4',
                    s3={'addressing_style': 'path'},
                    connect_timeout=2,
                    read_timeout=2,
                    retries={'max_attempts': 1},
                )
            
            self.s3_client = boto3.client('s3', **client_params)
            
            # 尝试创建存储桶（如果不存在）
            self._ensure_bucket_exists()
            
        except Exception as e:
            print(f"S3客户端初始化失败: {str(e)}")
            self.s3_client = None
    
    def _ensure_bucket_exists(self):
        """确保存储桶存在，如果不存在则创建"""
        if not self.s3_client:
            return
        
        try:
            # 先尝试列出存储桶来验证连接
            response = self.s3_client.list_buckets()
            bucket_names = [bucket['Name'] for bucket in response['Buckets']]
            
            if self.bucket_name in bucket_names:
                return
            
            # 存储桶不存在，尝试创建
            self.s3_client.create_bucket(Bucket=self.bucket_name)
            
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code != 'BucketAlreadyExists':
                print(f"存储桶操作失败 ({error_code}): {str(e)}")
    
    def upload_file(self, local_file_path, object_key, content_type='image/jpeg'):
        """
        上传文件到云存储
        
        Args:
            local_file_path: 本地文件路径
            object_key: 对象存储中的键名
            content_type: 文件MIME类型
            
        Returns:
            tuple: (success: bool, result: str)
        """
        if not self.s3_client:
            return False, "S3客户端未初始化"
        
        try:
            self.s3_client.upload_file(
                local_file_path,
                self.bucket_name,
                object_key,
                ExtraArgs={
                    'ContentType': content_type,
                    'CacheControl': 'max-age=86400',  # 缓存1天
                    'ACL': 'private'  # 私有访问
                }
            )

            return True, object_key
        except ClientError as e:
            error_msg = f"上传失败: {str(e)}"
            return False, error_msg
    
    def upload_from_memory(self, file_data, object_key, content_type='image/jpeg'):
        """
        从内存直接上传文件到云存储（无需保存到本地）
        
        Args:
            file_data: 文件的字节数据（bytes）
            object_key: 对象存储中的键名
            content_type: 文件MIME类型
            
        Returns:
            tuple: (success: bool, result: str)
        """
        if not self.s3_client:
            return False, "S3客户端未初始化"
        
        try:
            # 使用 put_object 直接上传字节数据
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=object_key,
                Body=file_data,
                ContentType=content_type,
                CacheControl='max-age=86400',  # 缓存1天
                ACL='private'  # 私有访问
            )
            
            return True, object_key
        except ClientError as e:
            error_msg = f"上传失败: {str(e)}"
            return False, error_msg
    
    def generate_presigned_url(self, object_key, expiration=3600):
        """
        生成预签名URL
        
        Args:
            object_key: 对象存储中的键名
            expiration: 过期时间（秒）
            
        Returns:
            str: 预签名URL，失败返回None
        """
        if not self.s3_client:
            return None
        
        try:
            url = self.s3_client.generate_presigned_url(
                'get_object',
                Params={'Bucket': self.bucket_name, 'Key': object_key},
                ExpiresIn=expiration
            )
            
            # 如果配置了CDN域名，替换为CDN地址
            if self.cdn_domain:
                url = url.replace(
                    f"https://{self.bucket_name}.s3.{self.config['region']}.amazonaws.com",
                    f"https://{self.cdn_domain}"
                )
            
            return url
        except ClientError as e:
            return None
    
    def download_file(self, object_key, local_file_path):
        """
        从云存储下载文件到本地
        
        Args:
            object_key: 对象存储中的键名
            local_file_path: 本地文件保存路径
            
        Returns:
            tuple: (success: bool, message: str)
        """
        if not self.s3_client:
            return False, "S3客户端未初始化"
        
        try:
            # 确保本地目录存在
            os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
            
            # 下载文件
            self.s3_client.download_file(self.bucket_name, object_key, local_file_path)
            return True, "下载成功"
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == 'NoSuchKey':
                return False, f"文件不存在: {object_key}"
            elif error_code == 'NoSuchBucket':
                return False, f"存储桶不存在: {self.bucket_name}"
            else:
                return False, f"下载失败: {str(e)}"
        except Exception as e:
            return False, f"下载过程中发生错误: {str(e)}"
    
    def download_to_memory(self, object_key):
        """
        从云存储直接下载文件到内存（返回字节流）
        
        Args:
            object_key: 对象存储中的键名
            
        Returns:
            tuple: (success: bool, data: bytes or error_message: str)
        """
        if not self.s3_client:
            return False, "S3客户端未初始化"
        
        try:
            # 使用 BytesIO 在内存中接收数据
            from io import BytesIO
            buffer = BytesIO()
            
            # 下载到内存
            self.s3_client.download_fileobj(self.bucket_name, object_key, buffer)
            
            # 获取字节数据
            buffer.seek(0)
            file_bytes = buffer.read()
            
            return True, file_bytes
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == 'NoSuchKey':
                return False, f"文件不存在: {object_key}"
            elif error_code == 'NoSuchBucket':
                return False, f"存储桶不存在: {self.bucket_name}"
            else:
                return False, f"下载失败: {str(e)}"
        except Exception as e:
            return False, f"下载过程中发生错误: {str(e)}"

    def delete_file(self, object_key):
        """
        删除云存储中的文件
        
        Args:
            object_key: 对象存储中的键名
            
        Returns:
            bool: 删除是否成功
        """
        if not self.s3_client:
            return False
        
        try:
            self.s3_client.delete_object(Bucket=self.bucket_name, Key=object_key)

            return True
        except ClientError as e:
            return False
    
    def list_files(self, prefix='', max_keys=1000):
        """
        列出存储桶中的文件
        
        Args:
            prefix: 文件前缀过滤
            max_keys: 最大返回数量
            
        Returns:
            list: 文件列表
        """
        if not self.s3_client:
            return []
        
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=prefix,
                MaxKeys=max_keys
            )
            
            files = []
            if 'Contents' in response:
                for obj in response['Contents']:
                    files.append({
                        'key': obj['Key'],
                        'size': obj['Size'],
                        'last_modified': obj['LastModified'].isoformat(),
                        'etag': obj['ETag'].strip('"')
                    })
            
            return files
        except ClientError as e:
            return []
    
    def get_file_info(self, object_key):
        """
        获取文件信息
        
        Args:
            object_key: 对象存储中的键名
            
        Returns:
            dict: 文件信息，失败返回None
        """
        if not self.s3_client:
            return None
        
        try:
            response = self.s3_client.head_object(Bucket=self.bucket_name, Key=object_key)
            return {
                'size': response['ContentLength'],
                'content_type': response.get('ContentType', ''),
                'last_modified': response['LastModified'].isoformat(),
                'etag': response['ETag'].strip('"'),
                'cache_control': response.get('CacheControl', ''),
            }
        except ClientError as e:
            return None
