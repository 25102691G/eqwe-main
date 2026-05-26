# 中医健康助手后端与小程序

本项目提供微信小程序端的舌象一期分析、面象肤况辅助分析、AI 问诊、历史报告和问询记录能力。当前分析结果定位为“皮肤状态/舌象特征辅助分析与护理调养建议”，不是医学诊断或治疗结论。

## 技术栈

- 后端：Python、Flask、Flask-CORS
- 队列：Celery、Redis
- 存储：MinIO/S3 兼容对象存储
- 会话持久化：PostgreSQL，文件 JSON 作为降级方案
- 图像算法：OpenCV、NumPy、ONNX 舌体分割、面部对齐与规则/模型混合肤况分析
- 大模型：OpenAI Responses API 兼容接口
- 前端：微信小程序原生 WXML/WXSS/JS
- 本地环境：Conda 环境 `tongue`

## 目录结构

```text
skin_alporithm/
  app.py                         # Flask 服务入口
  .env                           # 本地环境变量
  api/
    controllers/v1/              # HTTP API 路由
    queue/celery_app.py          # Celery app
    tasks/analysis_tasks.py      # 舌象/面象队列任务
    services/                    # 算法、报告、聊天、存储服务
    middleware/storage/          # MinIO/S3 存储封装
  docker/
    docker-compose.middleware.yaml # MinIO/Postgres/Redis
  miniapp/                       # 微信小程序
  upload_files/                  # 本地调试输出与临时文件
```

## 环境变量

主要配置在 `skin_alporithm/.env`：

```env
APP_HOST=0.0.0.0
APP_PORT=5000

S3_ENDPOINT_URL=http://127.0.0.1:9000
S3_BUCKET_NAME=aitest
S3_ACCESS_KEY=minioadmin
S3_SECRET_KEY=minioadmin

CHAT_STORAGE_BACKEND=postgres
CHAT_DATABASE_URL=postgresql://chatapp:chatapp@127.0.0.1:15432/skin_chat

REDIS_URL=redis://:chatredis@127.0.0.1:16379/0

OPENAI_API_KEY=你的 key
OPENAI_BASE_URL=https://你的 OpenAI 兼容地址/v1
SKIN_REPORT_MODEL=openai:gpt-5.4
```

注意不要把真实 `OPENAI_API_KEY` 提交到公开仓库。

## 启动命令

### 1. 启动中间件

```powershell
cd D:\aproject\shemian\skin_alporithm
docker compose -f docker\docker-compose.middleware.yaml up -d
```

默认端口：

- MinIO API: `9000`
- MinIO Console: `9001`
- PostgreSQL: `15432`
- Redis: `16379`

### 2. 启动 Flask 后端

```powershell
cd D:\aproject\shemian\skin_alporithm
conda activate tongue
python app.py
```

默认监听：

```text
http://0.0.0.0:5000
```

本机访问：

```text
http://127.0.0.1:5000/v1/
```

局域网同事访问时使用你的电脑 IP：

```text
http://你的电脑IP:5000/v1/
```

### 3. 启动 Celery 队列

另开一个终端：

```powershell
cd D:\aproject\shemian\skin_alporithm
conda activate tongue
celery -A api.queue.celery_app.celery_app worker --pool=solo --loglevel INFO -Q face_analysis,tongue_analysis
```

Windows 下建议使用 `--pool=solo`。

### 4. 小程序局域网测试

1. 微信开发者工具导入 `skin_alporithm/miniapp`。
2. `详情 -> 本地设置` 勾选“不校验合法域名、web-view、TLS 版本以及 HTTPS 证书”。
3. 小程序 `我的 -> 后端地址` 填写：

```text
http://你的电脑IP:5000
```

4. 点击“测试连接”。
5. 手机和后端电脑需要在同一局域网。

如果手机无法访问后端，检查 Windows 防火墙：

```powershell
New-NetFirewallRule -DisplayName "TCM Miniapp Backend 5000" -Direction Inbound -Action Allow -Protocol TCP -LocalPort 5000 -Profile Private
```

## 核心调用流程

### 舌象/面象队列分析流程

```text
小程序上传图片
  -> POST /v1/mobile/upload-image
  -> 返回 file_path + file_name
  -> POST /v1/analysis-tasks/tongue 或 /v1/analysis-tasks/face
  -> 返回 task_id
  -> GET /v1/analysis-tasks/<task_id> 轮询
  -> SUCCESS 后读取 result.analysis_result
  -> 小程序展示报告并可导入 AI 问诊
```

### AI 问诊流程

```text
创建/恢复会话
  -> POST /v1/mobile/chat/session 或 GET /v1/mobile/chat/session/<id>
导入报告上下文
  -> POST /v1/mobile/chat/diagnosis-context
发送消息
  -> POST /v1/mobile/chat/stream
问询记录
  -> GET /v1/mobile/chat/sessions
```

## HTTP 接口

所有业务接口前缀为 `/v1`。

### API 信息

#### `GET /v1/`

返回当前服务可用接口列表。

响应示例：

```json
{
  "welcome": "SKIN_DET OpenAPI",
  "api_version": "V1",
  "endpoints": ["GET /v1/ - API info"]
}
```

## 移动端图片与资源

### 上传图片

#### `POST /v1/mobile/upload-image`

请求类型：`multipart/form-data`

字段：

| 字段 | 必填 | 说明 |
| --- | --- | --- |
| file | 是 | 图片文件 |
| folder | 否 | 指定对象存储目录，不传则自动生成 |

响应示例：

```json
{
  "status": "success",
  "bucket": "aitest",
  "folder": "miniapp-xxx",
  "file_name": "original.jpg",
  "object_key": "miniapp-xxx/original.jpg",
  "file_path": "http://127.0.0.1:9001/browser/aitest/miniapp-xxx",
  "preview_url": "http://127.0.0.1:5000/v1/mobile/result-image/miniapp-xxx/original.jpg"
}
```

### 读取结果图片

#### `GET /v1/mobile/result-image/<folder_path>/<filename>`

从 MinIO 代理图片/JSON 资源给小程序。小程序只需要访问 Flask，不直接访问 MinIO。

示例：

```text
GET /v1/mobile/result-image/miniapp-xxx/tongue_segmented.jpg
```

## 队列任务接口

### 提交完整面象流程

#### `POST /v1/analysis-tasks/face`

用途：队列执行面部对齐 + 面象肤况分析。

输入：

```json
{
  "file_path": "http://127.0.0.1:9001/browser/aitest/miniapp-xxx",
  "file_name": "original.jpg"
}
```

输出：

```json
{
  "status": "queued",
  "analysis_type": "face",
  "task_id": "celery-task-id",
  "state": "PENDING",
  "status_url": "http://127.0.0.1:5000/v1/analysis-tasks/celery-task-id"
}
```

### 提交面象分析

#### `POST /v1/analysis-tasks/face-analysis`

用途：对已上传或已对齐的人脸图片做肤况分析。

输入同 `/v1/analysis-tasks/face`。

### 提交舌象一期分析

#### `POST /v1/analysis-tasks/tongue`

用途：队列执行完整舌象一期分析。

输入：

```json
{
  "file_path": "http://127.0.0.1:9001/browser/aitest/miniapp-xxx",
  "file_name": "original.jpg",
  "include_visualizations": true,
  "upload_visualizations": true
}
```

输出同队列提交响应，`analysis_type` 为 `tongue`。

### 查询任务状态

#### `GET /v1/analysis-tasks/<task_id>`

处理中：

```json
{
  "task_id": "celery-task-id",
  "state": "STARTED",
  "ready": false,
  "successful": false
}
```

成功：

```json
{
  "task_id": "celery-task-id",
  "state": "SUCCESS",
  "ready": true,
  "successful": true,
  "result": {
    "analysis_result": {},
    "align_result": {}
  }
}
```

失败：

```json
{
  "task_id": "celery-task-id",
  "state": "FAILURE",
  "ready": true,
  "successful": false,
  "error": "错误信息"
}
```

## 舌象接口

### 舌象质量检查

#### `POST /v1/tongue-quality-check`

输入：

```json
{
  "file_path": "http://127.0.0.1:9001/browser/aitest/miniapp-xxx",
  "file_name": "original.jpg"
}
```

输出关键字段：

```json
{
  "status": "success",
  "task_uuid": "miniapp-xxx",
  "results": {
    "quality_control": {},
    "image_size": {
      "width": 1080,
      "height": 1440
    },
    "bounding_box": {
      "x": 100,
      "y": 200,
      "w": 300,
      "h": 400
    },
    "segmentation_used": true
  },
  "storage": {
    "type": "cloud",
    "bucket": "aitest",
    "folder": "miniapp-xxx"
  }
}
```

### 舌象一期完整分析

#### `POST /v1/tongue-segment`

输入：

```json
{
  "file_path": "http://127.0.0.1:9001/browser/aitest/miniapp-xxx",
  "file_name": "original.jpg",
  "include_images": false,
  "include_visualizations": true,
  "upload_visualizations": true
}
```

输出关键字段：

```json
{
  "status": "success",
  "file_name": "original.jpg",
  "image_size": {
    "width": 1080,
    "height": 1440
  },
  "bounding_box": {
    "x": 399,
    "y": 939,
    "w": 209,
    "h": 264
  },
  "segmentation_quality": {},
  "tongue_color": {},
  "region_colors": {},
  "tongue_coat": {},
  "region_coat": {},
  "region_moisture": {},
  "tongue_moisture": {},
  "crack_observation": {},
  "tongue_image_assistance": {},
  "demo_report": {},
  "storage": {
    "type": "cloud",
    "bucket": "aitest",
    "folder": "miniapp-xxx",
    "analysis_results_object_key": "miniapp-xxx/tongue_analysis_result.json"
  },
  "uploaded_files": [
    {
      "filename": "tongue_segmented.jpg",
      "object_key": "miniapp-xxx/tongue_segmented.jpg"
    }
  ]
}
```

## 面象接口

### 人脸对齐

#### `POST /v1/face-align`

输入：

```json
{
  "file_path": "http://127.0.0.1:9001/browser/aitest/miniapp-xxx",
  "file_name": "original.jpg",
  "margin": 0.3
}
```

输出关键字段：

```json
{
  "status": "success",
  "uuid": "original",
  "message": "处理完成",
  "bucket": "aitest",
  "folder": "miniapp-xxx",
  "file_name": "original.jpg",
  "aligned_file_name": "align.jpg",
  "aligned_object_key": "miniapp-xxx/align.jpg",
  "original_image_url": "/v1/mobile/result-image/miniapp-xxx/original.jpg",
  "aligned_image_url": "/v1/mobile/result-image/miniapp-xxx/align.jpg",
  "storage": {
    "type": "cloud",
    "bucket": "aitest",
    "folder": "miniapp-xxx",
    "object_key": "miniapp-xxx/align.jpg"
  }
}
```

### 面象肤况分析

#### `POST /v1/analyze-face`

输入：

```json
{
  "file_path": "http://127.0.0.1:9001/browser/aitest/miniapp-xxx",
  "file_name": "align.jpg"
}
```

输出关键字段：

```json
{
  "status": "success",
  "task_uuid": "miniapp-xxx",
  "analysis_type": "face",
  "analysis_results": {
    "oil_moi": {},
    "skin_color": {},
    "sensitivity": {},
    "smoothness": {},
    "wrinkles": {}
  },
  "llm_report": {
    "total_score": 80,
    "overall_summary": "肤况摘要",
    "metric_reports": {}
  },
  "storage": {
    "type": "cloud",
    "bucket": "aitest",
    "folder": "miniapp-xxx",
    "analysis_results_object_key": "miniapp-xxx/analysis_results.json"
  },
  "uploaded_files": []
}
```

## AI 问诊接口

### 创建或恢复会话

#### `POST /v1/mobile/chat/session`

输入：

```json
{
  "session_id": ""
}
```

`session_id` 为空时创建新会话；不为空时恢复或创建指定会话。

输出：

```json
{
  "status": "success",
  "session_id": "session-id",
  "session": {
    "session_id": "session-id",
    "title": "",
    "pinned": false,
    "messages": [],
    "diagnosis_context": null
  }
}
```

### 获取会话详情

#### `GET /v1/mobile/chat/session/<session_id>`

输出：

```json
{
  "status": "success",
  "session_id": "session-id",
  "session": {
    "messages": [],
    "diagnosis_context": {}
  }
}
```

### 更新会话元信息

#### `PATCH /v1/mobile/chat/session/<session_id>`

输入：

```json
{
  "title": "问询标题",
  "pinned": true
}
```

当前小程序只使用 `pinned`，重命名接口保留但入口暂时隐藏。

### 删除会话

#### `DELETE /v1/mobile/chat/session/<session_id>`

输出：

```json
{
  "status": "success",
  "session_id": "session-id"
}
```

### 会话列表

#### `GET /v1/mobile/chat/sessions?limit=20`

输出：

```json
{
  "status": "success",
  "count": 1,
  "sessions": [
    {
      "session_id": "session-id",
      "title": "",
      "pinned": false,
      "message_count": 4,
      "last_message_preview": "最近消息摘要",
      "has_diagnosis_context": true,
      "diagnosis_summary": "已导入的报告摘要"
    }
  ]
}
```

### 上传聊天附件

#### `POST /v1/mobile/chat/attachment`

请求类型：`multipart/form-data`

字段：

| 字段 | 必填 | 说明 |
| --- | --- | --- |
| file | 是 | 图片、txt、pdf 等附件 |
| session_id | 是 | 会话 ID |

输出：

```json
{
  "status": "success",
  "session_id": "session-id",
  "attachment": {
    "attachment_id": "attachment-id",
    "name": "example.jpg",
    "kind": "image",
    "object_key": "chat_attachments/session-id/xxx.jpg",
    "download_url": "/v1/mobile/chat/attachment/session-id/attachment-id"
  },
  "session": {}
}
```

### 普通聊天

#### `POST /v1/mobile/chat/message`

输入：

```json
{
  "session_id": "session-id",
  "text": "请解释我的报告",
  "attachment_ids": [],
  "client_message_id": "turn-xxx"
}
```

输出：

```json
{
  "status": "success",
  "session_id": "session-id",
  "user_message": {},
  "assistant_message": {},
  "session": {}
}
```

### 流式聊天

#### `POST /v1/mobile/chat/stream`

输入同 `/v1/mobile/chat/message`。

响应类型：`application/x-ndjson`

事件示例：

```json
{"type":"start","session_id":"session-id","user_message":{}}
{"type":"delta","delta":"你好"}
{"type":"delta","delta":"，我可以帮你解释报告。"}
{"type":"done","assistant_message":{},"session":{},"session_id":"session-id"}
```

### 导入报告上下文

#### `POST /v1/mobile/chat/diagnosis-context`

输入单报告：

```json
{
  "session_id": "session-id",
  "analysis_result": {}
}
```

输入多报告：

```json
{
  "session_id": "session-id",
  "analysis_results": [
    {},
    {}
  ]
}
```

小程序当前会导入“最新舌象 + 最新面象”各一份；只有一类报告时只导入该类，不报错。

输出：

```json
{
  "status": "success",
  "session_id": "session-id",
  "diagnosis_context": {
    "sourceType": "combined",
    "sourceLabel": "综合辅助分析",
    "summary": "舌象一期: ...；面象肤况: ..."
  },
  "session": {}
}
```

### 移除报告上下文

#### `DELETE /v1/mobile/chat/diagnosis-context/<session_id>`

输出：

```json
{
  "status": "success",
  "session_id": "session-id",
  "session": {}
}
```

## 小程序页面能力

- 首页：核心任务入口、最近报告摘要
- 健康方案：根据最近报告生成静态方案视图
- AI 问诊：聊天、导入报告上下文、恢复历史会话
- 商城：占位商城页
- 我的：后端地址配置、历史报告、问询记录
- 历史报告：本地保存的舌象/面象历史列表，点击进入详情
- 问询记录：会话列表、继续聊天、置顶/取消置顶、删除

## 开发检查

小程序静态检查：

```powershell
node .\skin_alporithm\miniapp\tools\check_static.js
```

Python 语法检查：

```powershell
cd D:\aproject\shemian
conda run -n tongue python -m compileall skin_alporithm\api
```

## 常见问题

### 小程序访问后端失败

- 确认手机和电脑在同一局域网。
- 小程序后端地址必须是电脑局域网 IP，例如 `http://192.168.1.23:5000`。
- 不要填 `127.0.0.1`，那代表手机自己。
- 微信开发者工具本地测试需勾选“不校验合法域名”。
- Windows 防火墙需放行 5000 端口。

### 队列任务一直 PENDING

- 确认 Redis 已启动。
- 确认 Celery worker 已启动。
- 确认 worker 队列包含 `face_analysis,tongue_analysis`。

### MinIO 上传失败

- 确认 MinIO 容器启动。
- 确认 `.env` 里的 `S3_ENDPOINT_URL`、账号、密码、bucket 与 docker compose 一致。
- 默认 bucket 是 `aitest`。

### AI 问诊无回复

- 确认 `.env` 中 `OPENAI_API_KEY` 和 `OPENAI_BASE_URL` 可用。
- 后端使用 OpenAI Responses API 兼容调用。
- 如果大模型不可用，后端会返回降级提示。

## 安全与产品说明

- 面象和舌象部分结果包含规则阈值和辅助模型，不是医学级诊断。
- 页面文案应使用“辅助分析”“护理建议”“调养建议”，避免使用“诊断结论”。
- 体验版或正式版小程序需要 HTTPS 域名，并在微信公众平台配置合法 request 域名。
