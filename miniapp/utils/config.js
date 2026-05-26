const DEFAULT_LAN_HOST = '192.168.1.23';
const DEFAULT_PORT = 5000;
const DEFAULT_BASE_URL = `http://${DEFAULT_LAN_HOST}:${DEFAULT_PORT}`;

const API_PATHS = {
  UPLOAD_IMAGE: '/v1/mobile/upload-image',
  FACE_ALIGN: '/v1/face-align',
  ANALYZE_FACE: '/v1/analyze-face',
  TONGUE_SEGMENT: '/v1/tongue-segment',
  ANALYSIS_TASK_FACE: '/v1/analysis-tasks/face',
  ANALYSIS_TASK_FACE_ANALYSIS: '/v1/analysis-tasks/face-analysis',
  ANALYSIS_TASK_TONGUE: '/v1/analysis-tasks/tongue',
  CHAT_SESSION: '/v1/mobile/chat/session',
  CHAT_SESSIONS: '/v1/mobile/chat/sessions',
  CHAT_MESSAGE: '/v1/mobile/chat/message',
  CHAT_STREAM: '/v1/mobile/chat/stream',
  CHAT_ATTACHMENT: '/v1/mobile/chat/attachment',
  CHAT_DIAGNOSIS_CONTEXT: '/v1/mobile/chat/diagnosis-context',
};

function buildAnalysisTaskStatusPath(taskId) {
  return `/v1/analysis-tasks/${taskId}`;
}

function buildChatSessionDetailPath(sessionId) {
  return `/v1/mobile/chat/session/${sessionId}`;
}

function buildChatDiagnosisContextDetailPath(sessionId) {
  return `/v1/mobile/chat/diagnosis-context/${sessionId}`;
}

module.exports = {
  API_PATHS,
  buildAnalysisTaskStatusPath,
  buildChatDiagnosisContextDetailPath,
  buildChatSessionDetailPath,
  DEFAULT_BASE_URL,
  DEFAULT_LAN_HOST,
  DEFAULT_PORT,
};
