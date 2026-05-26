const app = getApp();
const {
  API_PATHS,
  DEFAULT_BASE_URL,
  buildAnalysisTaskStatusPath,
} = require('../../utils/config');
const { extractGallery } = require('../../utils/analysis');
const { normalizeError, request, uploadImage } = require('../../utils/request');

const MODE_CONFIG = {
  face: {
    label: '面象',
    title: '面象肤况辅助分析',
    subtitle: '上传正面照片，系统会完成面部对齐、肤况指标分析和护理建议整理。',
    emptyText: '请选择一张正脸照片开始面象辅助分析。',
    selectedText: '照片已选择，可以提交到面象分析队列。',
    actionButtonText: '提交面象分析',
    folderPrefix: 'miniapp-face',
    taskPath: API_PATHS.ANALYSIS_TASK_FACE,
    tips: ['正脸入镜，额头、脸颊、下巴尽量完整', '光线均匀，避免强美颜、遮挡和逆光', '结果用于皮肤状态辅助分析和护理建议'],
  },
  tongue: {
    label: '舌象',
    title: '舌象一期辅助分析',
    subtitle: '上传舌面清晰照片，系统会生成舌色、舌苔、津润、裂纹候选和体质倾向参考。',
    emptyText: '请选择一张舌面照片开始舌象一期辅助分析。',
    selectedText: '照片已选择，可以提交到舌象分析队列。',
    actionButtonText: '提交舌象分析',
    folderPrefix: 'miniapp-tongue',
    taskPath: API_PATHS.ANALYSIS_TASK_TONGUE,
    tips: ['舌体自然伸出，尽量完整入镜', '避免彩色灯光、过曝、强滤镜和剧烈阴影', '舌象结果仅作体质倾向辅助参考'],
  },
};

const STEP_TEMPLATE = [
  { key: 'upload', label: '上传图片', status: 'pending' },
  { key: 'queue', label: '进入队列', status: 'pending' },
  { key: 'process', label: '后台分析', status: 'pending' },
  { key: 'render', label: '整理结果', status: 'pending' },
];

function cloneSteps() {
  return STEP_TEMPLATE.map((item) => ({ ...item }));
}

function formatFileSize(bytes) {
  if (!bytes) {
    return '未知大小';
  }
  if (bytes < 1024 * 1024) {
    return `${Math.round(bytes / 102.4) / 10} KB`;
  }
  return `${Math.round(bytes / (1024 * 102.4)) / 10} MB`;
}

function createFolderName(mode) {
  const config = MODE_CONFIG[mode] || MODE_CONFIG.face;
  return `${config.folderPrefix}-${Date.now()}-${Math.random().toString(16).slice(2, 8)}`;
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function taskErrorText(payload) {
  if (!payload) {
    return '队列任务执行失败。';
  }
  if (typeof payload === 'string') {
    return payload;
  }
  return payload.error || payload.message || payload.traceback || '队列任务执行失败。';
}

function resultImageCount(analysisResult) {
  return extractGallery(analysisResult).length;
}

Page({
  data: {
    baseUrl: DEFAULT_BASE_URL,
    activeMode: 'face',
    modeLabel: MODE_CONFIG.face.label,
    modeTitle: MODE_CONFIG.face.title,
    modeSubtitle: MODE_CONFIG.face.subtitle,
    photoTips: MODE_CONFIG.face.tips,
    selectedImagePath: '',
    selectedImageName: '',
    selectedImageSize: '',
    actionButtonText: MODE_CONFIG.face.actionButtonText,
    loading: false,
    currentStepText: MODE_CONFIG.face.emptyText,
    steps: cloneSteps(),
    errorMessage: '',
    workflowDone: false,
    latestFolderLabel: '--',
    latestResultCount: 0,
    taskId: '',
    taskStateDisplay: '--',
  },

  onLoad(options = {}) {
    const initialMode = options.mode === 'tongue' ? 'tongue' : 'face';
    this.setData({
      baseUrl: app.globalData.baseUrl || DEFAULT_BASE_URL,
    });
    this.applyMode(initialMode);
  },

  applyMode(mode) {
    const normalizedMode = mode === 'tongue' ? 'tongue' : 'face';
    const config = MODE_CONFIG[normalizedMode];
    this.setData({
      activeMode: normalizedMode,
      modeLabel: config.label,
      modeTitle: config.title,
      modeSubtitle: config.subtitle,
      photoTips: config.tips,
      actionButtonText: config.actionButtonText,
      currentStepText: this.data.selectedImagePath ? config.selectedText : config.emptyText,
      steps: cloneSteps(),
      errorMessage: '',
      workflowDone: false,
      latestFolderLabel: '--',
      latestResultCount: 0,
      taskId: '',
      taskStateDisplay: '--',
    });
  },

  switchMode(event) {
    this.applyMode(event.currentTarget.dataset.mode);
  },

  chooseImage() {
    wx.chooseMedia({
      count: 1,
      mediaType: ['image'],
      sourceType: ['album', 'camera'],
      sizeType: ['compressed'],
      success: (response) => {
        const file = response.tempFiles[0];
        const pathSegments = file.tempFilePath.split('/');
        const config = MODE_CONFIG[this.data.activeMode] || MODE_CONFIG.face;
        this.setData({
          selectedImagePath: file.tempFilePath,
          selectedImageName: pathSegments[pathSegments.length - 1],
          selectedImageSize: formatFileSize(file.size),
          errorMessage: '',
          workflowDone: false,
          steps: cloneSteps(),
          actionButtonText: config.actionButtonText,
          currentStepText: config.selectedText,
          latestFolderLabel: '--',
          latestResultCount: 0,
          taskId: '',
          taskStateDisplay: '--',
        });
      },
      fail: (error) => {
        this.setData({
          errorMessage: normalizeError(error, '图片选择失败。'),
        });
      },
    });
  },

  previewSelectedImage() {
    if (!this.data.selectedImagePath) {
      return;
    }

    wx.previewImage({
      current: this.data.selectedImagePath,
      urls: [this.data.selectedImagePath],
    });
  },

  updateStep(stepKey, status) {
    const steps = this.data.steps.map((item) =>
      item.key === stepKey ? { ...item, status } : item
    );
    this.setData({ steps });
  },

  async pollTask(taskId) {
    const baseUrl = this.data.baseUrl || DEFAULT_BASE_URL;
    for (let attempt = 0; attempt < 90; attempt += 1) {
      const statusResult = await request({
        baseUrl,
        path: buildAnalysisTaskStatusPath(taskId),
        method: 'GET',
      });
      const state = statusResult.state || '';
      this.setData({
        taskStateDisplay: state || (statusResult.ready ? 'READY' : 'PENDING'),
      });

      if (state === 'SUCCESS' || (statusResult.ready && statusResult.result)) {
        return statusResult;
      }

      if (state === 'FAILURE') {
        throw new Error(taskErrorText(statusResult));
      }

      this.setData({
        currentStepText: '后台队列正在处理，请稍候...',
      });
      await sleep(2000);
    }

    throw new Error('队列任务等待超时，请稍后在任务状态接口中继续查询。');
  },

  buildTaskPayload(uploadResult) {
    const payload = {
      file_path: uploadResult.file_path,
      file_name: uploadResult.file_name,
    };
    if (this.data.activeMode === 'tongue') {
      payload.include_visualizations = true;
      payload.upload_visualizations = true;
    }
    return payload;
  },

  async startAnalysis() {
    if (!this.data.selectedImagePath) {
      wx.showToast({
        title: `请先选择${this.data.modeLabel}照片`,
        icon: 'none',
      });
      return;
    }

    const mode = this.data.activeMode;
    const config = MODE_CONFIG[mode] || MODE_CONFIG.face;
    const baseUrl = this.data.baseUrl || DEFAULT_BASE_URL;
    app.setBaseUrl(baseUrl);
    app.clearWorkflowResult();

    this.setData({
      loading: true,
      actionButtonText: '队列处理中...',
      workflowDone: false,
      errorMessage: '',
      latestFolderLabel: '--',
      latestResultCount: 0,
      steps: cloneSteps(),
      currentStepText: '正在上传图片到后端服务...',
      taskId: '',
      taskStateDisplay: '--',
    });

    const folder = createFolderName(mode);

    try {
      this.updateStep('upload', 'active');
      const uploadResult = await uploadImage({
        baseUrl,
        path: API_PATHS.UPLOAD_IMAGE,
        filePath: this.data.selectedImagePath,
        folder,
      });
      this.updateStep('upload', 'done');

      this.updateStep('queue', 'active');
      this.setData({ currentStepText: '正在提交到后台队列...' });
      const taskSubmitResult = await request({
        baseUrl,
        path: config.taskPath,
        method: 'POST',
        data: this.buildTaskPayload(uploadResult),
      });
      this.updateStep('queue', 'done');

      this.setData({
        taskId: taskSubmitResult.task_id || '',
        taskStateDisplay: taskSubmitResult.state || 'PENDING',
        currentStepText: '正在等待队列执行...',
      });
      this.updateStep('process', 'active');
      const taskStatusResult = await this.pollTask(taskSubmitResult.task_id);
      const taskResult = taskStatusResult.result || {};
      const analyzeResult = taskResult.analysis_result || taskResult;
      const alignResult = taskResult.align_result || {};
      this.updateStep('process', 'done');

      this.updateStep('render', 'active');
      const workflowResult = {
        analysisMode: mode,
        analysisType: mode,
        baseUrl,
        createdAt: Date.now(),
        selectedImagePath: this.data.selectedImagePath,
        selectedImageName: this.data.selectedImageName,
        uploadResult,
        taskSubmitResult,
        taskStatusResult,
        taskResult,
        alignResult,
        analyzeResult,
      };

      app.setWorkflowResult(workflowResult);
      this.updateStep('render', 'done');

      this.setData({
        loading: false,
        actionButtonText: config.actionButtonText,
        workflowDone: true,
        currentStepText: `${config.label}辅助分析已完成。`,
        latestFolderLabel: uploadResult.folder || folder || '--',
        latestResultCount: resultImageCount(analyzeResult),
      });

      wx.navigateTo({
        url: '/pages/result/result',
      });
    } catch (error) {
      const message = normalizeError(error.message || error, `${config.title}流程失败。`);
      this.setData({
        loading: false,
        actionButtonText: config.actionButtonText,
        workflowDone: false,
        currentStepText: '流程在完成前中断。',
        errorMessage: message,
      });

      const activeStep = this.data.steps.find((item) => item.status === 'active');
      if (activeStep) {
        this.updateStep(activeStep.key, 'error');
      }
    }
  },

  openResult() {
    wx.navigateTo({
      url: '/pages/result/result',
    });
  },
});
