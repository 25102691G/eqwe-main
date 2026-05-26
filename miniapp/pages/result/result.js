const app = getApp();
const {
  buildAssistanceContext,
  contextContainsAssistanceContext,
  extractGallery,
  extractOverallReport,
  extractSummaryCards,
  extractTongueConstitutionCards,
  isTongueAnalysis,
  stringifyPayload,
} = require('../../utils/analysis');
const { API_PATHS, DEFAULT_BASE_URL } = require('../../utils/config');
const { joinUrl, normalizeError, request, requestUrl } = require('../../utils/request');

function resolveResultUrl(url, baseUrl) {
  if (!url) {
    return '';
  }
  if (/^https?:\/\//.test(url)) {
    return url;
  }
  return joinUrl(baseUrl || DEFAULT_BASE_URL, url);
}

Page({
  data: {
    hasResult: false,
    activeTab: 'summary',
    sourceImagePath: '',
    alignedImageUrl: '',
    showAlignedImage: false,
    folderLabel: '--',
    analysisTimestamp: '',
    reportUrl: '',
    rawJson: '',
    overallReport: {
      totalScoreText: '--',
      summary: '',
      generationMode: '',
    },
    summaryCards: [],
    gallery: [],
    galleryCount: 0,
    analysisModeLabel: '面象肤况',
    overallTitle: '辅助分析摘要',
    scoreLabel: '参考分',
    isTongueResult: false,
    constitutionCards: [],
    hasConstitutionCards: false,
    disclaimerText: '本结果用于健康状态辅助分析和生活护理建议，不替代医生面诊。',
    diagnosisLinked: false,
    diagnosisStatusDisplay: '结果页已打开。',
    linkingDiagnosis: false,
    errorMessage: '',
    reportId: '',
  },

  onLoad(options) {
    this.setData({
      reportId: options && options.reportId ? decodeURIComponent(options.reportId) : '',
    });
  },

  onShow() {
    this.loadResult();
  },

  applyResult(workflowResult, analyzeResultOverride) {
    const uploadResult = workflowResult.uploadResult || {};
    const alignResult = workflowResult.alignResult || {};
    const analyzeResult = analyzeResultOverride || workflowResult.analyzeResult || {};
    const isTongueResult = isTongueAnalysis(analyzeResult);
    const summaryCards = extractSummaryCards(analyzeResult);
    const baseUrl = workflowResult.baseUrl || app.globalData.baseUrl || DEFAULT_BASE_URL;
    const gallery = extractGallery(analyzeResult).map((item) => ({
      ...item,
      url: resolveResultUrl(item.url || '', baseUrl),
    }));
    const constitutionCards = isTongueResult ? extractTongueConstitutionCards(analyzeResult) : [];
    const metadata = analyzeResult.metadata || {};
    const storage = analyzeResult.storage || {};
    const folder = uploadResult.folder || storage.folder || '';

    this.setData({
      hasResult: true,
      analysisModeLabel: isTongueResult ? '舌象一期' : '面象肤况',
      overallTitle: isTongueResult ? '舌象概要' : '肤况概要',
      scoreLabel: isTongueResult ? '质量' : '参考分',
      isTongueResult,
      sourceImagePath: workflowResult.selectedImagePath || '',
      alignedImageUrl: alignResult.aligned_image_url || '',
      showAlignedImage: Boolean(alignResult.aligned_image_url && !isTongueResult),
      folderLabel: folder || '--',
      analysisTimestamp: metadata.analysis_timestamp || '',
      reportUrl: analyzeResult.analysis_report_url || '',
      rawJson: stringifyPayload(analyzeResult),
      overallReport: extractOverallReport(analyzeResult, summaryCards),
      summaryCards,
      gallery,
      galleryCount: gallery.length,
      constitutionCards,
      hasConstitutionCards: constitutionCards.length > 0,
      disclaimerText:
        (analyzeResult.demo_report && analyzeResult.demo_report.disclaimer) ||
        (analyzeResult.tongue_image_assistance &&
          analyzeResult.tongue_image_assistance.disclaimer) ||
        '本结果用于健康状态辅助分析和生活护理建议，不替代医生面诊。',
    });
  },

  async refreshLatestAnalyzeResult(workflowResult) {
    const analyzeResult = workflowResult.analyzeResult || {};
    const reportUrl = analyzeResult.analysis_report_url || '';
    if (!reportUrl) {
      return;
    }

    try {
      const latestAnalyzeResult = await requestUrl({ url: reportUrl });
      const nextWorkflowResult = {
        ...workflowResult,
        analyzeResult: latestAnalyzeResult,
      };
      if (!this.data.reportId) {
        app.setWorkflowResult(nextWorkflowResult);
      }
      this.applyResult(nextWorkflowResult, latestAnalyzeResult);
    } catch (error) {
      console.warn('Failed to refresh latest analysis result.', error);
    }
  },

  loadResult() {
    const reportId = String(this.data.reportId || '').trim();
    const workflowResult = reportId ? app.getWorkflowResultById(reportId) : app.getWorkflowResult();
    if (!workflowResult) {
      this.setData({
        hasResult: false,
      });
      return;
    }

    this.applyResult(workflowResult);
    this.refreshLatestAnalyzeResult(workflowResult);
    this.attachDiagnosisContext(workflowResult);
  },

  switchTab(event) {
    this.setData({
      activeTab: event.currentTarget.dataset.value,
    });
  },

  previewSourceImage() {
    if (!this.data.sourceImagePath) {
      return;
    }
    wx.previewImage({
      current: this.data.sourceImagePath,
      urls: [this.data.sourceImagePath],
    });
  },

  previewAlignedImage() {
    if (!this.data.alignedImageUrl) {
      return;
    }
    wx.previewImage({
      current: this.data.alignedImageUrl,
      urls: [this.data.alignedImageUrl],
    });
  },

  previewGalleryImage(event) {
    const current = event.currentTarget.dataset.url;
    wx.previewImage({
      current,
      urls: this.data.gallery.map((item) => item.url),
    });
  },

  copyJson() {
    wx.setClipboardData({
      data: this.data.rawJson,
    });
  },

  async attachDiagnosisContext(workflowResult) {
    const baseUrl = workflowResult.baseUrl || app.globalData.baseUrl || DEFAULT_BASE_URL;
    let chatSession = app.getChatSession();
    let sessionId = chatSession.sessionId || '';
    const analyzeResult = workflowResult.analyzeResult || {};
    const isTongueResult = isTongueAnalysis(analyzeResult);
    const currentContext = app.getDiagnosisContext();
    const nextContext = buildAssistanceContext(analyzeResult);

    if (contextContainsAssistanceContext(currentContext, nextContext)) {
      this.setData({
        diagnosisLinked: true,
        diagnosisStatusDisplay: isTongueResult
          ? '本次舌象一期摘要已在 AI 问诊上下文中。'
          : '本次面象肤况摘要已在 AI 问诊上下文中。',
      });
      return;
    }

    if (!sessionId) {
      try {
        const response = await request({
          baseUrl,
          path: API_PATHS.CHAT_SESSION,
          method: 'POST',
          data: { session_id: '' },
        });
        if (response.session) {
          app.setChatSession(response.session);
          chatSession = app.getChatSession();
          sessionId = chatSession.sessionId || '';
        }
      } catch (error) {
        this.setData({
          diagnosisLinked: false,
          diagnosisStatusDisplay: normalizeError(
            error.message || error,
            '创建聊天会话失败，暂时无法写入上下文。'
          ),
        });
        return;
      }
    }

    if (!sessionId) {
      this.setData({
        diagnosisLinked: false,
        diagnosisStatusDisplay: '当前还没有可用的聊天会话。',
      });
      return;
    }

    this.setData({
      linkingDiagnosis: true,
      diagnosisStatusDisplay: '正在把本次辅助分析摘要写入聊天上下文...',
    });

    try {
      const response = await request({
        baseUrl,
        path: API_PATHS.CHAT_DIAGNOSIS_CONTEXT,
        method: 'POST',
        data: {
          session_id: sessionId,
          analysis_result: analyzeResult,
        },
      });

      if (response.session) {
        app.setChatSession(response.session);
      }

      this.setData({
        diagnosisLinked: true,
        diagnosisStatusDisplay: isTongueResult
          ? '本次舌象一期摘要已加入 AI 问诊上下文。'
          : '本次面象肤况摘要已加入 AI 问诊上下文。',
      });
    } catch (error) {
      this.setData({
        diagnosisLinked: false,
        diagnosisStatusDisplay: normalizeError(error.message || error, '写入聊天上下文失败。'),
      });
    } finally {
      this.setData({
        linkingDiagnosis: false,
      });
    }
  },

  goHome() {
    wx.reLaunch({
      url: '/pages/index/index',
    });
  },

  goAi() {
    wx.reLaunch({
      url: '/pages/ai/index',
    });
  },
});
