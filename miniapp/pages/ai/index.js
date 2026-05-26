const app = getApp();
const { DEFAULT_BASE_URL } = require('../../utils/config');
const { normalizeError } = require('../../utils/request');
const {
  buildPageState,
  dropDiagnosisContext,
  ensureSession,
  sendChatMessage,
  writeLatestDiagnosisContext,
} = require('../../utils/chat_page');

const QUICK_PROMPTS = [
  {
    key: 'report',
    title: '解释报告',
    text: '请用通俗的话解释我最近一次舌象或面象辅助分析报告，并列出最重要的三条护理建议。',
  },
  {
    key: 'plan',
    title: '一周方案',
    text: '请结合当前上下文，给我一份一周健康管理方案，包含饮食、睡眠、护肤和复测建议。',
  },
  {
    key: 'constitution',
    title: '体质倾向',
    text: '请按中医九大体质做辅助问询，先逐步问我关键问题，再给出体质倾向参考。',
  },
  {
    key: 'skin',
    title: '改善肤况',
    text: '请结合当前肤况辅助分析，给出温和、可执行的护理顺序。',
  },
];

Page({
  data: {
    baseUrl: DEFAULT_BASE_URL,
    sessionId: '',
    sessionIdDisplay: '未创建会话',
    messages: [],
    diagnosisContext: null,
    diagnosisMetricHighlights: [],
    inputText: '',
    loading: false,
    errorMessage: '',
    quickPrompts: QUICK_PROMPTS,
  },

  onShow() {
    this.setData({
      baseUrl: app.globalData.baseUrl || DEFAULT_BASE_URL,
      ...buildPageState(app.getChatSession()),
    });
  },

  handleInput(event) {
    this.setData({
      inputText: event.detail.value || '',
      errorMessage: '',
    });
  },

  useQuickPrompt(event) {
    const key = event.currentTarget.dataset.key;
    const prompt = QUICK_PROMPTS.find((item) => item.key === key);
    if (!prompt) {
      return;
    }
    this.setData({
      inputText: prompt.text,
      errorMessage: '',
    });
  },

  async createSession() {
    try {
      await ensureSession(this);
      wx.showToast({
        title: '会话已准备',
        icon: 'success',
      });
    } catch (error) {
      this.setData({
        errorMessage: normalizeError(error.message || error, '创建会话失败。'),
      });
    }
  },

  async useLatestReport() {
    try {
      const result = await writeLatestDiagnosisContext(this);
      wx.showToast({
        title: result && result.skipped ? '已导入' : '已写入上下文',
        icon: 'success',
      });
    } catch (error) {
      this.setData({
        errorMessage: normalizeError(error.message || error, '写入上下文失败。'),
      });
    }
  },

  async removeContext() {
    try {
      await dropDiagnosisContext(this);
      wx.showToast({
        title: '已移除上下文',
        icon: 'success',
      });
    } catch (error) {
      this.setData({
        errorMessage: normalizeError(error.message || error, '移除上下文失败。'),
      });
    }
  },

  startNewSession() {
    app.clearChatSession();
    this.setData({
      ...buildPageState(app.getChatSession()),
      inputText: '',
      errorMessage: '',
    });
  },

  openChatHistory() {
    wx.navigateTo({
      url: '/pages/chat-history/index',
    });
  },

  async sendMessage() {
    if (this.data.loading) {
      return;
    }

    const text = String(this.data.inputText || '').trim();
    if (!text) {
      wx.showToast({
        title: '请输入问题',
        icon: 'none',
      });
      return;
    }

    this.setData({
      loading: true,
      errorMessage: '',
      inputText: '',
    });

    try {
      await sendChatMessage(this, text);
    } catch (error) {
      this.setData({
        errorMessage: normalizeError(error.message || error, '发送失败。'),
      });
    } finally {
      this.setData({
        loading: false,
      });
    }
  },
});
