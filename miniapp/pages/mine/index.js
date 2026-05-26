const app = getApp();
const { API_PATHS, DEFAULT_BASE_URL } = require('../../utils/config');
const { normalizeError, request } = require('../../utils/request');

Page({
  data: {
    baseUrl: DEFAULT_BASE_URL,
    baseUrlDraft: DEFAULT_BASE_URL,
    connectionTesting: false,
    connectionStatus: '',
    errorMessage: '',
    profileStats: '报告 0 份 · 连续打卡 0 天',
  },

  onShow() {
    const baseUrl = app.globalData.baseUrl || DEFAULT_BASE_URL;
    const reportCount = app.getWorkflowResults().length;
    this.setData({
      baseUrl,
      baseUrlDraft: baseUrl,
      profileStats: `报告 ${reportCount} 份 · 连续打卡 ${reportCount ? 2 : 0} 天`,
    });
  },

  handleBaseUrlInput(event) {
    this.setData({
      baseUrlDraft: String(event.detail.value || '').trim(),
      connectionStatus: '',
      errorMessage: '',
    });
  },

  saveBaseUrl() {
    const baseUrl = String(this.data.baseUrlDraft || '').trim();
    if (!baseUrl) {
      wx.showToast({
        title: '请填写后端地址',
        icon: 'none',
      });
      return;
    }

    app.setBaseUrl(baseUrl);
    this.setData({
      baseUrl,
      connectionStatus: '后端地址已保存。',
      errorMessage: '',
    });
  },

  async testConnection() {
    const baseUrl = String(this.data.baseUrlDraft || this.data.baseUrl || DEFAULT_BASE_URL).trim();
    this.setData({
      connectionTesting: true,
      connectionStatus: '正在测试后端连接...',
      errorMessage: '',
    });

    try {
      const response = await request({
        baseUrl,
        path: API_PATHS.CHAT_SESSION,
        method: 'POST',
        data: {
          session_id: app.getChatSession().sessionId || '',
        },
      });
      app.setBaseUrl(baseUrl);
      app.setChatSession(response.session || {});
      this.setData({
        baseUrl,
        baseUrlDraft: baseUrl,
        connectionTesting: false,
        connectionStatus: '后端连接正常，聊天会话已准备。',
      });
    } catch (error) {
      this.setData({
        connectionTesting: false,
        connectionStatus: '',
        errorMessage: normalizeError(error.message || error, '后端连接失败。'),
      });
    }
  },

  openReport() {
    wx.navigateTo({
      url: '/pages/report-history/index',
    });
  },

  openChatHistory() {
    wx.navigateTo({
      url: '/pages/chat-history/index',
    });
  },
});
