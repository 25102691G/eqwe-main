const app = getApp();
const { API_PATHS, DEFAULT_BASE_URL, buildChatSessionDetailPath } = require('../../utils/config');
const { normalizeError, request } = require('../../utils/request');

function formatSessionTime(value) {
  const date = new Date(value || 0);
  if (Number.isNaN(date.getTime())) {
    return '';
  }

  const month = String(date.getMonth() + 1).padStart(2, '0');
  const day = String(date.getDate()).padStart(2, '0');
  const hour = String(date.getHours()).padStart(2, '0');
  const minute = String(date.getMinutes()).padStart(2, '0');
  return `${month}-${day} ${hour}:${minute}`;
}

function buildContextLabel(summary) {
  if (!summary || !summary.has_diagnosis_context) {
    return '';
  }

  const text = String(summary.diagnosis_summary || '').trim();
  if (text.includes('舌象') && text.includes('面象')) {
    return '综合辅助分析';
  }
  if (text.includes('舌象')) {
    return '舌象一期';
  }
  if (text.includes('肤况') || text.includes('面象')) {
    return '面象肤况';
  }
  return '已关联报告';
}

function buildSessionItem(summary) {
  const sessionId = summary.session_id || '';
  const title = String(summary.title || '').trim() || `会话 ${sessionId.slice(0, 8)}`;
  const preview = String(summary.last_message_preview || '').trim();
  const contextLabel = buildContextLabel(summary);
  return {
    sessionId,
    title,
    pinned: Boolean(summary.pinned),
    messageCount: Number(summary.message_count || 0),
    preview: preview || (contextLabel ? '已导入报告上下文' : '暂无问询内容'),
    time: formatSessionTime(summary.updated_at || summary.created_at),
    contextLabel,
    hasContext: Boolean(contextLabel),
  };
}

function isMeaningfulSession(item) {
  return item.messageCount > 0 || item.hasContext || item.pinned;
}

Page({
  data: {
    baseUrl: DEFAULT_BASE_URL,
    loading: false,
    errorMessage: '',
    sessions: [],
    pinnedSessions: [],
    recentSessions: [],
  },

  onShow() {
    this.setData({
      baseUrl: app.globalData.baseUrl || DEFAULT_BASE_URL,
    });
    this.loadSessions();
  },

  async loadSessions() {
    this.setData({
      loading: true,
      errorMessage: '',
    });

    try {
      const response = await request({
        baseUrl: this.data.baseUrl,
        path: `${API_PATHS.CHAT_SESSIONS}?limit=50`,
      });
      const sessions = Array.isArray(response.sessions)
        ? response.sessions.map(buildSessionItem).filter(isMeaningfulSession)
        : [];
      this.setData({
        sessions,
        pinnedSessions: sessions.filter((item) => item.pinned),
        recentSessions: sessions.filter((item) => !item.pinned),
      });
    } catch (error) {
      this.setData({
        errorMessage: normalizeError(error.message || error, '加载问询记录失败。'),
      });
    } finally {
      this.setData({
        loading: false,
      });
    }
  },

  async restoreSession(event) {
    const sessionId = event.currentTarget.dataset.sessionId || '';
    if (!sessionId) {
      return;
    }

    try {
      const response = await request({
        baseUrl: this.data.baseUrl,
        path: buildChatSessionDetailPath(sessionId),
      });
      app.setChatSession(response.session || {});
      wx.reLaunch({
        url: '/pages/ai/index',
      });
    } catch (error) {
      wx.showToast({
        title: normalizeError(error.message || error, '恢复会话失败。'),
        icon: 'none',
      });
    }
  },

  async togglePinned(event) {
    const sessionId = event.currentTarget.dataset.sessionId || '';
    const pinnedPayload = event.currentTarget.dataset.pinned;
    const pinned = pinnedPayload === true || pinnedPayload === 'true';
    if (!sessionId) {
      return;
    }
    await this.updateSession(sessionId, { pinned: !pinned });
  },

  async updateSession(sessionId, patch) {
    try {
      await request({
        baseUrl: this.data.baseUrl,
        path: buildChatSessionDetailPath(sessionId),
        method: 'PATCH',
        data: patch,
      });
      await this.loadSessions();
    } catch (error) {
      wx.showToast({
        title: normalizeError(error.message || error, '更新会话失败。'),
        icon: 'none',
      });
    }
  },

  deleteSession(event) {
    const sessionId = event.currentTarget.dataset.sessionId || '';
    if (!sessionId) {
      return;
    }

    wx.showModal({
      title: '删除问询记录',
      content: '删除后无法恢复，是否继续？',
      confirmText: '删除',
      confirmColor: '#d96d57',
      success: async (modalResult) => {
        if (!modalResult.confirm) {
          return;
        }
        try {
          await request({
            baseUrl: this.data.baseUrl,
            path: buildChatSessionDetailPath(sessionId),
            method: 'DELETE',
          });
          if (app.getChatSession().sessionId === sessionId) {
            app.clearChatSession();
          }
          await this.loadSessions();
        } catch (error) {
          wx.showToast({
            title: normalizeError(error.message || error, '删除会话失败。'),
            icon: 'none',
          });
        }
      },
    });
  },

  startNewSession() {
    app.clearChatSession();
    wx.reLaunch({
      url: '/pages/ai/index',
    });
  },
});
