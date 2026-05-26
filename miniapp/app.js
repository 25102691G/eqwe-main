const { DEFAULT_BASE_URL } = require('./utils/config');

const STORAGE_KEYS = {
  baseUrl: 'miniapp.baseUrl',
  workflowResult: 'miniapp.workflowResult',
  workflowResults: 'miniapp.workflowResults',
  chatSession: 'miniapp.chatSession',
  diagnosisContext: 'miniapp.diagnosisContext',
};

const MAX_WORKFLOW_HISTORY = 50;

function pickText() {
  for (let index = 0; index < arguments.length; index += 1) {
    const value = arguments[index];
    if (typeof value === 'string' && value.trim()) {
      return value.trim();
    }
  }
  return '';
}

function isTongueAnalysisResult(result) {
  const payload = result || {};
  return Boolean(payload.tongue_color || payload.tongue_moisture || payload.demo_report);
}

function resolveWorkflowAnalysisType(workflowResult) {
  const payload = workflowResult || {};
  const analyzeResult = payload.analyzeResult || {};
  return pickText(
    payload.analysisType,
    payload.analysisMode,
    isTongueAnalysisResult(analyzeResult) ? 'tongue' : '',
    'face'
  );
}

function resolveWorkflowReportId(workflowResult) {
  const payload = workflowResult || {};
  const analyzeResult = payload.analyzeResult || {};
  const storage = analyzeResult.storage || {};
  const metadata = analyzeResult.metadata || {};
  const taskSubmitResult = payload.taskSubmitResult || {};
  const taskStatusResult = payload.taskStatusResult || {};
  return pickText(
    payload.reportId,
    storage.folder,
    metadata.folder,
    taskStatusResult.task_id,
    taskSubmitResult.task_id,
    payload.selectedImageName && payload.createdAt
      ? `${payload.selectedImageName}-${payload.createdAt}`
      : '',
    payload.createdAt ? `report-${payload.createdAt}` : ''
  );
}

function normalizeWorkflowResult(payload) {
  if (!payload || typeof payload !== 'object') {
    return null;
  }

  const createdAt = Number(payload.createdAt || Date.now());
  const normalized = {
    ...payload,
    createdAt,
  };
  normalized.analysisType = resolveWorkflowAnalysisType(normalized);
  normalized.reportId = resolveWorkflowReportId(normalized);
  return normalized;
}

function normalizeWorkflowResults(payload) {
  const list = Array.isArray(payload) ? payload : [];
  const byId = {};
  list.forEach((item) => {
    const normalized = normalizeWorkflowResult(item);
    if (!normalized || !normalized.reportId) {
      return;
    }
    byId[normalized.reportId] = normalized;
  });
  return Object.keys(byId)
    .map((key) => byId[key])
    .sort((left, right) => Number(right.createdAt || 0) - Number(left.createdAt || 0))
    .slice(0, MAX_WORKFLOW_HISTORY);
}

function mergeWorkflowResults(existingResults, nextResult) {
  const normalizedNext = normalizeWorkflowResult(nextResult);
  if (!normalizedNext) {
    return normalizeWorkflowResults(existingResults);
  }
  return normalizeWorkflowResults([...(existingResults || []), normalizedNext]);
}

function createDefaultChatSession() {
  return {
    sessionId: '',
    title: '',
    pinned: false,
    messages: [],
    diagnosisContext: null,
    pendingAttachments: [],
    attachments: [],
  };
}

function normalizeAttachment(payload) {
  const nextPayload = payload || {};
  return {
    attachment_id: nextPayload.attachment_id || nextPayload.attachmentId || '',
    session_id: nextPayload.session_id || nextPayload.sessionId || '',
    name: nextPayload.name || '',
    content_type: nextPayload.content_type || nextPayload.contentType || '',
    kind: nextPayload.kind || '',
    size_bytes:
      typeof nextPayload.size_bytes === 'number'
        ? nextPayload.size_bytes
        : nextPayload.sizeBytes || 0,
    stored_path: nextPayload.stored_path || nextPayload.storedPath || '',
    object_key: nextPayload.object_key || nextPayload.objectKey || '',
    download_url: nextPayload.download_url || nextPayload.downloadUrl || '',
    text_excerpt: nextPayload.text_excerpt || nextPayload.textExcerpt || '',
    extraction_error:
      nextPayload.extraction_error || nextPayload.extractionError || '',
    created_at: nextPayload.created_at || nextPayload.createdAt || '',
  };
}

function normalizeDiagnosisHighlight(payload) {
  const nextPayload = payload || {};
  const score =
    typeof nextPayload.score === 'number'
      ? nextPayload.score
      : typeof nextPayload.score === 'string' && nextPayload.score.trim()
        ? Number(nextPayload.score)
        : null;
  return {
    key: nextPayload.key || '',
    title: nextPayload.title || nextPayload.key || '',
    score: Number.isFinite(score) ? score : null,
    summary: nextPayload.summary || '',
  };
}

function normalizeDiagnosisContext(payload) {
  if (!payload) {
    return null;
  }

  const nextPayload = payload || {};
  const totalScore =
    typeof nextPayload.totalScore === 'number'
      ? nextPayload.totalScore
      : typeof nextPayload.total_score === 'number'
        ? nextPayload.total_score
        : null;
  return {
    sourceType: nextPayload.sourceType || nextPayload.source_type || '',
    sourceLabel: nextPayload.sourceLabel || nextPayload.source_label || '',
    sourceFolder: nextPayload.sourceFolder || nextPayload.source_folder || '',
    totalScore,
    totalScoreText: nextPayload.totalScoreText || nextPayload.total_score_text || '',
    summary: nextPayload.summary || '',
    metricHighlights: Array.isArray(nextPayload.metricHighlights)
      ? nextPayload.metricHighlights.map(normalizeDiagnosisHighlight)
      : Array.isArray(nextPayload.metric_highlights)
        ? nextPayload.metric_highlights.map(normalizeDiagnosisHighlight)
        : [],
    diagnosisContexts: Array.isArray(nextPayload.diagnosisContexts)
      ? nextPayload.diagnosisContexts.map(normalizeDiagnosisContext).filter(Boolean)
      : Array.isArray(nextPayload.diagnosis_contexts)
        ? nextPayload.diagnosis_contexts.map(normalizeDiagnosisContext).filter(Boolean)
        : [],
    reportUrl: nextPayload.reportUrl || nextPayload.report_url || '',
    updatedAt: nextPayload.updatedAt || nextPayload.updated_at || '',
  };
}

function normalizeChatMessage(payload) {
  const nextPayload = payload || {};
  return {
    message_id: nextPayload.message_id || nextPayload.messageId || '',
    role: nextPayload.role || '',
    content: nextPayload.content || '',
    message_type: nextPayload.message_type || nextPayload.messageType || 'text',
    attachments: Array.isArray(nextPayload.attachments)
      ? nextPayload.attachments.map(normalizeAttachment)
      : [],
    metadata:
      nextPayload.metadata && typeof nextPayload.metadata === 'object'
        ? nextPayload.metadata
        : {},
    created_at: nextPayload.created_at || nextPayload.createdAt || '',
  };
}

function normalizeChatSession(payload) {
  const nextPayload = payload || {};
  return {
    sessionId: nextPayload.sessionId || nextPayload.session_id || '',
    title: nextPayload.title || '',
    pinned: Boolean(nextPayload.pinned),
    messages: Array.isArray(nextPayload.messages)
      ? nextPayload.messages.map(normalizeChatMessage)
      : [],
    diagnosisContext: normalizeDiagnosisContext(
      nextPayload.diagnosisContext || nextPayload.diagnosis_context || null
    ),
    pendingAttachments: Array.isArray(nextPayload.pendingAttachments)
      ? nextPayload.pendingAttachments.map(normalizeAttachment)
      : [],
    attachments: Array.isArray(nextPayload.attachments)
      ? nextPayload.attachments.map(normalizeAttachment)
      : [],
    createdAt: nextPayload.createdAt || nextPayload.created_at || '',
    updatedAt: nextPayload.updatedAt || nextPayload.updated_at || '',
  };
}

App({
  globalData: {
    baseUrl: DEFAULT_BASE_URL,
    workflowResult: null,
    workflowResults: [],
    chatSession: createDefaultChatSession(),
  },

  onLaunch() {
    try {
      const savedBaseUrl = wx.getStorageSync(STORAGE_KEYS.baseUrl);
      if (savedBaseUrl) {
        this.globalData.baseUrl = savedBaseUrl;
      }

      const savedWorkflowResults = normalizeWorkflowResults(
        wx.getStorageSync(STORAGE_KEYS.workflowResults)
      );
      this.globalData.workflowResults = savedWorkflowResults;

      const savedWorkflowResult = normalizeWorkflowResult(
        wx.getStorageSync(STORAGE_KEYS.workflowResult)
      );
      if (savedWorkflowResult) {
        this.globalData.workflowResult = savedWorkflowResult;
        this.globalData.workflowResults = mergeWorkflowResults(
          this.globalData.workflowResults,
          savedWorkflowResult
        );
      } else if (this.globalData.workflowResults.length) {
        this.globalData.workflowResult = this.globalData.workflowResults[0];
      }
      wx.setStorageSync(STORAGE_KEYS.workflowResults, this.globalData.workflowResults);

      const savedChatSession = wx.getStorageSync(STORAGE_KEYS.chatSession);
      if (savedChatSession) {
        this.globalData.chatSession = normalizeChatSession(savedChatSession);
      }

      const savedDiagnosisContext = wx.getStorageSync(STORAGE_KEYS.diagnosisContext);
      if (savedDiagnosisContext) {
        this.globalData.chatSession = normalizeChatSession({
          ...this.globalData.chatSession,
          diagnosisContext: savedDiagnosisContext,
        });
      }
    } catch (error) {
      console.warn('Failed to load cached miniapp state.', error);
    }
  },

  setBaseUrl(baseUrl) {
    this.globalData.baseUrl = baseUrl;
    wx.setStorageSync(STORAGE_KEYS.baseUrl, baseUrl);
  },

  setWorkflowResult(payload) {
    const normalized = normalizeWorkflowResult(payload);
    if (!normalized) {
      return null;
    }
    this.globalData.workflowResult = normalized;
    this.globalData.workflowResults = mergeWorkflowResults(
      this.globalData.workflowResults,
      normalized
    );
    wx.setStorageSync(STORAGE_KEYS.workflowResult, normalized);
    wx.setStorageSync(STORAGE_KEYS.workflowResults, this.globalData.workflowResults);
    return normalized;
  },

  clearWorkflowResult() {
    this.globalData.workflowResult = null;
    wx.removeStorageSync(STORAGE_KEYS.workflowResult);
  },

  getWorkflowResult() {
    if (this.globalData.workflowResult) {
      return this.globalData.workflowResult;
    }

    try {
      const cached = normalizeWorkflowResult(wx.getStorageSync(STORAGE_KEYS.workflowResult));
      if (cached) {
        this.globalData.workflowResult = cached;
        return cached;
      }

    } catch (error) {
      console.warn('Failed to load cached workflow result.', error);
    }

    return null;
  },

  getWorkflowResults() {
    if (Array.isArray(this.globalData.workflowResults) && this.globalData.workflowResults.length) {
      return this.globalData.workflowResults;
    }

    try {
      const cached = normalizeWorkflowResults(wx.getStorageSync(STORAGE_KEYS.workflowResults));
      this.globalData.workflowResults = cached;
      return cached;
    } catch (error) {
      console.warn('Failed to load cached workflow results.', error);
    }

    return [];
  },

  getLatestWorkflowResultByType(analysisType) {
    const normalizedType = String(analysisType || '').trim();
    return (
      this.getWorkflowResults().find(
        (item) => String(item.analysisType || item.analysisMode || '').trim() === normalizedType
      ) || null
    );
  },

  getWorkflowResultById(reportId) {
    const normalizedReportId = String(reportId || '').trim();
    if (!normalizedReportId) {
      return null;
    }
    return (
      this.getWorkflowResults().find(
        (item) => String(item.reportId || '').trim() === normalizedReportId
      ) || null
    );
  },

  setChatSession(payload) {
    const nextSession = normalizeChatSession(payload);
    this.globalData.chatSession = nextSession;
    wx.setStorageSync(STORAGE_KEYS.chatSession, nextSession);
    if (nextSession.diagnosisContext) {
      wx.setStorageSync(STORAGE_KEYS.diagnosisContext, nextSession.diagnosisContext);
    } else {
      wx.removeStorageSync(STORAGE_KEYS.diagnosisContext);
    }
    return nextSession;
  },

  updateChatSession(patch) {
    return this.setChatSession({
      ...this.getChatSession(),
      ...patch,
    });
  },

  getChatSession() {
    if (this.globalData.chatSession && this.globalData.chatSession.sessionId) {
      return this.globalData.chatSession;
    }

    try {
      const cached = wx.getStorageSync(STORAGE_KEYS.chatSession);
      if (cached) {
        this.globalData.chatSession = normalizeChatSession(cached);
        return this.globalData.chatSession;
      }
    } catch (error) {
      console.warn('Failed to load cached chat session.', error);
    }

    return this.globalData.chatSession || createDefaultChatSession();
  },

  clearChatSession() {
    this.globalData.chatSession = createDefaultChatSession();
    wx.removeStorageSync(STORAGE_KEYS.chatSession);
    wx.removeStorageSync(STORAGE_KEYS.diagnosisContext);
  },

  setDiagnosisContext(payload) {
    const nextSession = this.updateChatSession({
      diagnosisContext: payload || null,
    });
    return nextSession.diagnosisContext;
  },

  getDiagnosisContext() {
    return this.getChatSession().diagnosisContext || null;
  },

  setPendingAttachments(attachments) {
    return this.updateChatSession({
      pendingAttachments: Array.isArray(attachments) ? attachments : [],
    }).pendingAttachments;
  },

  appendChatMessage(message) {
    const chatSession = this.getChatSession();
    return this.updateChatSession({
      messages: [...chatSession.messages, message],
    });
  },
});
