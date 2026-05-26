const app = getApp();
const {
  API_PATHS,
  DEFAULT_BASE_URL,
  buildChatDiagnosisContextDetailPath,
} = require('./config');
const {
  appendAssistantDelta,
  createLocalId,
  createLocalMessage,
  decodeArrayBuffer,
  parseNdjsonChunk,
  replaceMessage,
} = require('./chat');
const { normalizeError, request, requestStream } = require('./request');
const { buildAssistanceContext, contextContainsAssistanceContext } = require('./analysis');

const CONTEXT_MESSAGE_TYPES = {
  'diagnosis-summary': true,
  'diagnosis-report': true,
};

function normalizeScore(score) {
  if (typeof score === 'number' && Number.isFinite(score)) {
    return Math.round(score * 10) / 10;
  }
  if (typeof score === 'string' && score.trim()) {
    const parsed = Number(score);
    if (Number.isFinite(parsed)) {
      return Math.round(parsed * 10) / 10;
    }
  }
  return null;
}

function scoreText(score) {
  const normalizedScore = normalizeScore(score);
  if (normalizedScore === null) {
    return '--';
  }
  return String(normalizedScore);
}

function buildMessageView(message) {
  const role = message.role === 'user' ? 'user' : 'assistant';
  return {
    ...message,
    role,
    roleLabel: role === 'user' ? '我' : 'AI',
    contentDisplay: message.content || (role === 'assistant' ? '生成中...' : ''),
    isAssistant: role === 'assistant',
    isUser: role === 'user',
  };
}

function isContextMessage(message) {
  return Boolean(CONTEXT_MESSAGE_TYPES[String(message.message_type || '')]);
}

function buildDiagnosisContextView(diagnosisContext) {
  if (!diagnosisContext) {
    return null;
  }

  const metricHighlights = Array.isArray(diagnosisContext.metricHighlights)
    ? diagnosisContext.metricHighlights
    : [];
  const diagnosisContexts = Array.isArray(diagnosisContext.diagnosisContexts)
    ? diagnosisContext.diagnosisContexts
    : [];
  return {
    ...diagnosisContext,
    totalScoreText: scoreText(diagnosisContext.totalScore),
    metricHighlights,
    diagnosisContexts,
    contextLabels: diagnosisContexts.length
      ? diagnosisContexts
          .map((item) => item.sourceLabel || item.sourceType || '辅助分析')
          .join('、')
      : diagnosisContext.sourceLabel || '',
  };
}

function buildSessionDisplay(chatSession) {
  const sessionId = chatSession.sessionId || '';
  const title = String(chatSession.title || '').trim();
  if (title) {
    return title;
  }
  return sessionId ? `会话 ${sessionId.slice(0, 8)}` : '未创建会话';
}

function buildPageState(chatSession) {
  const diagnosisContext = buildDiagnosisContextView(chatSession.diagnosisContext || null);
  return {
    sessionId: chatSession.sessionId || '',
    sessionIdDisplay: buildSessionDisplay(chatSession),
    messages: Array.isArray(chatSession.messages)
      ? chatSession.messages.filter((message) => !isContextMessage(message)).map(buildMessageView)
      : [],
    diagnosisContext,
    diagnosisMetricHighlights: diagnosisContext ? diagnosisContext.metricHighlights : [],
    pendingAttachments: Array.isArray(chatSession.pendingAttachments)
      ? chatSession.pendingAttachments
      : [],
  };
}

async function ensureSession(page) {
  if (page.data.sessionId) {
    return page.data.sessionId;
  }

  const response = await request({
    baseUrl: page.data.baseUrl || DEFAULT_BASE_URL,
    path: API_PATHS.CHAT_SESSION,
    method: 'POST',
    data: {
      session_id: app.getChatSession().sessionId || '',
    },
  });
  app.setChatSession(response.session || {});
  page.setData(buildPageState(app.getChatSession()));
  return app.getChatSession().sessionId || '';
}

async function writeLatestDiagnosisContext(page) {
  const latestReports = ['tongue', 'face']
    .map((analysisType) => app.getLatestWorkflowResultByType(analysisType))
    .filter((item) => item && item.analyzeResult);
  const workflowResult = latestReports[0] || app.getWorkflowResult();
  const analysisResults = latestReports.length
    ? latestReports.map((item) => item.analyzeResult)
    : workflowResult && workflowResult.analyzeResult
      ? [workflowResult.analyzeResult]
      : [];
  if (!analysisResults.length) {
    throw new Error('没有可用的最近辅助分析结果，请先完成一次舌象或面象分析。');
  }

  const sessionId = await ensureSession(page);
  const latestContexts = analysisResults.map(buildAssistanceContext);
  const currentContext = app.getDiagnosisContext();
  const allImported = latestContexts.every((item) =>
    contextContainsAssistanceContext(currentContext, item)
  );
  if (allImported) {
    page.setData(buildPageState(app.getChatSession()));
    return { skipped: true };
  }

  const response = await request({
    baseUrl: page.data.baseUrl || DEFAULT_BASE_URL,
    path: API_PATHS.CHAT_DIAGNOSIS_CONTEXT,
    method: 'POST',
    data: {
      session_id: sessionId,
      analysis_result: analysisResults[0],
      analysis_results: analysisResults,
    },
  });
  app.setChatSession(response.session || {});
  page.setData(buildPageState(app.getChatSession()));
  return { skipped: false };
}

async function dropDiagnosisContext(page) {
  const sessionId = await ensureSession(page);
  const response = await request({
    baseUrl: page.data.baseUrl || DEFAULT_BASE_URL,
    path: buildChatDiagnosisContextDetailPath(sessionId),
    method: 'DELETE',
  });
  app.setChatSession(response.session || {});
  page.setData(buildPageState(app.getChatSession()));
}

async function sendStreamTurn(page, { text, placeholderId, clientMessageId }) {
  let remainder = '';
  let streamError = '';
  let completedSession = null;

  const applyEvents = (events) => {
    events.forEach((eventPayload) => {
      const eventType = eventPayload.type;
      if (eventType === 'start' && eventPayload.session_id) {
        page.setData({
          sessionId: eventPayload.session_id,
        });
        return;
      }

      if (eventType === 'delta') {
        const nextMessages = appendAssistantDelta(
          app.getChatSession().messages || [],
          placeholderId,
          eventPayload.delta || ''
        );
        app.updateChatSession({
          messages: nextMessages,
        });
        page.setData(buildPageState(app.getChatSession()));
        return;
      }

      if (eventType === 'done') {
        completedSession = eventPayload.session || null;
        if (completedSession) {
          app.setChatSession(completedSession);
          page.setData(buildPageState(app.getChatSession()));
          return;
        }

        if (eventPayload.assistant_message) {
          app.updateChatSession({
            messages: replaceMessage(
              app.getChatSession().messages || [],
              placeholderId,
              eventPayload.assistant_message
            ),
          });
          page.setData(buildPageState(app.getChatSession()));
        }
        return;
      }

      if (eventType === 'context' && eventPayload.diagnosis_context) {
        app.setDiagnosisContext(eventPayload.diagnosis_context);
        page.setData(buildPageState(app.getChatSession()));
        return;
      }

      if (eventType === 'error') {
        streamError = eventPayload.message || '流式回复失败。';
      }
    });
  };

  const response = await requestStream({
    baseUrl: page.data.baseUrl || DEFAULT_BASE_URL,
    path: API_PATHS.CHAT_STREAM,
    method: 'POST',
    data: {
      session_id: page.data.sessionId,
      text,
      attachment_ids: [],
      client_message_id: clientMessageId,
    },
    onChunk: (chunk) => {
      const parsed = parseNdjsonChunk(decodeArrayBuffer(chunk), remainder);
      remainder = parsed.remainder;
      applyEvents(parsed.events);
    },
  });

  const finalText = decodeArrayBuffer(response.data);
  if (finalText) {
    const parsed = parseNdjsonChunk(finalText, remainder);
    remainder = parsed.remainder;
    applyEvents(parsed.events);
  }

  if (remainder.trim()) {
    applyEvents(parseNdjsonChunk('\n', remainder).events);
  }

  if (streamError) {
    throw new Error(streamError);
  }
  if (!completedSession) {
    throw new Error('流式回复未正常完成。');
  }
}

async function sendFallbackTurn(page, { text, clientMessageId }) {
  const response = await request({
    baseUrl: page.data.baseUrl || DEFAULT_BASE_URL,
    path: API_PATHS.CHAT_MESSAGE,
    method: 'POST',
    data: {
      session_id: page.data.sessionId,
      text,
      attachment_ids: [],
      client_message_id: clientMessageId,
    },
  });
  app.setChatSession(response.session || {});
  page.setData(buildPageState(app.getChatSession()));
}

async function sendChatMessage(page, text) {
  const normalizedText = String(text || '').trim();
  if (!normalizedText) {
    throw new Error('请输入问题。');
  }

  const sessionId = await ensureSession(page);
  if (!sessionId) {
    throw new Error('聊天会话创建失败。');
  }

  const clientMessageId = createLocalId('turn');
  const optimisticUserMessage = createLocalMessage({
    role: 'user',
    content: normalizedText,
    metadata: { client_message_id: clientMessageId },
  });
  const assistantPlaceholderId = createLocalId('assistant');
  const assistantPlaceholder = {
    ...createLocalMessage({
      role: 'assistant',
      content: '',
      metadata: { streaming: true, client_message_id: clientMessageId },
    }),
    message_id: assistantPlaceholderId,
  };

  app.updateChatSession({
    messages: [
      ...(app.getChatSession().messages || []),
      optimisticUserMessage,
      assistantPlaceholder,
    ],
    pendingAttachments: [],
  });
  page.setData(buildPageState(app.getChatSession()));

  try {
    await sendStreamTurn(page, {
      text: normalizedText,
      placeholderId: assistantPlaceholderId,
      clientMessageId,
    });
  } catch (streamError) {
    try {
      await sendFallbackTurn(page, {
        text: normalizedText,
        clientMessageId,
      });
    } catch (fallbackError) {
      throw new Error(normalizeError(fallbackError.message || fallbackError, streamError.message));
    }
  }
}

module.exports = {
  buildPageState,
  dropDiagnosisContext,
  ensureSession,
  sendChatMessage,
  writeLatestDiagnosisContext,
};
