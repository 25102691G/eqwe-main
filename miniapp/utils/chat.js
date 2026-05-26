function createLocalId(prefix = 'local') {
  return `${prefix}-${Date.now()}-${Math.random().toString(16).slice(2, 8)}`;
}

const SLASH_COMMANDS = [
  {
    name: '/help',
    description: '查看可用命令',
    insertText: '/help',
    action: 'help',
  },
  {
    name: '/new',
    description: '创建一个新的聊天会话',
    insertText: '/new',
    action: 'new',
  },
  {
    name: '/threads',
    description: '查看最近会话',
    insertText: '/threads',
    action: 'thread',
  },
  {
    name: '/use-latest-diagnosis',
    description: '把最近一次辅助分析结果写入当前会话上下文',
    insertText: '/use-latest-diagnosis',
    action: 'use-latest-diagnosis',
  },
  {
    name: '/drop-diagnosis',
    description: '移除当前会话里的辅助分析上下文',
    insertText: '/drop-diagnosis',
    action: 'drop-diagnosis',
  },
  {
    name: '/clear',
    description: '清空当前输入框',
    insertText: '/clear',
    action: 'clear',
  },
];

function normalizeSlashInput(text = '') {
  return String(text || '').replace(/^\s+/, '');
}

function getSlashCommandByName(name = '') {
  const normalizedName = String(name || '').trim().toLowerCase();
  return SLASH_COMMANDS.find((item) => item.name === normalizedName) || null;
}

function parseSlashCommand(text = '') {
  const normalizedText = normalizeSlashInput(text);
  if (!normalizedText.startsWith('/')) {
    return null;
  }

  const trimmedText = normalizedText.trim();
  if (!trimmedText) {
    return null;
  }

  const firstSpaceIndex = trimmedText.indexOf(' ');
  const rawName =
    firstSpaceIndex === -1
      ? trimmedText.toLowerCase()
      : trimmedText.slice(0, firstSpaceIndex).toLowerCase();
  const argsText =
    firstSpaceIndex === -1 ? '' : trimmedText.slice(firstSpaceIndex + 1).trim();
  const command = getSlashCommandByName(rawName);
  if (!command) {
    return {
      rawName,
      name: rawName,
      action: rawName,
      argsText,
      recognized: false,
    };
  }

  return {
    rawName,
    name: command.name,
    action: command.action || command.name,
    argsText,
    recognized: true,
  };
}

function buildSlashCommandSuggestions(text = '') {
  const normalizedText = normalizeSlashInput(text);
  if (!normalizedText.startsWith('/')) {
    return [];
  }

  const commandTokenMatch = normalizedText.match(/^\/\S*/);
  const searchToken = commandTokenMatch ? commandTokenMatch[0].toLowerCase() : '/';
  const hasArgs = /\s/.test(normalizedText);
  if (hasArgs && getSlashCommandByName(searchToken)) {
    return [];
  }

  if (searchToken === '/') {
    return SLASH_COMMANDS.map((item) => ({ ...item }));
  }

  return SLASH_COMMANDS.filter((item) => item.name.indexOf(searchToken) === 0).map(
    (item) => ({ ...item })
  );
}

function createLocalMessage({
  role,
  content = '',
  messageType = 'text',
  attachments = [],
  metadata = {},
}) {
  return {
    message_id: createLocalId('message'),
    role,
    content,
    message_type: messageType,
    attachments,
    metadata,
    created_at: new Date().toISOString(),
  };
}

function replaceMessage(messages, targetId, nextMessage) {
  return messages.map((item) => (item.message_id === targetId ? nextMessage : item));
}

function appendAssistantDelta(messages, targetId, delta) {
  return messages.map((item) => {
    if (item.message_id !== targetId) {
      return item;
    }
    return {
      ...item,
      content: `${item.content || ''}${delta || ''}`,
    };
  });
}

function decodeArrayBuffer(payload) {
  if (!payload) {
    return '';
  }
  if (typeof payload === 'string') {
    return payload;
  }
  if (typeof TextDecoder !== 'undefined') {
    return new TextDecoder('utf-8').decode(new Uint8Array(payload));
  }

  const bytes = new Uint8Array(payload);
  let result = '';
  for (let index = 0; index < bytes.length; index += 1) {
    result += String.fromCharCode(bytes[index]);
  }
  try {
    return decodeURIComponent(escape(result));
  } catch (error) {
    return result;
  }
}

function parseNdjsonChunk(chunkText, remainder = '') {
  const merged = `${remainder}${chunkText || ''}`;
  const lines = merged.split('\n');
  const nextRemainder = lines.pop() || '';
  const events = [];

  lines.forEach((line) => {
    const trimmed = line.trim();
    if (!trimmed) {
      return;
    }
    try {
      events.push(JSON.parse(trimmed));
    } catch (error) {
      console.warn('Failed to parse stream event.', error, trimmed);
    }
  });

  return {
    events,
    remainder: nextRemainder,
  };
}

module.exports = {
  SLASH_COMMANDS,
  appendAssistantDelta,
  buildSlashCommandSuggestions,
  createLocalId,
  createLocalMessage,
  decodeArrayBuffer,
  parseSlashCommand,
  parseNdjsonChunk,
  replaceMessage,
};
