const { DEFAULT_BASE_URL } = require('./config');

function joinUrl(baseUrl, path) {
  const normalizedBase = (baseUrl || DEFAULT_BASE_URL).replace(/\/+$/, '');
  const normalizedPath = path && path.startsWith('/') ? path : `/${path || ''}`;
  return `${normalizedBase}${normalizedPath}`;
}

function safeJsonParse(payload) {
  if (typeof payload !== 'string') {
    return payload;
  }

  try {
    return JSON.parse(payload);
  } catch (error) {
    return payload;
  }
}

function normalizeError(payload, fallbackMessage = '请求失败') {
  if (!payload) {
    return fallbackMessage;
  }

  if (typeof payload === 'string') {
    return payload;
  }

  return payload.message || payload.errMsg || payload.error || fallbackMessage;
}

function request({ baseUrl = DEFAULT_BASE_URL, path, method = 'GET', data = null, header = {} }) {
  return new Promise((resolve, reject) => {
    const url = joinUrl(baseUrl, path);
    wx.request({
      url,
      method,
      data,
      timeout: 120000,
      header: {
        'content-type': 'application/json',
        ...header,
      },
      success(response) {
        const payload = safeJsonParse(response.data);
        if (response.statusCode >= 200 && response.statusCode < 300) {
          resolve(payload);
          return;
        }

        reject(new Error(normalizeError(payload, `请求失败：${url}`)));
      },
      fail(error) {
        reject(new Error(`${normalizeError(error, '网络请求失败')}：${url}`));
      },
    });
  });
}

function requestUrl({ url, method = 'GET', data = null, header = {} }) {
  return new Promise((resolve, reject) => {
    wx.request({
      url,
      method,
      data,
      timeout: 120000,
      header: {
        'content-type': 'application/json',
        ...header,
      },
      success(response) {
        const payload = safeJsonParse(response.data);
        if (response.statusCode >= 200 && response.statusCode < 300) {
          resolve(payload);
          return;
        }

        reject(new Error(normalizeError(payload, `请求失败：${url}`)));
      },
      fail(error) {
        reject(new Error(`${normalizeError(error, '网络请求失败')}：${url}`));
      },
    });
  });
}

function uploadImage({ baseUrl = DEFAULT_BASE_URL, path, filePath, folder }) {
  return uploadFile({
    baseUrl,
    path,
    filePath,
    formData: folder ? { folder } : {},
  });
}

function uploadFile({
  baseUrl = DEFAULT_BASE_URL,
  path,
  filePath,
  name = 'file',
  formData = {},
  header = {},
}) {
  return new Promise((resolve, reject) => {
    const url = joinUrl(baseUrl, path);
    wx.uploadFile({
      url,
      filePath,
      name,
      timeout: 120000,
      formData,
      header,
      success(response) {
        const payload = safeJsonParse(response.data);
        if (response.statusCode >= 200 && response.statusCode < 300) {
          resolve(payload);
          return;
        }

        reject(new Error(normalizeError(payload, `文件上传失败：${url}`)));
      },
      fail(error) {
        reject(new Error(`${normalizeError(error, '文件上传失败')}：${url}`));
      },
    });
  });
}

function requestStream({
  baseUrl = DEFAULT_BASE_URL,
  path,
  method = 'POST',
  data = null,
  header = {},
  onChunk = null,
}) {
  return new Promise((resolve, reject) => {
    let responseData = '';
    const url = joinUrl(baseUrl, path);

    const requestTask = wx.request({
      url,
      method,
      data,
      timeout: 120000,
      enableChunked: true,
      responseType: 'arraybuffer',
      header: {
        'content-type': 'application/json',
        ...header,
      },
      success(response) {
        if (response.statusCode < 200 || response.statusCode >= 300) {
          reject(new Error(normalizeError(safeJsonParse(response.data), `流式请求失败：${url}`)));
          return;
        }

        if (response.data) {
          responseData = response.data;
        }

        resolve({
          data: responseData,
          statusCode: response.statusCode,
        });
      },
      fail(error) {
        reject(new Error(`${normalizeError(error, '流式请求失败')}：${url}`));
      },
    });

    if (requestTask && typeof requestTask.onChunkReceived === 'function' && onChunk) {
      requestTask.onChunkReceived((chunkEvent) => {
        onChunk(chunkEvent.data);
      });
    }
  });
}

module.exports = {
  joinUrl,
  normalizeError,
  request,
  requestUrl,
  requestStream,
  uploadFile,
  uploadImage,
};
