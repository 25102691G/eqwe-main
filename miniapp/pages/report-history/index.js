const app = getApp();
const {
  extractOverallReport,
  extractSummaryCards,
  isTongueAnalysis,
} = require('../../utils/analysis');

function formatReportTime(timestamp) {
  const date = new Date(Number(timestamp || 0));
  if (Number.isNaN(date.getTime())) {
    return '';
  }

  const year = date.getFullYear();
  const month = String(date.getMonth() + 1).padStart(2, '0');
  const day = String(date.getDate()).padStart(2, '0');
  const hour = String(date.getHours()).padStart(2, '0');
  const minute = String(date.getMinutes()).padStart(2, '0');
  return `${year}-${month}-${day} ${hour}:${minute}`;
}

function buildReportItem(workflowResult) {
  const analyzeResult = workflowResult.analyzeResult || {};
  const cards = extractSummaryCards(analyzeResult);
  const overall = extractOverallReport(analyzeResult, cards);
  const isTongue = isTongueAnalysis(analyzeResult);
  const storage = analyzeResult.storage || {};
  const metadata = analyzeResult.metadata || {};
  return {
    reportId: workflowResult.reportId || '',
    title: isTongue ? '舌象一期辅助分析' : '面象肤况辅助分析',
    typeLabel: isTongue ? '舌象' : '面象',
    typeClass: isTongue ? 'green' : 'blue',
    time: formatReportTime(workflowResult.createdAt),
    folder: storage.folder || metadata.folder || '',
    summary: overall.summary || '辅助分析结果已生成。',
    scoreText: overall.totalScoreText || '--',
  };
}

Page({
  data: {
    reports: [],
    totalCount: 0,
  },

  onShow() {
    const reports = app.getWorkflowResults().map(buildReportItem);
    this.setData({
      reports,
      totalCount: reports.length,
    });
  },

  openReport(event) {
    const reportId = event.currentTarget.dataset.reportId || '';
    if (!reportId) {
      wx.showToast({
        title: '报告不存在',
        icon: 'none',
      });
      return;
    }

    wx.navigateTo({
      url: `/pages/result/result?reportId=${encodeURIComponent(reportId)}`,
    });
  },

  goDiagnosis() {
    wx.navigateTo({
      url: '/pages/diagnosis/index',
    });
  },
});
