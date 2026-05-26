const app = getApp();
const { extractOverallReport, extractSummaryCards, isTongueAnalysis } = require('../../utils/analysis');
const { DEFAULT_BASE_URL } = require('../../utils/config');

function buildRecentReport(workflowResult) {
  if (!workflowResult || !workflowResult.analyzeResult) {
    return null;
  }

  const analyzeResult = workflowResult.analyzeResult;
  const cards = extractSummaryCards(analyzeResult);
  const overall = extractOverallReport(analyzeResult, cards);
  const isTongue = isTongueAnalysis(analyzeResult);
  return {
    title: isTongue ? '舌象一期辅助分析' : '面象肤况辅助分析',
    summary: overall.summary || '辅助分析结果已生成。',
    tags: cards.slice(0, 3).map((item) => item.scoreText ? `${item.title} ${item.scoreText}` : item.title),
    hasReport: true,
  };
}

Page({
  data: {
    baseUrl: DEFAULT_BASE_URL,
    report: null,
    overviewScore: '--',
    overviewTags: ['气虚倾向', '湿热关注'],
    taskCards: [
      {
        key: 'tongue',
        title: '舌象分析',
        desc: '拍舌面，查看舌色、苔质、津润',
        iconClass: 'mini-icon--green',
      },
      {
        key: 'face',
        title: '面象分析',
        desc: '上传面部图，获得肤况建议',
        iconClass: 'mini-icon--blue',
      },
      {
        key: 'constitution',
        title: '九大体质',
        desc: '问卷结合图像，辅助判断倾向',
        iconClass: 'mini-icon--amber',
      },
      {
        key: 'chat',
        title: '快速问询',
        desc: '带着报告直接问 AI',
        iconClass: 'mini-icon--coral',
      },
    ],
  },

  onShow() {
    const workflowResult = app.getWorkflowResult();
    const report = buildRecentReport(workflowResult);
    const overallScore =
      report && workflowResult && workflowResult.analyzeResult
        ? extractOverallReport(workflowResult.analyzeResult).totalScoreText
        : '--';

    this.setData({
      baseUrl: app.globalData.baseUrl || DEFAULT_BASE_URL,
      report,
      overviewScore: overallScore && overallScore !== '--' ? overallScore : '72',
    });
  },

  openTask(event) {
    const key = event.currentTarget.dataset.key;
    if (key === 'tongue' || key === 'face') {
      wx.navigateTo({
        url: `/pages/diagnosis/index?mode=${key}`,
      });
      return;
    }

    if (key === 'constitution' || key === 'chat') {
      wx.reLaunch({
        url: '/pages/ai/index',
      });
    }
  },

  openReport() {
    wx.navigateTo({
      url: '/pages/result/result',
    });
  },

  openPlan() {
    wx.reLaunch({
      url: '/pages/plan/index',
    });
  },
});
