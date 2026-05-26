const app = getApp();
const { extractOverallReport, extractSummaryCards } = require('../../utils/analysis');

function buildFocus(workflowResult) {
  if (!workflowResult || !workflowResult.analyzeResult) {
    return {
      title: '睡眠节律 · 清淡饮食 · 屏障护理',
      tags: ['气虚', '湿热', '肤况观察'],
      summary: '完成一次舌象或面象分析后，这里会自动更新为个性化健康方案。',
    };
  }

  const cards = extractSummaryCards(workflowResult.analyzeResult);
  const overall = extractOverallReport(workflowResult.analyzeResult, cards);
  return {
    title: cards.slice(0, 3).map((item) => item.title).join(' · ') || '健康管理建议',
    tags: cards.slice(0, 3).map((item) => item.scoreText || item.title),
    summary: overall.summary,
  };
}

Page({
  data: {
    focus: buildFocus(null),
    tasks: [
      { slot: '晨间', title: '温水、轻早餐，避免空腹咖啡', color: 'green' },
      { slot: '午后', title: '减少辛辣油炸，补充蔬菜和水分', color: 'amber' },
      { slot: '晚间', title: '23:30 前入睡，面部温和清洁', color: 'blue' },
    ],
    modules: [
      { title: '饮食', desc: '少辛辣 · 重规律', color: 'amber' },
      { title: '睡眠', desc: '固定入睡窗口', color: 'blue' },
      { title: '护肤', desc: '温和清洁保湿', color: 'green' },
      { title: '复测', desc: '3 天后再拍', color: 'coral' },
    ],
  },

  onShow() {
    this.setData({
      focus: buildFocus(app.getWorkflowResult()),
    });
  },

  askAi() {
    wx.reLaunch({
      url: '/pages/ai/index',
    });
  },
});
