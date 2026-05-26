Component({
  properties: {
    active: {
      type: String,
      value: 'home',
    },
  },

  data: {
    items: [
      { key: 'home', label: '首页', url: '/pages/index/index' },
      { key: 'plan', label: '健康方案', url: '/pages/plan/index' },
      { key: 'ai', label: 'AI问诊', url: '/pages/ai/index' },
      { key: 'store', label: '商城', url: '/pages/store/index' },
      { key: 'mine', label: '我的', url: '/pages/mine/index' },
    ],
  },

  methods: {
    switchPage(event) {
      const url = event.currentTarget.dataset.url;
      const key = event.currentTarget.dataset.key;
      if (!url || key === this.properties.active) {
        return;
      }
      wx.reLaunch({ url });
    },
  },
});
