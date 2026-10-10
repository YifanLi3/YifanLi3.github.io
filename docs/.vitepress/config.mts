import { defineConfig } from 'vitepress'

export default defineConfig({
  title: "Yifan's AI Infra Notes",
  description: 'AI Infrastructure Learning Notes',
  lang: 'zh-CN',

  base: '/',
  cleanUrls: true,

  themeConfig: {
    siteTitle: "Yifan's AI Infra Notes",

    // 不显示顶部 Part I、Part II 等导航标签
    nav: [],

    // 左侧完整章节目录
    sidebar: [
      {
        text: 'Part 0 — Introduction',
        collapsed: false,
        items: [
          { text: '首页', link: '/' }
        ]
      },
      {
        text: 'Part 1 — GPU Computing Fundamentals',
        collapsed: false,
        items: [
          {
            text: 'Arithmetic Intensity',
            link: '/fundamentals/arithmetic-intensity'
          }
        ]
      }
    ],

    // 右侧文章目录
    outline: {
      level: [2, 3],
      label: '本页目录'
    },

    // 搜索
    search: {
      provider: 'local'
    },

    // 上一章 / 下一章
    docFooter: {
      prev: '上一章',
      next: '下一章'
    },

    sidebarMenuLabel: '目录',
    returnToTopLabel: '返回顶部'
  },

  markdown: {
    math: true
  }
})
