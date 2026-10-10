
import { defineConfig } from 'vitepress'

export default defineConfig({
  title: "Yifan's AI Infra Notes",
  description: "Notes on LLM Inference and GPU Computing",
  lang: 'zh-CN',
  base: '/',
  cleanUrls: true,

  themeConfig: {
    siteTitle: "Yifan's AI Infra Notes",

    nav: [],

    sidebar: [
      {
        text: 'Part I: Fundamentals',
        collapsed: false,
        items: [
          {
            text: 'Arithmetic Intensity',
            link: '/fundamentals/Arithmetic-intensity'
          }
        ]
      }
    ],

    outline: {
      level: [2, 3],
      label: '本页目录'
    },

    sidebarMenuLabel: '目录',
    returnToTopLabel: '返回顶部',
    docFooter: {
      prev: '上一页',
      next: '下一页'
    },

    search: {
      provider: 'local'
    }
  },

  markdown: {
    math: true
  }
})
