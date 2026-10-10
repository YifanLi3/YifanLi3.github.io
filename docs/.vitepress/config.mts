import { defineConfig, type DefaultTheme } from 'vitepress'

// 顶部导航：与 Datawhale 使用相同的组织方式
const nav: DefaultTheme.NavItem[] = [
  { text: '首页', link: '/' },
  { text: 'Part I', link: '/fundamentals/arithmetic-intensity' }
]

// 左侧目录
const sidebar: DefaultTheme.SidebarItem[] = [
  {
    text: 'Part I — Foundations',
    collapsed: false,
    items: [
      {
        text: '第 1 章 Arithmetic Intensity',
        link: '/fundamentals/arithmetic-intensity'
      }
    ]
  }
]

export default defineConfig({
  title: "Yifan's AI Infra Notes",
  description: 'AI Infrastructure Learning Notes',

  lang: 'zh-CN',
  base: '/',
  cleanUrls: true,

  markdown: {
    math: true
  },

  themeConfig: {
    nav,
    sidebar,

    search: {
      provider: 'local'
    },

    outline: {
      level: [2, 3],
      label: '本页目录'
    },

    docFooter: {
      prev: '上一页',
      next: '下一页'
    },

    returnToTopLabel: '返回顶部',
    sidebarMenuLabel: '菜单',
    darkModeSwitchLabel: '外观',
    lightModeSwitchTitle: '切换到浅色模式',
    darkModeSwitchTitle: '切换到深色模式',

    socialLinks: [
      {
        icon: 'github',
        link: 'https://github.com/YifanLi3/YifanLi3.github.io'
      }
    ]
  }
})
