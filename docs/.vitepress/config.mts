import { defineConfig, type DefaultTheme } from 'vitepress'
import { writeFileSync } from 'node:fs'
import { resolve } from 'node:path'

// 顶部导航：与 Datawhale 使用相同的组织方式
const nav: DefaultTheme.NavItem[] = [
  {
    text: '首页',
    link: '/'
  },
  {
    text: 'Part I',
    link: '/fundamentals/Mathematics'
  },
  {
    text: 'Part II',
    link: '/fundamentals/Paged-Attention'
  }
]

// 左侧目录
const sidebar: DefaultTheme.SidebarItem[] = [
  {
    text: 'Part I — Foundations',
    collapsed: false,
    items: [
      {
        text: '第 1 章 Mathematics',
        link: '/fundamentals/Mathematics'
      }
    ]
  },
  {
    text: 'Part II — Inference Techniques',
    collapsed: false,
    items: [
      {
        text: '第 1 章 Paged Attention',
        link: '/inference/Paged-Attention'
      },
      {
        text: '第 2 章 Continuous Batching',
        link: '/inference/Continuous-Batching'
      },
      {
        text: '第 3 章 Prefix Caching',
        link: '/inference/Prefix-Caching'
      }
    ]
  }
]

const homeContent = [
  '# Yifan\'s AI Infra Notes',
  '',
  'Welcome to my AI Infrastructure learning notes!',
  '',
  'This site documents my exploration of LLM inference, GPU computing, and distributed systems.',
  '',
  '## 📚 Learning Topics',
  ''
]

for (const section of sidebar) {
  if (!section.text || !section.items) continue

  homeContent.push(`### ${section.text}`, '')

  for (const item of section.items) {
    if (item.text && item.link) {
      homeContent.push(`- [${item.text}](${item.link})`)
    }
  }

  homeContent.push('')
}

writeFileSync(
  resolve(process.cwd(), 'docs/index.md'),
  homeContent.join('\n'),
  'utf-8'
)

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
