import { createApp } from 'vue'
import { createRouter, createWebHistory } from 'vue-router'
import ElementPlus from 'element-plus'
import 'element-plus/dist/index.css'
import zhCn from 'element-plus/es/locale/lang/zh-cn'
import App from './App.vue'
import api from './api'

// v1.1 朴素路由：视图级组件 + 登录守卫（v2 引入 Pinia 全局状态）
import Dashboard from './views/Dashboard.vue'
import Materials from './views/Materials.vue'
import KnowledgePoints from './views/KnowledgePoints.vue'
import Tasks from './views/Tasks.vue'
import Practice from './views/Practice.vue'
import WrongBook from './views/WrongBook.vue'
import AiAssistant from './views/AiAssistant.vue'
import Sync from './views/Sync.vue'
import Login from './views/Login.vue'
import { auth } from './api'

const router = createRouter({
  history: createWebHistory(),
  routes: [
    { path: '/login', name: 'login', component: Login, meta: { title: '登录', public: true } },
    { path: '/', name: 'dashboard', component: Dashboard, meta: { title: '学习看板' } },
    { path: '/materials', name: 'materials', component: Materials, meta: { title: '课程资料' } },
    { path: '/knowledge', name: 'knowledge', component: KnowledgePoints, meta: { title: '知识点' } },
    { path: '/tasks', name: 'tasks', component: Tasks, meta: { title: '学习任务' } },
    { path: '/practice', name: 'practice', component: Practice, meta: { title: '练习' } },
    { path: '/wrong-book', name: 'wrongBook', component: WrongBook, meta: { title: '错题本' } },
    { path: '/ai', name: 'ai', component: AiAssistant, meta: { title: 'AI 助手' } },
    { path: '/sync', name: 'sync', component: Sync, meta: { title: '数据同步' } },
  ],
})

// 未登录一律回登录页；已登录访问 /login 则直接进看板
router.beforeEach((to) => {
  if (to.meta.public) return auth.token() ? { path: '/' } : true
  return auth.token() ? true : { path: '/login' }
})

router.afterEach((to) => {
  document.title = to.meta.title ? `${to.meta.title} · AI 智能学习助手` : 'AI 智能学习助手'
})

// 暴露 api 给各视图（v1 朴素做法；v2 改为模块化注入）
const app = createApp(App)
app.config.globalProperties.$api = api
app.use(router)
app.use(ElementPlus, { locale: zhCn })
app.mount('#app')
