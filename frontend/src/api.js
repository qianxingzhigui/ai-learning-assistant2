// v1.1 API 封装：fetch + JSON + Bearer 令牌 + 统一错误提示
const TOKEN_KEY = 'lp_token'
const USER_KEY = 'lp_user'

export const auth = {
  token: () => localStorage.getItem(TOKEN_KEY) || '',
  username: () => {
    try { return JSON.parse(localStorage.getItem(USER_KEY) || 'null')?.username || '' } catch { return '' }
  },
  user: () => {
    try { return JSON.parse(localStorage.getItem(USER_KEY) || 'null') } catch { return null }
  },
  save: (token, user) => {
    localStorage.setItem(TOKEN_KEY, token)
    localStorage.setItem(USER_KEY, JSON.stringify(user))
  },
  clear: () => {
    localStorage.removeItem(TOKEN_KEY)
    localStorage.removeItem(USER_KEY)
  },
}

async function request(url, options = {}) {
  const headers = { ...(options.headers || {}) }
  if (!(options.body instanceof FormData)) headers['Content-Type'] = 'application/json'
  const token = auth.token()
  if (token) headers['Authorization'] = `Bearer ${token}`

  const resp = await fetch(url, { ...options, headers })
  if (!resp.ok) {
    // 令牌失效/缺失：清掉本地登录态，回登录页
    if (resp.status === 401 && !url.startsWith('/api/auth/')) {
      auth.clear()
      window.location.assign('/login')
      throw new Error('登录已过期，请重新登录')
    }
    let detail = `请求失败 (${resp.status})`
    try {
      const data = await resp.json()
      if (data.detail) detail = typeof data.detail === 'string' ? data.detail : JSON.stringify(data.detail)
    } catch { /* ignore */ }
    throw new Error(detail)
  }
  return resp.json()
}

// 带 Authorization 的文件下载（浏览器 <a> 下载带不了请求头，所以走 fetch+blob）
async function downloadFile(url, filename) {
  const resp = await fetch(url, { headers: { Authorization: `Bearer ${auth.token()}` } })
  if (!resp.ok) {
    let detail = `导出失败 (${resp.status})`
    try { const d = await resp.json(); if (d.detail) detail = d.detail } catch { /* ignore */ }
    throw new Error(detail)
  }
  const blob = await resp.blob()
  const a = document.createElement('a')
  a.href = URL.createObjectURL(blob)
  a.download = filename
  document.body.appendChild(a)
  a.click()
  a.remove()
  URL.revokeObjectURL(a.href)
}

export default {
  // 认证
  register: (body) => request('/api/auth/register', { method: 'POST', body: JSON.stringify(body) }),
  login: (body) => request('/api/auth/login', { method: 'POST', body: JSON.stringify(body) }),
  me: () => request('/api/auth/me'),
  // 资料
  listMaterials: (course) => request(`/api/materials${course ? `?course=${encodeURIComponent(course)}` : ''}`),
  getMaterial: (id) => request(`/api/materials/${id}`),
  uploadMaterial: (file, course) => {
    const fd = new FormData()
    fd.append('file', file)
    fd.append('course', course)
    return request('/api/materials/upload', { method: 'POST', body: fd })
  },
  deleteMaterial: (id) => request(`/api/materials/${id}`, { method: 'DELETE' }),
  // 知识点
  listKps: (course) => request(`/api/knowledge-points${course ? `?course=${encodeURIComponent(course)}` : ''}`),
  createKp: (body) => request('/api/knowledge-points', { method: 'POST', body: JSON.stringify(body) }),
  updateKp: (id, body) => request(`/api/knowledge-points/${id}`, { method: 'PUT', body: JSON.stringify(body) }),
  deleteKp: (id) => request(`/api/knowledge-points/${id}`, { method: 'DELETE' }),
  // 任务
  listTasks: () => request('/api/tasks'),
  createTask: (body) => request('/api/tasks', { method: 'POST', body: JSON.stringify(body) }),
  setTaskStatus: (id, status) => request(`/api/tasks/${id}/status`, { method: 'PATCH', body: JSON.stringify({ status }) }),
  deleteTask: (id) => request(`/api/tasks/${id}`, { method: 'DELETE' }),
  dueSoon: (days = 3) => request(`/api/tasks/due-soon?days=${days}`),
  // 练习与错题
  generateQuestions: (body) => request('/api/questions/generate', { method: 'POST', body: JSON.stringify(body) }),
  listQuestions: (kpId) => request(`/api/questions${kpId ? `?kp_id=${kpId}` : ''}`),
  submitPractice: (body) => request('/api/practice/submit', { method: 'POST', body: JSON.stringify(body) }),
  wrongBook: (onlyUnresolved = true) => request(`/api/wrong-book?only_unresolved=${onlyUnresolved}`),
  updateWrong: (id, body) => request(`/api/wrong-book/${id}`, { method: 'PATCH', body: JSON.stringify(body) }),
  exportWrongCsv: (onlyUnresolved = false) =>
    downloadFile(`/api/wrong-book/export/csv?only_unresolved=${onlyUnresolved}`, `错题本-${new Date().toISOString().slice(0, 10)}.csv`),
  openWrongPrint: (onlyUnresolved = false) =>
    window.open(`/api/wrong-book/print?token=${encodeURIComponent(auth.token())}&only_unresolved=${onlyUnresolved}`, '_blank'),
  // 全局搜索
  search: (q) => request(`/api/search?q=${encodeURIComponent(q)}`),
  // 统计
  statsOverview: () => request('/api/stats/overview'),
  // AI
  aiHealth: () => request('/api/health'),
  aiTest: () => request('/api/ai/test', { method: 'POST' }),
  aiConfig: () => request('/api/config/ai'),
  saveAiConfig: (body) => request('/api/config/ai', { method: 'POST', body: JSON.stringify(body) }),
  aiSummarize: (body) => request('/api/ai/summarize', { method: 'POST', body: JSON.stringify(body) }),
  aiAsk: (body) => request('/api/ai/ask', { method: 'POST', body: JSON.stringify(body) }),
  aiWeakAnalysis: (body) => request('/api/ai/weak-analysis', { method: 'POST', body: JSON.stringify(body) }),
  // 跨机同步
  syncConfig: (body) => request('/api/sync/config', { method: 'POST', body: JSON.stringify(body) }),
  syncExport: () => request('/api/sync/export'),
  syncImport: (payload) => request('/api/sync/import', { method: 'POST', body: JSON.stringify(payload) }),
  syncPush: () => request('/api/sync/push', { method: 'POST' }),
  syncPull: () => request('/api/sync/pull', { method: 'POST' }),
  // 手动备份：下载/读取 JSON 文件（<a> 下载带不了请求头，走 fetch+blob）
  downloadSyncFile: () =>
    downloadFile('/api/sync/export', `学习数据备份-${new Date().toISOString().slice(0, 10)}.json`),
}
