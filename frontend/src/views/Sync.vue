<template>
  <div>
    <!-- 云端同步 -->
    <el-card class="page-card">
      <template #header><span>跨设备同步（GitHub 私库当中转）</span></template>

      <el-form label-width="110px" style="max-width:560px">
        <el-form-item label="仓库">
          <el-input v-model="cfg.repo" placeholder="owner/name，如 qianxingzhigui/ai-learning-assistant2" />
        </el-form-item>
        <el-form-item label="分支">
          <el-input v-model="cfg.branch" placeholder="develop" style="width:200px" />
        </el-form-item>
        <el-form-item label="访问令牌">
          <el-input v-model="cfg.token" type="password" show-password
                    placeholder="GitHub PAT，需要对该仓库有写权限（保存后不回显）" />
        </el-form-item>
        <el-form-item>
          <el-button type="primary" :loading="saving" @click="saveCfg">保存配置</el-button>
          <span class="muted" style="margin-left:12px">
            令牌只存在本机 data/sync.json，不会上传。在 GitHub → Settings → Developer settings →
            Personal access tokens 创建，勾选 repo 权限即可。
          </span>
        </el-form-item>
      </el-form>

      <el-divider />

      <div style="display:flex;gap:12px;align-items:center;margin-bottom:12px">
        <el-button type="success" :loading="pushing" @click="doPush">↑ 上传我的数据</el-button>
        <el-button type="primary" :loading="pulling" @click="doPull">↓ 拉取并合并他人数据</el-button>
        <span class="muted">
          每人各推一份 sync/用户名.json，拉取时按 uuid 合并、只增不覆盖，不会互相踩数据。
        </span>
      </div>

      <el-alert v-if="pushResult" type="success" :closable="false" style="margin-bottom:10px">
        已上传 {{ pushResult.pushed }}（提交 {{ pushResult.commit }}）：
        {{ fmtCounts(pushResult.rows) }}
      </el-alert>
      <el-alert v-if="pullResult" type="success" :closable="false">
        <div>合并了 {{ pullResult.pulled }} 份数据：</div>
        <div v-for="d in pullResult.detail" :key="d.file" style="font-size:12px;margin-top:4px">
          {{ d.file }} → {{ d.error || fmtCounts(d.merged) }}
        </div>
        <div v-if="pullResult.note" style="font-size:12px">{{ pullResult.note }}</div>
      </el-alert>
    </el-card>

    <!-- 手动备份 -->
    <el-card class="page-card">
      <template #header><span>手动备份（不依赖 GitHub，导出成文件）</span></template>
      <div style="display:flex;gap:12px;align-items:center">
        <el-button :loading="exporting" @click="doExport">导出全部数据为 JSON</el-button>
        <el-button @click="$refs.file.click()">从 JSON 文件导入</el-button>
        <input ref="file" type="file" accept=".json,application/json"
               style="display:none" @change="doImport" />
      </div>
      <el-alert v-if="importResult" type="success" :closable="false" style="margin-top:12px">
        导入完成，合并行数：{{ fmtCounts(importResult.merged) }}
      </el-alert>
      <p class="muted" style="margin-top:12px">
        说明：导入按 uuid 去重合并，同一份数据重复导入不会产生重复行；两台电脑互相导入对方的备份文件即可完成一次同步。
      </p>
    </el-card>
  </div>
</template>

<script setup>
import { ref } from 'vue'
import { ElMessage } from 'element-plus'
import api from '../api'

const cfg = ref({ repo: '', branch: 'develop', token: '' })
const saving = ref(false)
const pushing = ref(false)
const pulling = ref(false)
const exporting = ref(false)
const pushResult = ref(null)
const pullResult = ref(null)
const importResult = ref(null)

async function saveCfg() {
  saving.value = true
  try {
    await api.syncConfig({
      repo: cfg.value.repo,
      branch: cfg.value.branch || 'develop',
      token: cfg.value.token,
    })
    ElMessage.success('同步配置已保存')
    cfg.value.token = ''
  } catch (e) {
    ElMessage.error(e.message)
  } finally {
    saving.value = false
  }
}

async function doPush() {
  pushing.value = true
  pushResult.value = null
  try {
    pushResult.value = await api.syncPush()
    ElMessage.success('上传完成')
  } catch (e) {
    ElMessage.error(e.message)
  } finally {
    pushing.value = false
  }
}

async function doPull() {
  pulling.value = true
  pullResult.value = null
  try {
    pullResult.value = await api.syncPull()
    ElMessage.success(pullResult.value.pulled ? '拉取合并完成' : '远端暂无他人数据')
  } catch (e) {
    ElMessage.error(e.message)
  } finally {
    pulling.value = false
  }
}

async function doExport() {
  exporting.value = true
  try {
    await api.downloadSyncFile()
    ElMessage.success('备份文件已下载')
  } catch (e) {
    ElMessage.error(e.message)
  } finally {
    exporting.value = false
  }
}

async function doImport(ev) {
  const file = ev.target.files?.[0]
  ev.target.value = '' // 允许重复选同一个文件
  if (!file) return
  try {
    const payload = JSON.parse(await file.text())
    if (!payload || typeof payload !== 'object' || !payload.users) {
      throw new Error('这不是学习助手的备份文件（缺少 users 字段）')
    }
    importResult.value = await api.syncImport(payload)
    ElMessage.success('导入完成')
  } catch (e) {
    ElMessage.error(e.message)
  }
}

function fmtCounts(rows) {
  if (!rows) return ''
  const names = {
    users: '用户', materials: '资料', knowledge_points: '知识点', questions: '题目',
    practice_records: '练习', wrong_book: '错题', tasks: '任务',
  }
  return Object.entries(rows)
    .filter(([, n]) => n > 0)
    .map(([k, n]) => `${names[k] || k} ${n}`)
    .join('，') || '空数据'
}
</script>
