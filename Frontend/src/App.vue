<script setup>
import { ref, watch, onMounted, computed } from 'vue'
import ToastComponent from './Toast.vue'

defineProps({
  projectName: {
    type: String,
    default: 'Gesturesphone'
  },
})

/* ===== State ===== */
const activeTab = ref('main')
const displayMode = ref('partial')
const isRuninngProgramm = ref(false)
const cameraUrl = '/api/camera'

const aiStatus = ref('not_ready')
const isAiProcessing = ref(false)

const appVersion = ref('not loaded')
fetch(`https://api.github.com/repos/ogled/Gesturesphone/commits?per_page=1`)
  .then(res => res.json())
  .then(commits => {
    appVersion.value = commits[0].sha.slice(0, 7)
  })

const cpuLoad = ref(0)
const ramLoad = ref(0)
const fps = ref(0)

/* ===== Gestures ===== */
const gestures = ref([])
const gestureHistory = ref([])
const isGesturesLoading = ref(false)
const gesturesError = ref(null)

const sortedGestures = computed(() =>
  [...gestures.value].sort((a, b) => b.confidence - a.confidence)
)

const topGestures = computed(() => sortedGestures.value.slice(0, 3))

/* ===== Lifecycle ===== */
onMounted(() => {
  setInterval(checkAiStatus, 2000)
  checkAiStatus()
  const saved = localStorage.getItem('displayMode')
  if (saved) displayMode.value = saved

  setInterval(loadUsageVals, 500)
})

async function startProgram() {
  const res = await fetch('/api/start')
  const data = await res.json()

  if (res.status === 200) {
    isRuninngProgramm.value = true
    setInterval(() => {
      loadGestures()
      loadGestureHistory()
    }, 500)
  } else {
    window.showToast('Error ' + res.status + ': ' + data.detail, 'error')
  }
}

async function checkAiStatus() {
  try {
    const res = await fetch('/api/ai-get-status')
    if (!res.ok) return
    
    const data = await res.json()
    aiStatus.value = data.status
  } catch (e) {
    console.error('AI status check failed:', e)
    aiStatus.value = 'error'
  }
}
async function sendToAI() {
  if (aiStatus.value !== 'ready' || isAiProcessing.value) return
  
  isAiProcessing.value = true

  try {
    const startRes = await fetch('/api/ai-corect-gestures')

    if (!startRes.ok) throw new Error('Failed to start AI processing')
    
    const pollInterval = setInterval(async () => {
      try {
        const res = await fetch('/api/get-ai-corect-text')
        if (!res.ok) return
        const data = await res.json()
        if (!data.busy) {
          clearInterval(pollInterval)
          isAiProcessing.value = false
          window.showToast(data.text, 'ai')
          checkAiStatus()
        }
      } catch (e) {
        window.showToast('Ошибка получения результата от ИИ')
        console.error('Polling failed:', e)
        clearInterval(pollInterval)
        isAiProcessing.value = false
      }
    }, 500)
    
  } catch (e) {
    console.error('AI processing failed:', e)
    window.showToast("AI processing failed")
    isAiProcessing.value = false
  }
}

async function loadGestures() {
  try {
    const res = await fetch('/api/gestures')
    if (!res.ok) return

    const data = await res.json()
    const raw = data?.gestures ?? {}

    gestures.value = Object.entries(raw).map(([name, confidence]) => ({
      name,
      confidence,
    }))
  } catch {
    gestures.value = []
  }
}
async function loadGestureHistory() {
  try {
    const res = await fetch('/api/gesture-history')
    if (!res.ok) return

    const data = await res.json()
    gestureHistory.value = data.history
  } catch (e) {
    console.error(e)
  }
}
async function loadUsageVals() {
  const res = await fetch('/api/getUsageVals')
  if (!res.ok) return

  const data = await res.json()
  cpuLoad.value = data.CPU
  ramLoad.value = data.RAM
  fps.value = data.FPS
}

watch(displayMode, v => localStorage.setItem('displayMode', v))
</script>

<template>
  <div class="app">
    <!-- Header -->
    <nav class="tabs">
      <span class="projectName">{{ projectName }}</span>

      <button class="tab" :class="{ active: activeTab === 'main' }" @click="activeTab = 'main'">
        Главная
      </button>

      <button class="tab" :class="{ active: activeTab === 'settings' }" @click="activeTab = 'settings'">
        Настройки
      </button>
    </nav>

    <main class="content">
      <section v-if="activeTab === 'main'" class="fill">
        <div v-if="!isRuninngProgramm" class="StartingProgramm">
          <button class="start-btn" @click="startProgram">▶ Запуск</button>
        </div>

        <div v-else class="layout">
          <!-- Camera -->
          <div class="main">
            <img :src="cameraUrl" class="camera-feed" />
          </div>

          <!-- Right panel -->
          <div class="gestures-panel">
            <!-- Top -->
            <section class="panel-card">
              <h3>Наиболее вероятные</h3>

              <div v-for="(g, i) in topGestures" :key="g.name" class="top-gesture">
                <div class="rank-badge">{{ i + 1 }}</div>

                <div class="info">
                  <div class="name">{{ g.name }}</div>
                  <div class="bar">
                    <div class="bar-fill" :style="{ width: g.confidence + '%' }" />
                  </div>
                </div>

                <div class="percent">{{ g.confidence }}%</div>
              </div>
            </section>

            <!-- HISTORY -->
            <section class="panel-card history">
              <div class="history-header">
                <h4>История жестов</h4>
                <button 
                  class="ai-button"
                  :class="{ 
                    'ready': aiStatus === 'ready',
                    'disabled': aiStatus !== 'ready',
                    'processing': isAiProcessing
                  }"
                  @click="sendToAI"
                  :disabled="aiStatus !== 'ready' || isAiProcessing"
                >
                  <span v-if="isAiProcessing" class="spinner">↻</span>
                  <span v-else>Завершить высказывание</span>
                </button>
              </div>
              <div class="history-body">
                <span
                  v-for="(g, i) in gestureHistory"
                  :key="i"
                  class="gesture-word"
                  :class="{ latest: i === gestureHistory.length - 1 }"
                >
                  {{ g }}
                </span>

                <div v-if="gestureHistory.length === 0" class="gesture-placeholder">
                  Пока жесты не распознаны
                </div>
              </div>
            </section>
          </div>
        </div>
      </section>

      <section v-if="activeTab === 'settings'" class="settings">
        <label>
          Режим отображения
          <select v-model="displayMode">
            <option value="partial">Частичный</option>
            <option value="full">Полный</option>
          </select>
        </label>
      </section>
    </main>

    <footer class="status-bar">
      <span>{{ projectName }} • {{ appVersion }}</span>
      <div>
        CPU: <b>{{ cpuLoad }}%</b> |
        RAM: <b>{{ ramLoad }}%</b> |
        FPS: <b>{{ fps }}</b>
      </div>
    </footer>
  </div>
  
  <ToastComponent />
</template>

<style scoped>

.tabs {
  height: 50px;
  display: flex;
  align-items: center;
  padding: 0 1rem;
  background: #f9fafb;
  border-bottom: 1px solid #e5e7eb;
}

.projectName {
  margin-right: auto;
  font-size: 1.2rem;
  font-weight: 500;
}

.tab {
  margin-left: 0.5rem;
  padding: 0.4rem 0.9rem;
  border-radius: 6px;
  border: none;
  background: transparent;
  cursor: pointer;
}

.tab.active {
  background: white;
  border: 1px solid #d1d5db;
}

.content {
  flex: 1;
  padding: 1rem;
  overflow: hidden;
}

.fill {
  height: 100%;
}

.top-gestures {
  margin-bottom: 1rem;
}

.top-gesture {
  display: grid;
  grid-template-columns: 32px 1fr 48px;
  align-items: center;
  gap: 0.6rem;
  margin-bottom: 0.6rem;
}

.rank-badge {
  width: 28px;
  height: 28px;
  border-radius: 50%;
  background: #eef2ff;
  color: #3730a3;
  font-weight: 600;
  display: flex;
  align-items: center;
  justify-content: center;
}

.name {
  font-size: 0.95rem;
  font-weight: 500;
}

.bar {
  height: 6px;
  background: #e5e7eb;
  border-radius: 4px;
}

.bar-fill {
  height: 100%;
  background: linear-gradient(90deg, #22c55e, #4ade80);
}

.percent {
  font-size: 0.8rem;
}


/* ===== Full ===== */
.full-mode {
  height: 100%;
  background: white;
  border-radius: 12px;
  display: flex;
}

.placeholder {
  margin: auto;
  color: #9ca3af;
}

/* ===== Status bar ===== */
.status-bar {
  height: 28px;
  padding: 0 0.75rem;
  display: flex;
  align-items: center;
  justify-content: space-between;
  background: #f9fafb;
  border-top: 1px solid #e5e7eb;
  font-size: 0.75rem;
  color: #6b7280;
}

.right-aligned {
  display: flex;
  align-items: center;
  
}
.StartingProgramm {
  height: 100%;
  display: flex;
  align-items: center;
  justify-content: center;
}

.start-btn,
.stop-btn {
  padding: 0.8rem 1.6rem;
  font-size: 1rem;
  border-radius: 10px;
  border: none;
  cursor: pointer;
  background: #22c55e;
  color: white;
  transition: transform 0.07s ease, opacity 0.2s;
}

.start-btn:active,
.stop-btn:active {
  transform: scale(0.95);
}

.stop-btn {
  background: #ef4444;
  margin-bottom: 0.5rem;
}

/* ===== Gesture history ===== */
.gesture-history {
  border-top: 1px solid #e5e7eb;
  padding-top: 0.6rem;
  margin-top: 0.6rem;
}

.gesture-history h4 {
  font-size: 0.8rem;
  font-weight: 500;
  color: #6b7280;
  margin-bottom: 0.4rem;
}

.history-row {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  max-height: 80px;
  overflow-y: auto;
}

.gesture-chip {
  padding: 4px 10px;
  font-size: 0.75rem;
  border-radius: 999px;
  background: #f3f4f6;
  color: #374151;
  white-space: nowrap;
}

.gesture-chip.latest {
  background: #e0f2fe;
  color: #0369a1;
  font-weight: 500;
}

.history-empty {
  font-size: 0.75rem;
  color: #9ca3af;
}

/* ===== Base ===== */
.app {
  height: 100dvh;
  display: flex;
  flex-direction: column;
  background: #f6f7f9;
  font-family: system-ui;
}

/* ===== Layout ===== */
.layout {
  height: 100%;
  display: grid;
  grid-template-columns: 1fr 420px;
  gap: 1rem;
}

/* ===== Camera ===== */
.main {
  background: black;
  border-radius: 12px;
  overflow: hidden;
  display: flex;
}

.camera-feed {
  width: 100%;
  object-fit: contain;
}

/* ===== Right panel ===== */
.gestures-panel {
  display: flex;
  flex-direction: column;
  gap: 0.8rem;
}

/* ===== Cards ===== */
.panel-card {
  background: white;
  border-radius: 12px;
  padding: 0.9rem;
}

.panel-card h3,
.panel-card h4 {
  margin-bottom: 0.6rem;
  font-weight: 600;
}

.scroll {
  flex: 1;
  overflow-y: auto;
}

/* ===== History ===== */
.history {
  background: #f9fafb;
  border: 1px dashed #d1d5db;
}

.gesture-word {
  opacity: 0;
  animation: fadeIn 0.25s forwards;
}

.gesture-word.latest {
  color: #2563eb;
  font-weight: 600;
}

.gesture-placeholder {
  color: #9ca3af;
  font-size: 0.95rem;
}

/* ===== Anim ===== */
@keyframes fadeIn {
  to {
    opacity: 1;
  }
}

@media (max-width: 768px) {

  .content {
    padding: 0.5rem;
    overflow-y: auto;
  }

  .layout {
    grid-template-columns: 1fr;
    gap: 0.6rem;
  }

  .main {
    aspect-ratio: 4 / 3;
    border-radius: 10px;
  }

  .gestures-panel {
    gap: 0.6rem;
  }

  .panel-card {
    padding: 0.7rem;
  }

  .history-body {
    font-size: 1rem;
  }

  .tabs {
    position: sticky;
    top: 0;
    z-index: 10;
  }

  .status-bar {
    display: none;
  }
}

/* ===== Ai button ===== */
.history-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 0.6rem;
  flex-wrap: wrap;
  gap: 0.5rem;
}

.ai-button {
  padding: 0.6rem 1rem;
  border-radius: 8px;
  border: none;
  font-size: 0.75rem;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.2s;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  display: flex;
  align-items: center;
  gap: 0.3rem;
  white-space: nowrap;
}

.ai-button.ready:hover {
  transform: translateY(-1px);
  box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
}

.ai-button.disabled {
  background: #9ca3af;
  cursor: not-allowed;
  opacity: 0.7;
}

.ai-button.processing {
  background: #6b7280;
}

.ai-button .spinner {
  animation: spin 1s linear infinite;
}

/* Анимации */
@keyframes spin {
  from { transform: rotate(0deg); }
  to { transform: rotate(360deg); }
}

@media (max-width: 900px) {
  .layout {
    display: flex;
    flex-direction: column;
    height: auto;
    min-height: 0;
    gap: 0.6rem;
  }
  
  .main {
    height: 280px;
    min-height: 280px;
    flex: none;
  }
  
  .gestures-panel {
    width: 100%;
    min-width: 100%;
    height: auto;
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    grid-template-rows: auto auto;
    gap: 0.6rem;
    overflow: visible;
  }

  .panel-card:first-child {
    grid-column: 1 / -1;
  }
  
  .panel-card.scroll {
    grid-column: 1;
    min-height: 150px;
    max-height: 180px;
  }

  .panel-card.history {
    grid-column: 2;
    min-height: 150px;
    max-height: 180px;
    grid-row: 2;
  }
  
  .history-header {
    display: flex;
    flex-direction: column;
    align-items: stretch;
    gap: 0.4rem;
    margin-bottom: 0.5rem;
  }
  
  .history-header h4 {
    width: 100%;
    text-align: center;
  }
  
  .ai-button {
    width: 100%;
    padding: 0.5rem;
    font-size: 0.8rem;
  }
  
  .history-body {
    height: calc(100% - 70px);
    overflow-y: auto;
    overflow-x: hidden;
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    align-content: flex-start;
  }
  
  .gesture-word {
    white-space: normal;
    word-break: break-word;
    max-width: 100%;
  }
}

@media (max-width: 700px) {
  .gestures-panel {
    grid-template-columns: 1fr;
    grid-template-rows: auto auto auto;
  }
  
  .panel-card:first-child {
    grid-column: 1;
    grid-row: 1;
  }
  
  .panel-card.scroll {
    grid-column: 1;
    grid-row: 2;
    min-height: 120px;
    max-height: 150px;
  }
  
  .panel-card.history {
    grid-column: 1;
    grid-row: 3;
    min-height: 140px;
    max-height: 160px;
  }
  
  .history-header {
    flex-direction: row;
    align-items: center;
  }
  
  .history-header h4 {
    width: auto;
    flex: 1;
    text-align: left;
  }
  
  .ai-button {
    width: auto;
    min-width: 160px;
  }
}

@media (max-width: 480px) {
  .tabs {
    height: 40px;
    padding: 0 0.3rem;
  }
  
  .projectName {
    font-size: 0.9rem;
  }
  
  .tab {
    padding: 0.2rem 0.4rem;
    font-size: 0.8rem;
  }
  
  .content {
    padding: 0.2rem;
  }
  
  .main {
    height: 220px;
    min-height: 220px;
  }
  
  .gestures-panel {
    gap: 0.4rem;
  }
  
  .panel-card {
    padding: 0.5rem;
  }
  
  .ai-button {
    font-size: 0.7rem;
    padding: 0.4rem 0.6rem;
    min-width: 140px;
  }
  
  .history-body {
    font-size: 0.8rem;
  }
  
  .gesture-word {
    font-size: 0.8rem;
    padding: 2px 6px;
  }
}

@media (max-height: 500px) {
  .layout {
    flex-direction: row;
  }
  
  .main {
    height: 100%;
    min-height: 0;
  }
  
  .gestures-panel {
    width: 280px;
    min-width: 280px;
  }
  
  .panel-card.history {
    max-height: 120px;
  }
}

.history-body {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  align-content: flex-start;
  padding: 4px 0;
  min-height: auto;
}

.gesture-word {
  display: inline-flex;
  padding: 6px 12px;
  margin: 1px;
  background: white;
  border: 1px solid #e5e7eb;
  border-radius: 8px;
  font-size: 0.95rem;
  line-height: 1.4;
  height: fit-content;
  transition: all 0.2s;
  white-space: nowrap;
  align-items: center;
  justify-content: center;
}

.gesture-word:hover {
  transform: translateY(-1px);
  box-shadow: 0 2px 6px rgba(0, 0, 0, 0.1);
}

.gesture-word.latest {
  background: #e0f2fe;
  border-color: #93c5fd;
  color: #0369a1;
  font-weight: 600;
  box-shadow: 0 1px 3px rgba(59, 130, 246, 0.2);
}

@media (max-width: 900px) {
  .gestures-panel {
    display: grid;
    grid-template-columns: 1fr 1fr;
    grid-template-rows: auto 1fr;
    gap: 0.6rem;
  }
  
  .panel-card:first-child {
    grid-column: 1 / -1;
    grid-row: 1;
    max-height: 120px;
  }
  
  .panel-card.scroll {
    grid-column: 1;
    grid-row: 2;
    min-height: 150px;
    max-height: 180px;
  }
  
  .panel-card.history {
    grid-column: 2;
    grid-row: 2;
    min-height: 150px;
    max-height: 180px;
    display: flex;
    flex-direction: column;
  }
  
  .history-body {
    flex: 1;
    min-height: 0;
    overflow-y: auto !important;
    overflow-x: hidden;
    max-height: calc(100% - 50px);
  }
  
  .gesture-word {
    padding: 4px 8px;
    font-size: 0.85rem;
    white-space: normal;
    word-break: break-word;
    line-height: 1.2;
  }
}

@media (max-width: 700px) {
  .gestures-panel {
    grid-template-columns: 1fr;
    grid-template-rows: auto auto auto;
  }
  
  .panel-card:first-child {
    grid-column: 1;
    grid-row: 1;
    max-height: 110px;
  }
  
  .panel-card.scroll {
    grid-column: 1;
    grid-row: 2;
    min-height: 120px;
    max-height: 140px;
  }
  
  .panel-card.history {
    grid-column: 1;
    grid-row: 3;
    min-height: 160px;
    max-height: 200px;
  }
  
  .history-body {
    max-height: calc(100% - 60px);
  }
}

@media (max-width: 480px) {
  .panel-card.history {
    min-height: 140px;
    max-height: 180px;
  }
  
  .history-body {
    max-height: calc(100% - 55px);
  }
}

.panel-card.history {
  overflow: hidden;
}

.history-body {
  scrollbar-width: thin;
  scrollbar-color: #c1c1c1 #f1f1f1;
}

.gesture-word {
  animation: fadeIn 0.3s ease forwards;
  opacity: 0;
}

@keyframes fadeIn {
  from {
    opacity: 0;
    transform: translateY(3px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

.gesture-placeholder {
  color: #9ca3af;
  font-size: 0.9rem;
  padding: 1rem;
  text-align: center;
  width: 100%;
  font-style: italic;
  display: flex;
  align-items: center;
  justify-content: center;
  height: 100%;
}

.history-body {
  align-content: flex-start;
  max-height: none;
}

.gesture-word {
  max-width: 100%;
  overflow: hidden;
  text-overflow: ellipsis;
}

@media (max-width: 1024px) and (max-height: 600px) {
  .gesture-word {
    white-space: normal;
    word-break: break-word;
    line-height: 1.2;
    text-align: center;
    padding: 3px 6px;
    font-size: 0.8rem;
  }
}
</style>
