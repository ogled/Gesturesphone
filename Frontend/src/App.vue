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
const otherGestures = computed(() => sortedGestures.value.slice(3, 10))

/* ===== Lifecycle ===== */
onMounted(() => {
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

            <!-- Others -->
            <section class="panel-card scroll">
              <h4>Другие варианты</h4>

              <div v-for="g in otherGestures" :key="g.name" class="other-row">
                <span class="dot" />
                <span>{{ g.name }}</span>
                <span class="other-percent">{{ g.confidence }}%</span>
              </div>
            </section>

            <!-- HISTORY -->
            <section class="panel-card history">
              <h4>История жестов</h4>

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

/* ===== Others ===== */
.other-gestures {
  flex: 1;
  overflow-y: auto;
  border-top: 1px solid #e5e7eb;
  padding-top: 0.5rem;
}

.other-row {
  display: grid;
  grid-template-columns: 16px 1fr 40px;
  font-size: 0.78rem;
}

.dot {
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background: #9ca3af;
}

.other-percent {
  text-align: right;
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

.history-body {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  font-size: 1.1rem;
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


</style>
