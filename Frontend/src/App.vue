<script setup>
import { ref, watch, onMounted, computed, inject } from 'vue'
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
fetch(`https://api.github.com/repos/ogled/Gesturesphone/commits?per_page=1`).then(res => res.json()).then(commits => {
    appVersion.value = commits[0].sha.slice(0, 7)
  });
const cpuLoad = ref(0)
const ramLoad = ref(0)
/* ===== Gestures ===== */
const gestures = ref([])
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
  setInterval(() => {
    loadUsageVals()
  }, 500)
})

async function startProgram() {
  const res = await fetch('/api/start')
  const data = await res.json()
  if(res.status == 200)
  {
      isRuninngProgramm.value = true
      setInterval(() => {
      loadGestures()
    }, 500)
  }
  else
  {
    window.showToast('Error ' + res.status + ': ' + data.detail, 'error')
  }
}
async function loadGestures() {
  try {
    const res = await fetch('/api/gestures')
    if (!res.ok) return

    const data = await res.json()

    const raw = data?.gestures ?? {}

    gestures.value = Object.entries(raw).map(
      ([name, confidence]) => ({
        name,
        confidence
      })
    )
  } catch (e) {
    gestures.value = []
  }
}

async function loadUsageVals() {
  isGesturesLoading.value = true
  gesturesError.value = null

  const res = await fetch('/api/getUsageVals')
  if (!res.ok) {
    throw new Error('Ошибка загрузки покозателей загружености системы')
  }
  const data = await res.json()
  cpuLoad.value = data.CPU
  ramLoad.value = data.RAM
  
}
watch(displayMode, (v) => localStorage.setItem('displayMode', v))
</script>

<template>
  <div class="app">
    <!-- Header -->
    <nav class="tabs">
      <span class="projectName">{{ projectName }}</span>

      <button
        class="tab"
        :class="{ active: activeTab === 'main' }"
        @click="activeTab = 'main'"
      >
        Главная
      </button>

      <button
        class="tab"
        :class="{ active: activeTab === 'settings' }"
        @click="activeTab = 'settings'"
      >
        Настройки
      </button>
    </nav>

    <main class="content">
      <!-- ===== MAIN ===== -->
      <section v-if="activeTab === 'main'" class="fill">
        <!-- ===== PARTIAL MODE ===== -->
        <div v-if="!isRuninngProgramm" class="fill">
          <div class="StartingProgramm">
            <button class="start-btn" @click="startProgram">
              ▶ Запуск
            </button>
          </div>
        </div>
        <div v-else-if="isRuninngProgramm == true" class="fill">
          <div v-if="displayMode === 'partial'" class="layout">
            <!-- Камера -->
            <div class="main">
              <img :src="cameraUrl" class="camera-feed" />
            </div>

            <!-- Жесты -->
            <div class="gestures-panel">
              <div class="top-gestures">
                <h3>Наиболее вероятные</h3>

                <div
                  v-for="(g, i) in topGestures"
                  :key="g.name"
                  class="top-gesture"
                >
                  <div class="rank-badge">{{ i + 1 }}</div>

                  <div class="info">
                    <div class="name">{{ g.name }}</div>
                    <div class="bar">
                      <div
                        class="bar-fill"
                        :style="{ width: g.confidence + '%' }"
                      />
                    </div>
                  </div>

                  <div class="percent">{{ g.confidence }}%</div>
                </div>
              </div>

              <div class="other-gestures">
                <h4>Другие варианты</h4>

                <div
                  v-for="g in otherGestures"
                  :key="g.name"
                  class="other-row"
                >
                  <span class="dot"></span>
                  <span class="other-name">{{ g.name }}</span>
                  <span class="other-percent">{{ g.confidence }}%</span>
                </div>

                <p v-if="gestures.length > 10" class="more-gestures">
                  Здесь остальные {{ gestures.length - 10 }} жестов
                </p>
              </div>
            </div>
          </div>
        </div>
        

        <!-- ===== FULL MODE ===== -->
        <div v-if="displayMode === 'full'" class="full-mode">
          <div class="placeholder">
            Полный режим отображения<br />
            (будет реализован позже)
          </div>
        </div>
      </section>

      <!-- ===== SETTINGS ===== -->
      <section v-if="activeTab === 'settings'" class="settings">
        <div class="setting">
          <label>Режим отображения</label>
          <select v-model="displayMode">
            <option value="partial">Частичный</option>
            <option value="full">Полный</option>
          </select>
        </div>
      </section>
    </main>

    <!-- ===== STATUS BAR ===== -->
    <footer class="status-bar">
      <span class="version">
        {{ projectName }} • {{ appVersion }}
      </span>

      <div class="right-aligned">
        <span class="cpu">CPU: <b>{{ cpuLoad }}%</b></span>
        <span class="ram">RAM: <b>{{ ramLoad }}%</b></span>
      </div>
    </footer>
  </div>
  <ToastComponent />
</template>

<style scoped>
/* ===== Base ===== */
.app {
  height: 100vh;
  display: flex;
  flex-direction: column;
  background: #f6f7f9;
  font-family: system-ui, -apple-system;
}

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

/* ===== Layout ===== */
.layout {
  height: 100%;
  display: grid;
  grid-template-columns: auto 40%;
  gap: 1rem;
}

/* ===== Camera ===== */
.main {
  width: 100%;
  height: 0;
  padding-top: 75%;
  position: relative;
  background: black;
  border-radius: 6px;
  overflow: hidden;
  display: flex;
  justify-content: center;
  align-items: center;
}

.camera-feed {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  object-fit: contain;
}

/* ===== Gestures ===== */
.gestures-panel {
  background: white;
  border-radius: 12px;
  padding: 1rem;
  display: flex;
  flex-direction: column;
  overflow: hidden;
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

.cpu, .ram {
  display: inline-flex;
  align-items: center;
  margin-right: 8px;
}

.cpu b, .ram b {
  min-width: 4ch;
  text-align: right;
  padding-left: 4px;
  display: inline-block;
  font-variant-numeric: tabular-nums;
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

/* реакция на нажатие */
.start-btn:active,
.stop-btn:active {
  transform: scale(0.95);
}

.stop-btn {
  background: #ef4444;
  margin-bottom: 0.5rem;
}


/* ===== Mobile ===== */
@media (max-width: 768px) {
  .layout {
    grid-template-columns: 1fr;
  }
}
</style>
