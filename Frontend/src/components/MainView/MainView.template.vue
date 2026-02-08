<script setup>
import ToastComponent from '../Toast.vue'

defineProps({
  projectName: String,
  appVersion: Object,
  activeTab: Object,
  displayMode: Object,
  isRuninngProgramm: Object,
  isRecordingMode: Object,
  cameraUrl: String,
  aiStatus: Object,
  isAiProcessing: Object,
  voices: Object,
  selectedVoice: Object,
  volume: Object,
  cpuLoad: Object,
  ramLoad: Object,
  fps: Object,
  gestures: Object,
  gestureHistory: Object,
  topGestures: Object,
  playGestureHistory: Object,
  playAIResults: Object,

  toggleRecording: Function,
  startProgram: Function,
  sendToAI: Function,
  clearHustory: Function,
  speakText: Function,
  deleteGesture: Function
})
</script>

<template>
  <div class="app">
    <!-- Header -->
    <nav class="tabs">
      <span class="projectName">{{ projectName }}</span>

      <button class="tab" :class="{ active: activeTab.value === 'main' }" @click="activeTab.value = 'main'">
        Главная
      </button>

      <button class="tab" :class="{ active: activeTab.value === 'settings' }" @click="activeTab.value = 'settings'">
        Настройки
      </button>
    </nav>

    <main class="content">
      <section v-if="activeTab.value === 'main'" class="fill">
        <div v-if="!isRuninngProgramm.value" class="StartingProgramm">
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

              <div 
                v-for="(g, i) in (topGestures.value || [])" 
                :key="g?.name ?? i" 
                class="top-gesture"
              >
                <div class="rank-badge">{{ i + 1 }}</div>

                <div class="info">
                  <div class="name">{{ g?.name ?? '...' }}</div>
                  <div class="bar">
                    <div class="bar-fill" :style="{ width: (g?.confidence ?? 0) + '%' }" />
                  </div>
                </div>

                <div class="percent">{{ g?.confidence ?? 0 }}%</div>
              </div>
            </section>

            <!-- HISTORY -->
            <section class="panel-card history">
              <div class="history-header">
                <h4>История жестов</h4>
            
                <div class="history-buttons">
                  <button class="history-btn clear":class="{ 
                      'clear': !isRecordingMode.value,
                      'disabled': isRecordingMode.value
                    }"
                   @click="clearHustory" title="Очистить историю"
                   :disabled="isRecordingMode.value"
                  ><img src="@/assets/trash-2.svg" alt="Очистить" class="icon"></button>
                  <button 
                    class="history-btn start"
                    :class="{ 
                      'recording': isRecordingMode.value,
                      'disabled': aiStatus.value !== 'ready' || isAiProcessing.value
                    }"
                    @click="toggleRecording"
                    :disabled="aiStatus.value !== 'ready' || isAiProcessing.value"
                    :title="
                      isRecordingMode.value 
                        ? 'Закончить высказывание' 
                        : 'Начать высказывание'"
                    >
                    <span v-if="!isRecordingMode.value"><img src="@/assets/mic.svg" alt="Очистить" class="icon"></span>
                    <span v-else class="record-dot"></span>
                  </button>

                  <button 
                    class="history-btn send"
                    :class="{ 
                      'ready': aiStatus.value === 'ready',
                      'disabled': aiStatus.value !== 'ready' || isRecordingMode.value,
                      'processing': isAiProcessing.value
                    }"
                    @click="sendToAI"
                    :disabled="aiStatus.value !== 'ready' || isAiProcessing.value || isRecordingMode.value"
                    title="Отправить текущую историю на обработку"
                  >
                    <span v-if="!isAiProcessing.value"><img src="@/assets/send.svg" alt="Очистить" class="icon"></span>
                    <span v-else class="spinner"></span>
                  </button>
                </div>
              </div>
            
              <div class="history-body" v-if="!isRecordingMode.value">
                <span
                  v-for="(g, i) in (gestureHistory.value || [])"
                  :key="i"
                  class="gesture-word"
                  :class="{ 
                    latest: i === ((gestureHistory.value?.length ?? 0) - 1),
                    phrase: g.includes(' ')
                  }"
                  @click="deleteGesture(g, i)"
                >
                  {{ g }}
                </span>
              
                <div
                  v-if="!gestureHistory.value || gestureHistory.value.length === 0"
                  class="gesture-placeholder"
                >
                  Пока жесты не распознаны
                </div>
              </div>

              <div v-else class="recording-view">
                <div class="recording-animation">
                  <span></span>
                  <span></span>
                  <span></span>
                </div>
                <div class="recording-text">Идёт запись высказывания…</div>
              </div>
            </section>
          </div>
        </div>
      </section>

      <section v-if="activeTab.value === 'settings'" class="settings panel-card">
        <h3>Озвучка</h3>
        <div class="setting-row">
          <label>Голос</label>
          <select v-model="selectedVoice.value">
            <option
              v-for="v in voices.value"
              :key="v.name"
              :value="v.name"
            >
              {{ v.name }} ({{ v.lang }})
            </option>
          </select>
        
          <div v-if="voices.value.length === 0" class="hint">
            Русские голоса не найдены в системе
          </div>
        </div>
      
        <div class="setting-row">
          <label>Громкость: {{ Math.round(volume.value * 100) }}%</label>
          <input
            type="range"
            min="0"
            max="1"
            step="0.05"
            v-model="volume.value"
          />
        </div>
        
        <div class="setting-row">
          <label>
            <input type="checkbox" v-model="playGestureHistory.value" />
            Проигрывать историю жестов
          </label>
          <label>
            <input type="checkbox" v-model="playAIResults.value" />
            Проигрывать результаты обработки ИИ
          </label>
        </div>

        <button class="test-voice-btn" @click="speakText('Проверка звука')">
          🔊 Проверить голос
        </button>
      </section>
    </main>

    <footer class="status-bar">
      <span>{{ projectName }} • {{ appVersion }}</span>
      <div>
        CPU: <b>{{ cpuLoad.value ?? 0 }}%</b> |
        RAM: <b>{{ ramLoad.value ?? 0 }}%</b> |
        FPS: <b>{{ fps.value ?? 0 }}</b>
      </div>
    </footer>
    <ToastComponent/>
  </div>
</template>