<script setup>
import ToastComponent from '../Toast.vue'

defineProps({
  projectName: String,
  appVersion: Object,
  modelVersion: Object,
  activeTab: Object,
  displayMode: Object,
  isRuninngProgramm: Object,
  isDemoMode: Object,
  isRecordingMode: Object,
  cameraUrl: String,
  runtimeHealth: Object,
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
  canToggleRecording: Object,
  canSendToAI: Object,

  startDemoMode: Function,
  toggleRecording: Function,
  startProgram: Function,
  sendToAI: Function,
  clearHustory: Function,
  speakText: Function,
  deleteGesture: Function,
})
</script>

<template>
  <div class="app">
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
        <div v-if="!isRuninngProgramm.value" class="onboarding panel-card">
          <div class="onboarding-head">
            <h2>Gesturesphone</h2>
            <p>Распознавание жестов в реальном времени с преобразованием в текст и озвучку.</p>
          </div>

          <div class="onboarding-actions">
            <button class="start-btn" @click="startProgram">Запуск</button>
            <button class="demo-btn" @click="startDemoMode">Demo-режим</button>
          </div>
        </div>

        <div v-else class="layout">
          <div class="main panel-card camera-card">
            <div class="camera-header">
              <h3>Видео поток</h3>
              <button v-if="isDemoMode.value" class="switch-live-btn" @click="startProgram">Перейти в live</button>
            </div>

            <div v-if="isDemoMode.value" class="camera-state demo">
              <h4>Demo-режим активен</h4>
              <p>Показывается имитация распознавания без доступа к камере.</p>
            </div>

            <div v-else-if="runtimeHealth.value.camera !== 'ready'" class="camera-state error">
              <h4>Камера недоступна</h4>
              <p>Проверьте подключение камеры и права доступа приложения.</p>
            </div>

            <img v-else :src="cameraUrl" class="camera-feed" alt="Camera stream" />
          </div>

          <div class="gestures-panel">
            <section class="panel-card scroll">
              <h3>Наиболее вероятные</h3>

              <div v-if="!gestures.value || gestures.value.length === 0" class="placeholder-block">
                Жесты пока не распознаны
              </div>

              <div
                v-for="(g, i) in (topGestures.value || [])"
                :key="g?.name ?? i"
                class="top-gesture"
              >
                <div class="rank-badge">{{ i + 1 }}</div>

                <div class="info">
                  <div class="name">{{ g?.name ?? '—' }}</div>
                  <div class="bar">
                    <div class="bar-fill" :style="{ width: (g?.confidence ?? 0) + '%' }" />
                  </div>
                </div>

                <div class="percent">{{ g?.confidence ?? 0 }}%</div>
              </div>
            </section>

            <section class="panel-card history">
              <div class="history-header">
                <h4>История жестов</h4>

                <div class="history-buttons">
                  <button
                    class="history-btn clear"
                    :class="{ disabled: isRecordingMode.value }"
                    @click="clearHustory"
                    title="Очистить историю"
                    :disabled="isRecordingMode.value"
                  >
                    <img src="@/assets/trash-2.svg" alt="Очистить" class="icon" />
                  </button>

                  <button
                    class="history-btn start"
                    :class="{ disabled: !canToggleRecording.value, recording: isRecordingMode.value }"
                    @click="toggleRecording"
                    :disabled="!canToggleRecording.value"
                    :title="isRecordingMode.value ? 'Закончить высказывание' : 'Начать высказывание'"
                  >
                    <span v-if="!isRecordingMode.value"><img src="@/assets/mic.svg" alt="Запись" class="icon" /></span>
                    <span v-else class="record-dot"></span>
                  </button>

                  <button
                    class="history-btn send"
                    :class="{ disabled: !canSendToAI.value, processing: isAiProcessing.value }"
                    @click="sendToAI"
                    :disabled="!canSendToAI.value"
                    title="Отправить историю на обработку"
                  >
                    <span v-if="!isAiProcessing.value"><img src="@/assets/send.svg" alt="Отправить" class="icon" /></span>
                    <span v-else class="spinner"></span>
                  </button>
                </div>
              </div>

              <div v-if="!isRecordingMode.value" class="history-body">
                <div v-if="aiStatus.value !== 'ready' && !isDemoMode.value" class="hint warning">
                  AI недоступен: укажите токен в config.json
                </div>

                <span
                  v-for="(g, i) in (gestureHistory.value || [])"
                  :key="i"
                  class="gesture-word"
                  :class="{ latest: i === ((gestureHistory.value?.length ?? 0) - 1) }"
                  @click="deleteGesture(g, i)"
                >
                  {{ g || '?' }}
                </span>

                <div
                  v-if="!gestureHistory.value || gestureHistory.value.length === 0"
                  class="gesture-placeholder"
                >
                  Пока история пуста
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
          Проверить голос
        </button>
      </section>
    </main>

    <footer class="status-bar">
      <span>{{ projectName }} • app {{ appVersion.value }} • model {{ modelVersion.value }}</span>
      <div>
        CPU: <b>{{ cpuLoad.value ?? 0 }}%</b> |
        RAM: <b>{{ ramLoad.value ?? 0 }}%</b> |
        FPS: <b>{{ fps.value ?? 0 }}</b>
      </div>
    </footer>

    <ToastComponent />
  </div>
</template>
