import { ref, computed, watch, onMounted, onUnmounted } from 'vue'

const DEMO_SEQUENCE = [
  'Привет',
  'я',
  'хотеть',
  'вода',
  'друг',
  'идти',
  'дом',
]

const DEMO_ALTERNATIVES = [
  'помочь',
  'ждать',
  'слово',
  'время',
  'читать',
  'говорить',
  'делать',
  'книга',
]

const MAX_HISTORY_ITEMS = 20

export function useLocalStorage(key, defaultValue) {
  const state = ref(defaultValue)

  try {
    const saved = localStorage.getItem(key)
    if (saved !== null) {
      state.value = JSON.parse(saved)
    }
  } catch (e) {
    console.warn(`localStorage load failed for ${key}`, e)
  }

  watch(
    state,
    (v) => {
      try {
        localStorage.setItem(key, JSON.stringify(v))
      } catch (e) {
        console.warn(`localStorage save failed for ${key}`, e)
      }
    },
    { deep: true }
  )

  return state
}

export function useMainView() {
  const projectName = 'Gesturesphone'
  const appVersion = ref('dev')
  const modelVersion = ref('unknown')

  const activeTab = ref('main')
  const displayMode = ref('partial')

  const isRuninngProgramm = ref(false)
  const isDemoMode = ref(false)
  const isRecordingMode = ref(false)

  const cameraUrl = '/api/camera'

  const runtimeHealth = ref({
    status: 'degraded',
    camera: 'unavailable',
    model: 'error',
    ai: 'not_ready',
    runtime_backend: 'onnx',
  })

  const aiStatus = ref('not_ready')
  const isAiProcessing = ref(false)

  const selectedVoice = useLocalStorage('selectedVoice', 'НЕТ')
  const voices = ref([])
  const volume = useLocalStorage('volume', 0.5)
  const playGestureHistory = useLocalStorage('playGestureHistory', false)
  const playAIResults = useLocalStorage('playAIResults', false)

  const cpuLoad = ref(0)
  const ramLoad = ref(0)
  const fps = ref(0)

  const gestures = ref([])
  const gestureHistory = ref([])

  const canToggleRecording = computed(() => {
    if (isAiProcessing.value) return false
    if (isDemoMode.value) return true
    return aiStatus.value === 'ready'
  })

  const canSendToAI = computed(() => {
    if (isRecordingMode.value || isAiProcessing.value) return false
    if (isDemoMode.value) return gestureHistory.value.length > 0
    return aiStatus.value === 'ready'
  })

  const topGestures = computed(() => {
    const sorted = [...(gestures.value || [])].sort((a, b) => b.confidence - a.confidence)
    const result = []

    for (let i = 0; i < 3; i++) {
      if (sorted[i]) {
        result.push(sorted[i])
      } else {
        result.push({ name: '—', confidence: 0 })
      }
    }

    return result
  })

  const timers = {
    aiStatus: null,
    usage: null,
    health: null,
    liveFeed: null,
    demoFeed: null,
    aiPoll: null,
  }

  const lastSpokenIndex = ref(0)

  function clearTimer(name) {
    if (timers[name]) {
      clearInterval(timers[name])
      timers[name] = null
    }
  }

  function clearAllTimers() {
    Object.keys(timers).forEach(clearTimer)
  }

  function loadVoices() {
    const allVoices = speechSynthesis.getVoices()
    voices.value = allVoices.filter((v) => v.lang.startsWith('ru'))

    if (selectedVoice.value === 'НЕТ' && voices.value.length > 0) {
      selectedVoice.value = voices.value[0].name
    }
  }

  async function loadVersion() {
    try {
      const res = await fetch('/api/version')
      if (!res.ok) return
      const contentType = res.headers.get('content-type') || ''
      if (!contentType.includes('application/json')) return

      const data = await res.json()
      appVersion.value = data.app_version || 'dev'
      modelVersion.value = data.model_version || 'unknown'
    } catch (e) {
      console.error('Version fetch failed:', e)
    }
  }

  async function loadHealth() {
    try {
      const res = await fetch('/api/health')
      if (!res.ok) return
      const contentType = res.headers.get('content-type') || ''
      if (!contentType.includes('application/json')) return

      const data = await res.json()
      runtimeHealth.value = {
        status: data.status || 'degraded',
        camera: data.camera || 'unavailable',
        model: data.model || 'error',
        ai: data.ai || 'not_ready',
        runtime_backend: data.runtime_backend || 'onnx',
      }
      aiStatus.value = runtimeHealth.value.ai
    } catch (e) {
      console.error('Health fetch failed:', e)
    }
  }

  async function checkAiStatus() {
    try {
      const res = await fetch('/api/ai-get-status')
      if (!res.ok) return
      const contentType = res.headers.get('content-type') || ''
      if (!contentType.includes('application/json')) return

      const data = await res.json()
      aiStatus.value = data.status
    } catch (e) {
      console.error('AI status check failed:', e)
      aiStatus.value = 'error'
    }
  }

  function ensureLiveFeedPolling() {
    clearTimer('liveFeed')
    timers.liveFeed = setInterval(() => {
      loadGestures()
      loadGestureHistory()
    }, 500)
  }

  function buildDemoSnapshot(word, index) {
    const altFirst = DEMO_ALTERNATIVES[index % DEMO_ALTERNATIVES.length]
    const altSecond = DEMO_ALTERNATIVES[(index + 3) % DEMO_ALTERNATIVES.length]

    return [
      { name: word, confidence: 92 },
      { name: altFirst, confidence: 56 },
      { name: altSecond, confidence: 32 },
    ]
  }

  function pushHistoryItem(word) {
    const next = [...gestureHistory.value, word]
    gestureHistory.value = next.slice(-MAX_HISTORY_ITEMS)

    if (playGestureHistory.value && !isRecordingMode.value) {
      speakText(word)
    }
  }

  function startDemoMode() {
    if (isDemoMode.value) return

    clearTimer('liveFeed')
    clearTimer('demoFeed')

    isRuninngProgramm.value = true
    isDemoMode.value = true
    isRecordingMode.value = false

    gestureHistory.value = []
    gestures.value = []
    lastSpokenIndex.value = 0

    let cursor = 0
    const runDemoTick = () => {
      const word = DEMO_SEQUENCE[cursor % DEMO_SEQUENCE.length]
      gestures.value = buildDemoSnapshot(word, cursor)
      if (!isRecordingMode.value) {
        pushHistoryItem(word)
      }
      cursor += 1
    }

    runDemoTick()
    timers.demoFeed = setInterval(runDemoTick, 1500)
  }

  function stopDemoMode() {
    if (!isDemoMode.value) return

    clearTimer('demoFeed')
    isDemoMode.value = false
    gestures.value = []
    gestureHistory.value = []
    lastSpokenIndex.value = 0
  }

  async function startProgram() {
    if (isRuninngProgramm.value && !isDemoMode.value) return

    stopDemoMode()

    try {
      const res = await fetch('/api/start')
      const data = await res.json()

      if (res.status === 200) {
        isRuninngProgramm.value = true
        await Promise.all([loadHealth(), loadGestures(), loadGestureHistory()])
        ensureLiveFeedPolling()
      } else {
        window.showToast('Error ' + res.status + ': ' + data.detail, 'error')
      }
    } catch (e) {
      console.error('Program start failed:', e)
      window.showToast('Не удалось запустить runtime', 'error')
    }
  }

  async function sendToAI() {
    if (!canSendToAI.value) return

    if (isDemoMode.value) {
      const text = gestureHistory.value.join(' ')
      if (!text) {
        window.showToast('Нет данных для demo', 'error')
        return
      }

      window.showToast(text, 'ai')
      if (playAIResults.value) {
        speakText(text)
      }
      return
    }

    isAiProcessing.value = true

    try {
      const startRes = await fetch('/api/ai-corect-gestures')

      if (!startRes.ok) {
        throw new Error('Failed to start AI processing')
      }

      clearTimer('aiPoll')
      timers.aiPoll = setInterval(async () => {
        try {
          const res = await fetch('/api/get-ai-corect-text')
          if (!res.ok) return

          const data = await res.json()
          if (!data.busy) {
            clearTimer('aiPoll')
            isAiProcessing.value = false
            const text = data?.text ?? ' '
            window.showToast(text, 'ai')
            checkAiStatus()
            if (playAIResults.value && text) {
              speakText(text)
            }
          }
        } catch (e) {
          window.showToast('Ошибка получения результата от ИИ', 'error')
          console.error('Polling failed:', e)
          clearTimer('aiPoll')
          isAiProcessing.value = false
        }
      }, 500)
    } catch (e) {
      console.error('AI processing failed:', e)
      window.showToast('AI processing failed', 'error')
      isAiProcessing.value = false
    }
  }

  async function toggleRecording() {
    if (!canToggleRecording.value) return

    isRecordingMode.value = !isRecordingMode.value

    if (isDemoMode.value) return

    try {
      if (isRecordingMode.value) {
        await fetch('/api/start-recording-mode')
      } else {
        await fetch('/api/end-recording-mode')
      }
    } catch (e) {
      console.error('Recording mode switch failed:', e)
      window.showToast('Не удалось переключить режим записи', 'error')
    }
  }

  async function loadGestures() {
    if (isDemoMode.value) return

    try {
      const res = await fetch('/api/gestures')
      if (!res.ok) return
      const contentType = res.headers.get('content-type') || ''
      if (!contentType.includes('application/json')) return

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
    if (isDemoMode.value) return

    try {
      const res = await fetch('/api/gesture-history')
      if (!res.ok) return
      const contentType = res.headers.get('content-type') || ''
      if (!contentType.includes('application/json')) return

      const data = await res.json()
      let newHistory = Array.isArray(data.history) ? data.history : []
      newHistory = newHistory.filter((item) => item != null)

      if (newHistory.length === 0) {
        gestureHistory.value = []
        lastSpokenIndex.value = 0
        return
      }

      const newItems = newHistory.slice(lastSpokenIndex.value)
      gestureHistory.value = newHistory

      if (playGestureHistory.value && newItems.length > 0 && !isRecordingMode.value) {
        for (const item of newItems) {
          speakText(typeof item === 'string' ? item : item.text ?? '')
        }
      }

      lastSpokenIndex.value = newHistory.length
    } catch (e) {
      console.error(e)
    }
  }

  async function loadUsageVals() {
    try {
      const res = await fetch('/api/getUsageVals')
      if (!res.ok) return
      const contentType = res.headers.get('content-type') || ''
      if (!contentType.includes('application/json')) return

      const data = await res.json()
      cpuLoad.value = data.CPU
      ramLoad.value = data.RAM
      fps.value = data.FPS
    } catch (e) {
      console.error('Usage fetch failed:', e)
    }
  }

  async function clearHustory() {
    gestureHistory.value = []
    lastSpokenIndex.value = 0

    if (isDemoMode.value) return

    const res = await fetch('/api/clearHistory')
    if (!res.ok) {
      window.showToast('Не удалось очистить историю', 'error')
    }
  }

  async function speakText(text) {
    if (!selectedVoice.value) return

    const utter = new SpeechSynthesisUtterance(text)
    const voice = voices.value.find((v) => v.name === selectedVoice.value)
    if (voice) utter.voice = voice

    utter.lang = 'ru-RU'
    utter.volume = Number(volume.value)
    speechSynthesis.cancel()
    speechSynthesis.speak(utter)
  }

  async function deleteGesture(gestureName, index) {
    if (gestureName == null) return

    if (isDemoMode.value) {
      gestureHistory.value = gestureHistory.value.filter((_, i) => i !== index)
      return
    }

    await fetch(`/api/del-gesture?id=${index}&name=${encodeURIComponent(gestureName)}`, {
      method: 'DELETE',
    })
  }

  onMounted(() => {
    loadVoices()
    speechSynthesis.onvoiceschanged = loadVoices

    loadVersion()
    loadHealth()
    checkAiStatus()
    loadUsageVals()

    clearTimer('aiStatus')
    clearTimer('usage')
    clearTimer('health')

    timers.aiStatus = setInterval(checkAiStatus, 2000)
    timers.health = setInterval(loadHealth, 2000)
    timers.usage = setInterval(loadUsageVals, 500)
  })

  onUnmounted(() => {
    speechSynthesis.cancel()
    speechSynthesis.onvoiceschanged = null
    clearAllTimers()
  })

  return {
    projectName,
    appVersion,
    modelVersion,
    activeTab,
    displayMode,
    isRuninngProgramm,
    isDemoMode,
    isRecordingMode,
    cameraUrl,
    runtimeHealth,
    aiStatus,
    isAiProcessing,
    voices,
    selectedVoice,
    volume,
    cpuLoad,
    ramLoad,
    fps,
    gestures,
    gestureHistory,
    topGestures,
    playGestureHistory,
    playAIResults,
    canToggleRecording,
    canSendToAI,

    startDemoMode,
    toggleRecording,
    startProgram,
    sendToAI,
    clearHustory,
    speakText,
    deleteGesture,
  }
}
