import { ref, computed, watch, onMounted } from 'vue'

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
  const appVersion = ref('not loaded')
  fetch(`https://api.github.com/repos/ogled/Gesturesphone/commits?per_page=1`)
    .then(res => res.json())
    .then(commits => {
      appVersion.value = commits[0].sha.slice(0, 7)
    })

  const activeTab = ref('main')
  const displayMode = ref('partial')
  const isRuninngProgramm = ref(false)
  const isRecordingMode = ref(false)

  const cameraUrl = '/api/camera'

  const aiStatus = ref('not_ready')
  const isAiProcessing = ref(false)
  
  const selectedVoice = useLocalStorage("selectedVoice", ref("НЕТ"))
  const voices = ref([])
  const volume = useLocalStorage("volume", ref(0.5))
  const playGestureHistory = useLocalStorage("playGestureHistory", ref(false))
  const playAIResults = useLocalStorage("playAIResults", ref(false))

  const cpuLoad = ref(0)
  const ramLoad = ref(0)
  const fps = ref(0)

  const gestures = ref([])
  const gestureHistory = ref([])

  const topGestures = computed(() => {
    const sorted = (gestures.value || []).sort((a, b) => b.confidence - a.confidence)
    const result = []
    for (let i = 0; i < 3; i++) {
      if (sorted[i]) result.push(sorted[i])
      else result.push({ name: '...', confidence: 0 })
    }
    return result
  })

  function loadVoices() {
    const allVoices = speechSynthesis.getVoices()
    voices.value = allVoices.filter(v => v.lang.startsWith('ru'))

    if (selectedVoice.value === "НЕТ" && voices.value.length > 0) {
      selectedVoice.value = voices.value[0].name
    }
  }

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
      const contentType = res.headers.get('content-type') || ''
      if (!contentType.includes('application/json')) return
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
            const text = data?.text ?? ' '
            window.showToast(text, 'ai')
            checkAiStatus()
            if (playAIResults.value && text) {
              speakText(data.text)
            }
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
  async function toggleRecording() {
    isRecordingMode.value = !isRecordingMode.value

    if (isRecordingMode.value) {
      await fetch('/api/start-recording-mode')
    } else {
      await fetch('/api/end-recording-mode')
    }
  }

  async function loadGestures() {
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
  const lastSpokenIndex = ref(0)
	async function loadGestureHistory() {
    try {
      const res = await fetch('/api/gesture-history')
      if (!res.ok) return
      const contentType = res.headers.get('content-type') || ''
      if (!contentType.includes('application/json')) return
      const data = await res.json()
      const newHistory = Array.isArray(data.history) ? data.history : []
      if (newHistory.length === 0) {
        gestureHistory.value = []
        lastSpokenIndex.value = 0
        return
      }
      const newItems = newHistory.slice(lastSpokenIndex.value)
      gestureHistory.value = newHistory
      if (playGestureHistory.value && newItems.length > 0 && !isRecordingMode.value) {
        for (const item of newItems) {
          speakText(
            typeof item === 'string'
              ? item
              : item.text ?? ''
          )
        }
      }
      lastSpokenIndex.value = newHistory.length
    } catch (e) {
      console.error(e)
    }
  }

	async function loadUsageVals() {
	  const res = await fetch('/api/getUsageVals')
	  if (!res.ok) return
    const contentType = res.headers.get('content-type') || ''
    if (!contentType.includes('application/json')) return
	  const data = await res.json()
	  cpuLoad.value = data.CPU
	  ramLoad.value = data.RAM
	  fps.value = data.FPS
	}

  async function clearHustory() {
    gestureHistory.value = []
	  const res = await fetch('/api/clearHistory')
	  if (!res.ok) return
	}
  
  async function speakText(text) {
    if (!selectedVoice.value) return
    const utter = new SpeechSynthesisUtterance(text)
    const voice = voices.value.find(v => v.name === selectedVoice.value)
    if (voice) utter.voice = voice
    utter.lang = 'ru-RU'
    utter.volume = volume.value
    speechSynthesis.cancel()
    speechSynthesis.speak(utter)
  }
  
  async function deleteGesture(gestureName, index) {
    const res = await fetch(`/api/del-gesture?id=${index}&name=${encodeURIComponent(gestureName)}`, {method: 'DELETE'})
  }

  onMounted(() => {
    loadVoices()
    speechSynthesis.onvoiceschanged = loadVoices
	  checkAiStatus()
    setInterval(checkAiStatus, 2000)
	  setInterval(loadUsageVals, 500)
	})

  return {
    projectName,
    appVersion,
    activeTab,
    displayMode,
    isRuninngProgramm,
    isRecordingMode,
    cameraUrl,
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

    toggleRecording,
    startProgram,
    sendToAI,
    clearHustory,
    speakText,
    deleteGesture
  }
}
