import { ref, computed, watch, onMounted } from 'vue'

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
      const contentType = res.headers.get('content-type')
      if (contentType && !contentType.includes('application/json')) return
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
      const contentType = res.headers.get('content-type')
      if (contentType && !contentType.includes('application/json')) return
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
      const contentType = res.headers.get('content-type')
      if (contentType && !contentType.includes('application/json')) return
	    const data = await res.json()
	    gestureHistory.value = data.history
	  } catch (e) {
	    console.error(e)
	  }
	}

	async function loadUsageVals() {
	  const res = await fetch('/api/getUsageVals')
	  if (!res.ok) return
    const contentType = res.headers.get('content-type')
      if (contentType && !contentType.includes('application/json')) return
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

  onMounted(() => {
	  setInterval(checkAiStatus, 2000)
	  checkAiStatus()
	  const saved = localStorage.getItem('displayMode')
	  if (saved) displayMode.value = saved

	  setInterval(loadUsageVals, 500)
	})

  watch(displayMode, v => localStorage.setItem('displayMode', v))

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
    cpuLoad,
    ramLoad,
    fps,
    gestures,
    gestureHistory,
    topGestures,

    toggleRecording,
    startProgram,
    sendToAI,
    clearHustory
  }
}
