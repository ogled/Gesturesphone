<template>
  <div class="toast-container">
    <div v-for="toast in toasts"
          :key="toast.id"
         :class="['toast', `toast-${toast.type}`]"
         @click="removeToast(toast.id)">
      
      <template v-if="toast.type !== 'ai'">
        {{ toast.message }}
      </template>
      
      <template v-else>
        <div class="ai-toast-content">
          <div class="ai-header">
            <svg class="ai-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" 
                    d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
            </svg>
            <span class="ai-title">ИИ результат</span>
          </div>
          <div class="ai-message">{{ toast.message }}</div>
          <div class="ai-footer">
            <span class="ai-time">{{ toast.time }}</span>
            <span class="ai-tip">Нажмите, чтобы скрыть</span>
          </div>
        </div>
      </template>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'

const toasts = ref([])
let toastId = 0

function showToast(message, type = 'error') {
  const id = toastId++
  const time = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
  
  toasts.value.push({
    id,
    message,
    type,
    time
  })
  
  setTimeout(() => {
    removeToast(id)
  }, 8000)
}

function removeToast(id) {
  const index = toasts.value.findIndex(toast => toast.id === id)
  if (index > -1) {
    toasts.value.splice(index, 1)
  }
}

defineExpose({ showToast })

onMounted(() => {
  window.showToast = showToast
  
  return () => {
    delete window.showToast
  }
})
</script>

<style scoped>
.toast-container {
  position: fixed;
  top: 20px;
  right: 20px;
  z-index: 1000;
}

.toast {
  padding: 12px 20px;
  margin-bottom: 10px;
  border-radius: 12px;
  color: white;
  cursor: pointer;
  min-width: 300px;
  max-width: 400px;
  box-shadow: 0 4px 12px rgba(0,0,0,0.15);
  animation: slideIn 0.3s ease;
  transition: transform 0.2s, opacity 0.2s;
  backdrop-filter: blur(10px);
}

.toast:hover {
  transform: translateY(-2px);
  box-shadow: 0 6px 16px rgba(0,0,0,0.2);
}

.toast-error {
  background: linear-gradient(135deg, #f44336, #e53935);
  border-left: 4px solid #ff7961;
}

.toast-success {
  background: linear-gradient(135deg, #4caf50, #43a047);
  border-left: 4px solid #80e27e;
}

.toast-ai {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  padding: 0;
  overflow: hidden;
  min-width: 350px;
  max-width: 450px;
  border: 1px solid rgba(255, 255, 255, 0.1);
}

.toast-warning {
  background: linear-gradient(135deg, #ff9800, #f57c00);
  border-left: 4px solid #ffb74d;
}

.toast-info {
  background: linear-gradient(135deg, #2196f3, #1e88e5);
  border-left: 4px solid #64b5f6;
}

.ai-toast-content {
  padding: 16px;
}

.ai-header {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-bottom: 12px;
}

.ai-icon {
  width: 20px;
  height: 20px;
  stroke-width: 1.5;
  opacity: 0.9;
}

.ai-title {
  font-weight: 600;
  font-size: 0.9rem;
  letter-spacing: 0.5px;
  text-transform: uppercase;
  opacity: 0.95;
}

.ai-message {
  font-size: 1rem;
  line-height: 1.4;
  margin-bottom: 12px;
  padding: 8px 0;
  border-top: 1px solid rgba(255, 255, 255, 0.1);
  border-bottom: 1px solid rgba(255, 255, 255, 0.1);
}

.ai-footer {
  display: flex;
  justify-content: space-between;
  align-items: center;
  font-size: 0.75rem;
  opacity: 0.8;
}

.ai-time {
  background: rgba(255, 255, 255, 0.1);
  padding: 2px 8px;
  border-radius: 10px;
}

.ai-tip {
  font-style: italic;
}

@keyframes slideIn {
  from {
    transform: translateX(100%);
    opacity: 0;
  }
  to {
    transform: translateX(0);
    opacity: 1;
  }
}

.toast-ai {
  animation: slideInAi 0.4s cubic-bezier(0.68, -0.55, 0.265, 1.55);
}

@keyframes slideInAi {
  0% {
    transform: translateX(100%) scale(0.9);
    opacity: 0;
  }
  70% {
    transform: translateX(-10px) scale(1.02);
  }
  100% {
    transform: translateX(0) scale(1);
    opacity: 1;
  }
}

@media (max-width: 768px) {
  .toast-container {
    top: 10px;
    right: 10px;
    left: 10px;
  }
  
  .toast {
    min-width: auto;
    max-width: none;
    width: 100%;
  }
  
  .toast-ai {
    min-width: auto;
  }
}
</style>