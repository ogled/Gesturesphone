<template>
  <div class="toast-container">
    <div v-for="toast in toasts" 
         :key="toast.id"
         :class="['toast', `toast-${toast.type}`]"
         @click="removeToast(toast.id)">
      {{ toast.message }}
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'

const toasts = ref([])
let toastId = 0

function showToast(message, type = 'error') {
  const id = toastId++
  toasts.value.push({
    id,
    message,
    type
  })
  
  setTimeout(() => {
    removeToast(id)
  }, 5000)
}

function removeToast(id) {
  const index = toasts.value.findIndex(toast => toast.id === id)
  if (index > -1) {
    toasts.value.splice(index, 1)
  }
}

window.showToast = showToast

onMounted(() => {
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
  border-radius: 8px;
  color: white;
  cursor: pointer;
  min-width: 300px;
  max-width: 400px;
  box-shadow: 0 4px 12px rgba(0,0,0,0.15);
  animation: slideIn 0.3s ease;
}

.toast-error {
  background: #f44336;
}

.toast-success {
  background: #4caf50;
}

.toast-warning {
  background: #ff9800;
}

.toast-info {
  background: #2196f3;
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
</style>