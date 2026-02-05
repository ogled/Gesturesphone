<script setup>
import { useMainView } from './MainView.logic'
import MainViewTemplate from './MainView.template.vue'
import './MainView.css'

const state = useMainView()
</script>

<template>
    <MainViewTemplate v-bind="state" />
</template>
