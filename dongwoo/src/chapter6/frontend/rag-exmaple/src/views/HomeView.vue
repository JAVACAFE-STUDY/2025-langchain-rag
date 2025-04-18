<script setup lang="ts">
  import 'deep-chat';
  import { ref } from 'vue';

  const history = ref([
    { role: 'user', text: 'Hey, how are you today?' },
    { role: 'ai', text: 'I am doing very well!' },
  ]);  
</script>

<template>
  <h1>RAG Eaxmple</h1>
  <deep-chat
    :connect="{
      url: 'http://localhost:8080/chat',
    }"
  />
</template>

<style>
div {
  font-family: sans-serif;
  text-align: center;
  justify-content: center;
  display: grid;
}
</style>
