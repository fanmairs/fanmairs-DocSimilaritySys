<script setup>
import { computed } from "vue";

const props = defineProps({
  score: {
    type: Number,
    default: 0
  },
  label: {
    type: String,
    required: true
  }
});

const percentage = computed(() => `${(props.score * 100).toFixed(2)}%`);

const styleClass = computed(() => {
  if (props.score >= 0.7) {
    return "signal-chip--high";
  }
  if (props.score >= 0.35) {
    return "signal-chip--mid";
  }
  return "signal-chip--low";
});
</script>

<template>
  <span class="signal-chip" :class="styleClass">
    <span class="signal-chip__dot"></span>
    <span class="signal-chip__label">{{ label }}</span>
    <strong>{{ percentage }}</strong>
  </span>
</template>
