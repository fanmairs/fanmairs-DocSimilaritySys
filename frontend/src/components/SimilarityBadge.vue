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
    return "bg-rose-100 text-rose-700 border-rose-200";
  }
  if (props.score >= 0.35) {
    return "bg-amber-100 text-amber-700 border-amber-200";
  }
  return "bg-emerald-100 text-emerald-700 border-emerald-200";
});
</script>

<template>
  <span
    class="inline-flex items-center rounded-full border px-3 py-1 text-xs font-bold tracking-wide"
    :class="styleClass"
  >
    {{ label }} {{ percentage }}
  </span>
</template>
