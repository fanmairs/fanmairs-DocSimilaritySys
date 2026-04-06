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
    return "border-rose-200 bg-rose-100/85 text-rose-700";
  }
  if (props.score >= 0.35) {
    return "border-amber-200 bg-amber-100/85 text-amber-700";
  }
  return "border-emerald-200 bg-emerald-100/85 text-emerald-700";
});
</script>

<template>
  <span
    class="inline-flex items-center gap-1.5 rounded-full border px-3 py-1 text-xs font-bold tracking-wide"
    :class="styleClass"
  >
    <span class="inline-block h-1.5 w-1.5 rounded-full bg-current"></span>
    {{ label }} {{ percentage }}
  </span>
</template>
