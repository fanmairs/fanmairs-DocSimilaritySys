<script setup>
defineProps({
  visible: {
    type: Boolean,
    default: false
  },
  title: {
    type: String,
    default: ""
  },
  content: {
    type: String,
    default: ""
  },
  loading: {
    type: Boolean,
    default: false
  }
});

const emit = defineEmits(["close"]);

const close = () => {
  emit("close");
};
</script>

<template>
  <teleport to="body">
    <div v-if="visible" class="fixed inset-0 z-50 flex items-center justify-center p-4">
      <button class="absolute inset-0 bg-ink-900/60" @click="close"></button>

      <article class="relative z-10 w-full max-w-5xl overflow-hidden rounded-3xl bg-white shadow-soft">
        <header class="flex items-center justify-between border-b border-mint-500/20 bg-paper-50 px-5 py-4">
          <h3 class="font-display text-lg font-bold text-ink-900">{{ title }}</h3>
          <button class="btn-ghost" @click="close">关闭</button>
        </header>

        <div class="max-h-[72vh] overflow-y-auto px-5 py-4">
          <div v-if="loading" class="space-y-2">
            <div class="h-4 w-full animate-pulse rounded bg-mint-500/15"></div>
            <div class="h-4 w-11/12 animate-pulse rounded bg-mint-500/15"></div>
            <div class="h-4 w-9/12 animate-pulse rounded bg-mint-500/15"></div>
          </div>
          <pre v-else class="whitespace-pre-wrap font-prose text-sm leading-7 text-ink-900">{{ content }}</pre>
        </div>
      </article>
    </div>
  </teleport>
</template>
