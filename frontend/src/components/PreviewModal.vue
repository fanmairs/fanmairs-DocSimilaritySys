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
    <div v-if="visible" class="fixed inset-0 z-50 flex items-center justify-center p-4 sm:p-6">
      <button class="absolute inset-0 bg-night-950/70 backdrop-blur-sm" @click="close"></button>

      <article class="relative z-10 w-full max-w-6xl overflow-hidden rounded-[1.8rem] border border-white/70 bg-white/95 shadow-soft">
        <header class="flex items-center justify-between border-b border-mint-500/20 bg-paper-50/90 px-5 py-4 sm:px-6">
          <div>
            <p class="text-xs font-semibold uppercase tracking-[0.18em] text-ink-900/55">Document Preview</p>
            <h3 class="mt-1 font-display text-lg font-bold text-ink-900">{{ title }}</h3>
          </div>
          <button class="btn-ghost" @click="close">关闭</button>
        </header>

        <div class="max-h-[72vh] overflow-y-auto px-5 py-4 sm:px-6">
          <div v-if="loading" class="space-y-2">
            <div class="h-4 w-full animate-soft-pulse rounded bg-mint-500/15"></div>
            <div class="h-4 w-11/12 animate-soft-pulse rounded bg-mint-500/15"></div>
            <div class="h-4 w-9/12 animate-soft-pulse rounded bg-mint-500/15"></div>
          </div>
          <pre v-else class="rounded-2xl border border-mint-500/18 bg-white px-4 py-4 whitespace-pre-wrap font-prose text-sm leading-7 text-ink-900 sm:px-5">{{ content }}</pre>
        </div>
      </article>
    </div>
  </teleport>
</template>
