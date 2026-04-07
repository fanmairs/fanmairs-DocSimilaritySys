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
    <div v-if="visible" class="preview-layer">
      <button class="preview-layer__mask" @click="close"></button>

      <article class="preview-panel">
        <header class="preview-panel__header">
          <div>
            <p class="muted-label">Document Preview</p>
            <h3 class="preview-panel__title">{{ title }}</h3>
          </div>
          <button class="btn-ghost" @click="close">关闭</button>
        </header>

        <div class="preview-panel__body">
          <div v-if="loading" class="preview-loading">
            <span></span>
            <span></span>
            <span></span>
          </div>

          <pre v-else class="preview-content">{{ content }}</pre>
        </div>
      </article>
    </div>
  </teleport>
</template>
