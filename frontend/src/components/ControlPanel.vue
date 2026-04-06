<script setup>
import { computed, ref } from "vue";

const props = defineProps({
  targetFile: {
    type: Object,
    default: null
  },
  refFiles: {
    type: Array,
    default: () => []
  },
  mode: {
    type: String,
    default: "bert"
  },
  bertProfile: {
    type: String,
    default: "balanced"
  },
  bodyMode: {
    type: Boolean,
    default: true
  },
  loading: {
    type: Boolean,
    default: false
  },
  pollStatusMessage: {
    type: String,
    default: ""
  },
  notice: {
    type: String,
    default: ""
  }
});

const emit = defineEmits([
  "update:mode",
  "update:bertProfile",
  "update:bodyMode",
  "target-selected",
  "refs-selected",
  "clear-target",
  "remove-ref",
  "preview-file",
  "submit"
]);

const targetInputRef = ref(null);
const refInputRef = ref(null);

const modeModel = computed({
  get: () => props.mode,
  set: (value) => emit("update:mode", value)
});

const bertProfileModel = computed({
  get: () => props.bertProfile,
  set: (value) => emit("update:bertProfile", value)
});

const bodyModeModel = computed({
  get: () => props.bodyMode,
  set: (value) => emit("update:bodyMode", value)
});

const onTargetChange = (event) => {
  const file = event.target.files?.[0] || null;
  emit("target-selected", file);
};

const onRefChange = (event) => {
  const files = Array.from(event.target.files || []);
  emit("refs-selected", files);
};

const clearTarget = () => {
  if (targetInputRef.value) {
    targetInputRef.value.value = "";
  }
  emit("clear-target");
};

const removeRef = (index) => {
  emit("remove-ref", index);
};

const openPreview = (file) => {
  emit("preview-file", file);
};

const submit = () => {
  emit("submit");
};
</script>

<template>
  <section class="surface-card mb-8 px-6 py-6 sm:px-8">
    <div class="mb-5 flex items-center justify-between gap-4">
      <h2 class="panel-title">上传与检测配置</h2>
      <span class="rounded-full bg-ink-900 px-3 py-1 text-xs font-semibold tracking-wide text-white">
        实时查重
      </span>
    </div>

    <div class="grid gap-4 md:grid-cols-2">
      <article class="rounded-2xl border border-mint-500/20 bg-mint-500/5 p-4">
        <p class="mb-3 text-sm font-semibold text-mint-600">待检测文档</p>
        <input
          ref="targetInputRef"
          type="file"
          accept=".txt,.pdf,.docx"
          class="block w-full cursor-pointer text-sm text-ink-900 file:mr-4 file:rounded-full file:border-0 file:bg-mint-600 file:px-4 file:py-2 file:font-bold file:text-white hover:file:bg-mint-500"
          @change="onTargetChange"
        />

        <div v-if="targetFile" class="mt-4 rounded-xl bg-white p-3">
          <p class="truncate text-sm font-semibold text-ink-900" :title="targetFile.name">{{ targetFile.name }}</p>
          <div class="mt-3 flex gap-2">
            <button class="btn-ghost" @click="openPreview(targetFile)">预览</button>
            <button class="btn-ghost" @click="clearTarget">移除</button>
          </div>
        </div>
      </article>

      <article class="rounded-2xl border border-amber-500/20 bg-amber-500/5 p-4">
        <p class="mb-3 text-sm font-semibold text-amber-600">参考文档库</p>
        <input
          ref="refInputRef"
          type="file"
          accept=".txt,.pdf,.docx"
          multiple
          class="block w-full cursor-pointer text-sm text-ink-900 file:mr-4 file:rounded-full file:border-0 file:bg-amber-500 file:px-4 file:py-2 file:font-bold file:text-white hover:file:bg-amber-600"
          @change="onRefChange"
        />

        <div v-if="refFiles.length" class="mt-4 max-h-40 space-y-2 overflow-y-auto pr-1">
          <div v-for="(file, index) in refFiles" :key="`${file.name}-${index}`" class="rounded-xl bg-white p-3">
            <p class="truncate text-sm font-semibold text-ink-900" :title="file.name">{{ file.name }}</p>
            <div class="mt-3 flex gap-2">
              <button class="btn-ghost" @click="openPreview(file)">预览</button>
              <button class="btn-ghost" @click="removeRef(index)">移除</button>
            </div>
          </div>
        </div>
      </article>
    </div>

    <div class="mt-6 grid gap-4 rounded-2xl border border-mint-500/20 bg-white p-4 lg:grid-cols-[1fr_auto] lg:items-center">
      <div class="grid gap-4 md:grid-cols-2">
        <label class="flex flex-col gap-1">
          <span class="text-xs font-semibold uppercase tracking-wide text-ink-900/65">检测引擎</span>
          <select
            v-model="modeModel"
            class="rounded-xl border border-mint-500/20 bg-paper-50 px-3 py-2 text-sm font-semibold text-ink-900 focus:border-mint-600 focus:outline-none"
          >
            <option value="bert">深度语义引擎 (BGE)</option>
            <option value="tfidf">传统引擎 (TF-IDF + LSA)</option>
          </select>
        </label>

        <label
          v-if="modeModel === 'bert'"
          class="flex flex-col gap-1"
        >
          <span class="text-xs font-semibold uppercase tracking-wide text-ink-900/65">BGE阈值档位</span>
          <select
            v-model="bertProfileModel"
            class="rounded-xl border border-mint-500/20 bg-paper-50 px-3 py-2 text-sm font-semibold text-ink-900 focus:border-mint-600 focus:outline-none"
          >
            <option value="strict">Strict（终审/低误报）</option>
            <option value="balanced">Balanced（日常默认）</option>
            <option value="recall">Recall（初筛/高召回）</option>
          </select>
        </label>

        <label class="flex items-center gap-2 rounded-xl border border-mint-500/20 bg-paper-50 px-3 py-2">
          <input v-model="bodyModeModel" type="checkbox" class="h-4 w-4 accent-mint-600" />
          <span class="text-sm font-semibold text-ink-900">启用正文模式</span>
        </label>
      </div>

      <button class="btn-primary w-full lg:w-auto" :disabled="loading" @click="submit">
        <span v-if="loading" class="inline-block h-4 w-4 animate-spin rounded-full border-2 border-white/40 border-t-white"></span>
        {{ loading ? pollStatusMessage : "开始查重" }}
      </button>
    </div>

    <p v-if="notice" class="mt-4 rounded-xl border border-amber-200 bg-amber-100/70 px-3 py-2 text-sm font-semibold text-amber-700">
      {{ notice }}
    </p>
  </section>
</template>
