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
  <section class="surface-panel px-5 py-6 sm:px-6">
    <header class="flex flex-wrap items-start justify-between gap-4">
      <div>
        <p class="badge-kicker">Workspace</p>
        <h2 class="mt-3 panel-title">上传与检测配置</h2>
        <p class="mt-2 panel-subtitle">
          上传待测文档和参考文档后，选择引擎参数即可发起任务。
        </p>
      </div>
      <span class="inline-flex items-center rounded-full border border-mint-500/25 bg-mint-500/10 px-3 py-1 text-xs font-semibold tracking-wide text-mint-600">
        实时查重
      </span>
    </header>

    <div class="mt-6 grid gap-4 xl:grid-cols-2">
      <article class="upload-panel upload-panel-target">
        <div class="flex items-center justify-between gap-3">
          <p class="text-sm font-bold text-mint-600">待检测文档</p>
          <span class="muted-label">单文件</span>
        </div>
        <input
          ref="targetInputRef"
          type="file"
          accept=".txt,.pdf,.docx"
          class="mt-3 block w-full cursor-pointer text-sm text-ink-900 file:mr-3 file:cursor-pointer file:rounded-full file:border-0 file:bg-night-900 file:px-4 file:py-2 file:text-xs file:font-bold file:text-white hover:file:bg-night-950"
          @change="onTargetChange"
        />
        <p class="mt-2 text-xs text-ink-900/65">支持 txt / pdf / docx</p>

        <div v-if="targetFile" class="mt-4 rounded-xl border border-white/75 bg-white/90 p-3">
          <p class="truncate text-sm font-semibold text-ink-900" :title="targetFile.name">{{ targetFile.name }}</p>
          <div class="mt-3 flex flex-wrap gap-2">
            <button class="btn-ghost" @click="openPreview(targetFile)">预览</button>
            <button class="btn-danger" @click="clearTarget">移除</button>
          </div>
        </div>
      </article>

      <article class="upload-panel upload-panel-reference">
        <div class="flex items-center justify-between gap-3">
          <p class="text-sm font-bold text-amber-600">参考文档库</p>
          <span class="muted-label">{{ refFiles.length }} 份文件</span>
        </div>
        <input
          ref="refInputRef"
          type="file"
          accept=".txt,.pdf,.docx"
          multiple
          class="mt-3 block w-full cursor-pointer text-sm text-ink-900 file:mr-3 file:cursor-pointer file:rounded-full file:border-0 file:bg-amber-500 file:px-4 file:py-2 file:text-xs file:font-bold file:text-white hover:file:bg-amber-600"
          @change="onRefChange"
        />
        <p class="mt-2 text-xs text-ink-900/65">可一次选择多份参考文档</p>

        <div v-if="refFiles.length" class="mt-4 max-h-56 space-y-2 overflow-y-auto pr-1">
          <div v-for="(file, index) in refFiles" :key="`${file.name}-${index}`" class="rounded-xl border border-white/75 bg-white/90 p-3">
            <p class="truncate text-sm font-semibold text-ink-900" :title="file.name">{{ file.name }}</p>
            <div class="mt-3 flex flex-wrap gap-2">
              <button class="btn-ghost" @click="openPreview(file)">预览</button>
              <button class="btn-danger" @click="removeRef(index)">移除</button>
            </div>
          </div>
        </div>
      </article>
    </div>

    <section class="mt-6 rounded-2xl border border-mint-500/20 bg-white/85 p-4 sm:p-5">
      <div class="flex items-start justify-between gap-3">
        <div>
          <p class="muted-label">Engine Settings</p>
          <h3 class="mt-1 font-display text-lg font-bold text-ink-900">检测参数</h3>
        </div>
      </div>

      <div class="mt-4 grid gap-3 sm:grid-cols-2">
        <label class="flex flex-col gap-1.5">
          <span class="muted-label">检测引擎</span>
          <select
            v-model="modeModel"
            class="rounded-xl border border-mint-500/25 bg-paper-50 px-3 py-2.5 text-sm font-semibold text-ink-900 focus:border-mint-600 focus:outline-none"
          >
            <option value="bert">深度语义引擎 (BGE)</option>
            <option value="tfidf">传统引擎 (TF-IDF + LSA)</option>
          </select>
        </label>

        <label
          v-if="modeModel === 'bert'"
          class="flex flex-col gap-1.5"
        >
          <span class="muted-label">BGE阈值档位</span>
          <select
            v-model="bertProfileModel"
            class="rounded-xl border border-mint-500/25 bg-paper-50 px-3 py-2.5 text-sm font-semibold text-ink-900 focus:border-mint-600 focus:outline-none"
          >
            <option value="strict">Strict（终审/低误报）</option>
            <option value="balanced">Balanced（日常默认）</option>
            <option value="recall">Recall（初筛/高召回）</option>
          </select>
        </label>

        <label class="flex items-center gap-2 rounded-xl border border-mint-500/25 bg-paper-50 px-3 py-2.5 sm:col-span-2">
          <input v-model="bodyModeModel" type="checkbox" class="h-4 w-4 accent-mint-600" />
          <span class="text-sm font-semibold text-ink-900">启用正文模式</span>
        </label>
      </div>

      <button class="btn-primary mt-5 w-full" :disabled="loading" @click="submit">
        <span v-if="loading" class="inline-block h-4 w-4 animate-spin rounded-full border-2 border-white/40 border-t-white"></span>
        {{ loading ? pollStatusMessage : "开始查重" }}
      </button>
      <p v-if="loading" class="mt-2 text-center text-xs font-semibold text-mint-600">{{ pollStatusMessage }}</p>
    </section>

    <p v-if="notice" class="mt-4 rounded-xl border border-amber-300/45 bg-amber-100/70 px-3 py-2 text-sm font-semibold text-amber-700">
      {{ notice }}
    </p>
  </section>
</template>
