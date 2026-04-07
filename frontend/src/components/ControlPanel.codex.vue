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

const modeOptions = [
  {
    value: "bert",
    label: "BGE 深度语义",
    description: "适合检测语义改写、洗稿和结构借用"
  },
  {
    value: "tfidf",
    label: "TF-IDF + LSA",
    description: "适合快速完成基础文本比对和初筛"
  }
];

const profileOptions = [
  {
    value: "strict",
    label: "Strict",
    description: "强调低误报"
  },
  {
    value: "balanced",
    label: "Balanced",
    description: "适合日常检测"
  },
  {
    value: "recall",
    label: "Recall",
    description: "强调高召回"
  }
];

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

const sessionState = computed(() => {
  if (props.loading) {
    return props.pollStatusMessage || "分析中";
  }
  return "等待提交";
});

const openTargetPicker = () => {
  targetInputRef.value?.click();
};

const openRefPicker = () => {
  refInputRef.value?.click();
};

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
  <section class="surface-panel control-surface">
    <header class="control-header">
      <div>
        <p class="badge-kicker">Workspace</p>
        <h2 class="panel-title">文档上传与检测编排</h2>
        <p class="panel-subtitle">
          先确定输入文档，再切换检测策略。整个过程保留在一块连续操作面板里，而不是零散的卡片集合。
        </p>
      </div>

      <span class="state-pill" :class="{ 'state-pill--live': loading }">
        {{ sessionState }}
      </span>
    </header>

    <div class="control-grid">
      <section class="upload-lane upload-lane--target">
        <div class="lane-head">
          <div>
            <p class="lane-title">待检测文档</p>
            <small>一份主文档，作为整次检测的基准对象</small>
          </div>
          <span class="lane-tag">Single</span>
        </div>

        <input
          ref="targetInputRef"
          type="file"
          accept=".txt,.pdf,.docx"
          class="hidden"
          @change="onTargetChange"
        />

        <button class="lane-cta" @click="openTargetPicker">
          选择待检测文档
        </button>
        <p class="lane-note">支持 txt / pdf / docx，建议优先上传正文版本。</p>

        <div v-if="targetFile" class="file-stack">
          <article class="file-row">
            <div class="file-meta">
              <p>{{ targetFile.name }}</p>
              <small>待检测主文档</small>
            </div>

            <div class="file-actions">
              <button class="btn-ghost" @click="openPreview(targetFile)">预览</button>
              <button class="btn-danger" @click="clearTarget">移除</button>
            </div>
          </article>
        </div>

        <div v-else class="lane-empty">
          上传后会在这里显示当前主文档和快捷操作。
        </div>
      </section>

      <section class="upload-lane upload-lane--reference">
        <div class="lane-head">
          <div>
            <p class="lane-title">参考文档库</p>
            <small>一次可上传多份文档，系统将按相似度自动排序</small>
          </div>
          <span class="lane-tag">{{ refFiles.length }} 份</span>
        </div>

        <input
          ref="refInputRef"
          type="file"
          accept=".txt,.pdf,.docx"
          multiple
          class="hidden"
          @change="onRefChange"
        />

        <button class="lane-cta lane-cta--warm" @click="openRefPicker">
          添加参考文档
        </button>
        <p class="lane-note">适合批量比对参考库、往届材料或样本文本。</p>

        <div v-if="refFiles.length" class="file-stack file-stack--scroll">
          <article
            v-for="(file, index) in refFiles"
            :key="`${file.name}-${index}`"
            class="file-row"
          >
            <div class="file-meta">
              <p>{{ file.name }}</p>
              <small>第 {{ index + 1 }} 份参考文档</small>
            </div>

            <div class="file-actions">
              <button class="btn-ghost" @click="openPreview(file)">预览</button>
              <button class="btn-danger" @click="removeRef(index)">移除</button>
            </div>
          </article>
        </div>

        <div v-else class="lane-empty">
          参考文档上传后会形成待比对序列，可继续追加或替换。
        </div>
      </section>
    </div>

    <section class="setting-band">
      <div class="setting-group">
        <div class="setting-head">
          <p class="muted-label">Engine Switch</p>
          <h3 class="setting-title">检测引擎</h3>
        </div>

        <div class="choice-grid">
          <button
            v-for="option in modeOptions"
            :key="option.value"
            class="choice-chip"
            :class="{ 'choice-chip--active': modeModel === option.value }"
            @click="modeModel = option.value"
          >
            <strong>{{ option.label }}</strong>
            <span>{{ option.description }}</span>
          </button>
        </div>
      </div>

      <div v-if="modeModel === 'bert'" class="setting-group">
        <div class="setting-head">
          <p class="muted-label">Threshold Profile</p>
          <h3 class="setting-title">阈值档位</h3>
        </div>

        <div class="choice-grid choice-grid--triple">
          <button
            v-for="option in profileOptions"
            :key="option.value"
            class="choice-chip"
            :class="{ 'choice-chip--active': bertProfileModel === option.value }"
            @click="bertProfileModel = option.value"
          >
            <strong>{{ option.label }}</strong>
            <span>{{ option.description }}</span>
          </button>
        </div>
      </div>

      <div class="setting-group">
        <div class="setting-head">
          <p class="muted-label">Body Filter</p>
          <h3 class="setting-title">正文模式</h3>
        </div>

        <button class="toggle-field" @click="bodyModeModel = !bodyModeModel">
          <span class="toggle-track" :class="{ 'toggle-track--on': bodyModeModel }">
            <span class="toggle-thumb" :class="{ 'toggle-thumb--on': bodyModeModel }"></span>
          </span>
          <span class="toggle-copy">
            <strong>过滤封面、声明、参考文献等干扰区域</strong>
            <small>建议论文、报告类文档默认开启。</small>
          </span>
        </button>
      </div>
    </section>

    <section class="submit-band">
      <button class="btn-primary submit-cta" :disabled="loading" @click="submit">
        <span
          v-if="loading"
          class="inline-block h-4 w-4 animate-spin rounded-full border-2 border-white/40 border-t-white"
        ></span>
        {{ loading ? pollStatusMessage : "开始检测" }}
      </button>

      <p class="submit-note">
        BGE 模式适合语义改写检测，传统模式适合快速初筛。你也可以先初筛，再用语义引擎复核高风险文档。
      </p>
    </section>

    <p v-if="notice" class="notice-ribbon">
      {{ notice }}
    </p>
  </section>
</template>
