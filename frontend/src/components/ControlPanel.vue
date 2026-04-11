<script setup>
import { computed, ref } from "vue";
import {
  cloneCoarseConfig,
  coarseConfigGroups,
  coarseConfigPresets,
  customCoarseConfigGuide,
  detectCoarseConfigPreset
} from "../config/coarseRetrieval";

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
  bgeStrategy: {
    type: String,
    default: "coarse_then_fine"
  },
  bodyMode: {
    type: Boolean,
    default: true
  },
  coarseConfig: {
    type: Object,
    default: () => ({})
  },
  defaultCoarseConfig: {
    type: Object,
    default: () => ({})
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
  },
  windowEstimate: {
    type: Object,
    default: null
  },
  windowEstimateLoading: {
    type: Boolean,
    default: false
  },
  windowEstimateError: {
    type: String,
    default: ""
  }
});

const emit = defineEmits([
  "update:mode",
  "update:bertProfile",
  "update:bgeStrategy",
  "update:bodyMode",
  "update:coarseConfig",
  "reset-coarse-config",
  "estimate-bge-cost",
  "target-selected",
  "refs-selected",
  "clear-target",
  "remove-ref",
  "preview-file",
  "submit"
]);

const targetInputRef = ref(null);
const refInputRef = ref(null);
const coarseFieldGroups = coarseConfigGroups;
const coarsePresetOptions = computed(() =>
  coarseConfigPresets.map((preset) =>
    preset.key === "balanced"
      ? {
          ...preset,
          config: cloneCoarseConfig(
            Object.keys(props.defaultCoarseConfig || {}).length
              ? props.defaultCoarseConfig
              : preset.config
          )
        }
      : preset
  )
);

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

const bgeStrategyOptions = [
  {
    value: "coarse_then_fine",
    label: "快速模式",
    eyebrow: "粗筛后细检",
    description: "先用粗筛保留可疑参考文档，再对候选做窗口级细检，适合参考库较大时降低等待时间。"
  },
  {
    value: "full_fine",
    label: "完整模式",
    eyebrow: "全部细检",
    description: "跳过粗筛，所有参考文档都进入窗口级细检，适合文档规模可控且更看重完整性的任务。"
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

const bgeStrategyModel = computed({
  get: () => props.bgeStrategy,
  set: (value) => emit("update:bgeStrategy", value)
});

const bodyModeModel = computed({
  get: () => props.bodyMode,
  set: (value) => emit("update:bodyMode", value)
});

const activeCoarsePreset = computed(
  () => detectCoarseConfigPreset(props.coarseConfig, coarsePresetOptions.value)
);

const activeCoarsePresetLabel = computed(
  () => activeCoarsePreset.value?.label || "自定义"
);

const activeCoarsePresetGuide = computed(
  () => activeCoarsePreset.value || customCoarseConfigGuide
);

const activeStrategy = computed(
  () =>
    bgeStrategyOptions.find((option) => option.value === bgeStrategyModel.value) ||
    bgeStrategyOptions[0]
);

const formatInteger = (value) => Number(value || 0).toLocaleString("zh-CN");

const windowEstimateMetrics = computed(() => {
  if (!props.windowEstimate) {
    return [];
  }

  return [
    {
      label: "目标窗口",
      value: formatInteger(props.windowEstimate.target_window_count)
    },
    {
      label: "参考窗口",
      value: formatInteger(props.windowEstimate.reference_window_count)
    },
    {
      label: "全量矩阵",
      value: formatInteger(props.windowEstimate.full_pair_count)
    },
    {
      label: "参考文档",
      value: `${formatInteger(props.windowEstimate.reference_count)} 份`
    }
  ];
});

const windowScaleLabel = computed(() => {
  const level = props.windowEstimate?.scale_level;
  if (level === "large") {
    return "窗口规模较高";
  }
  if (level === "medium") {
    return "窗口规模中等";
  }
  if (level === "small") {
    return "窗口规模较小";
  }
  return "等待估算";
});

const updateCoarseConfigField = (field, event) => {
  const rawValue = event.target.value;
  const numericValue =
    field.type === "int" ? parseInt(rawValue, 10) : parseFloat(rawValue);

  if (!Number.isFinite(numericValue)) {
    return;
  }

  emit("update:coarseConfig", {
    ...props.coarseConfig,
    [field.key]: numericValue
  });
};

const applyCoarsePreset = (preset) => {
  emit("update:coarseConfig", cloneCoarseConfig(preset.config));
};

const resetCoarseConfig = () => {
  emit("reset-coarse-config");
};

const requestWindowEstimate = () => {
  emit("estimate-bge-cost");
};

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

      <div v-if="modeModel === 'bert'" class="setting-group">
        <div class="setting-head">
          <p class="muted-label">Verification Scope</p>
          <h3 class="setting-title">检测策略</h3>
        </div>

        <div class="strategy-layout">
          <div class="choice-grid">
            <button
              v-for="option in bgeStrategyOptions"
              :key="option.value"
              class="choice-chip choice-chip--strategy"
              :class="{ 'choice-chip--active': bgeStrategyModel === option.value }"
              type="button"
              @click="bgeStrategyModel = option.value"
            >
              <span class="choice-chip__eyebrow">{{ option.eyebrow }}</span>
              <strong>{{ option.label }}</strong>
              <span>{{ option.description }}</span>
            </button>
          </div>

          <article class="window-estimate-panel">
            <div class="window-estimate-panel__head">
              <div>
                <p class="muted-label">Window Estimate</p>
                <h4>{{ windowScaleLabel }}</h4>
              </div>
              <button
                class="btn-ghost"
                type="button"
                :disabled="windowEstimateLoading || !targetFile || !refFiles.length"
                @click="requestWindowEstimate"
              >
                {{ windowEstimateLoading ? "估算中" : "刷新估算" }}
              </button>
            </div>

            <div v-if="windowEstimate" class="window-metric-grid">
              <article
                v-for="metric in windowEstimateMetrics"
                :key="metric.label"
                class="window-metric"
              >
                <span>{{ metric.label }}</span>
                <strong>{{ metric.value }}</strong>
              </article>
            </div>

            <p v-if="windowEstimate" class="window-estimate-message">
              {{ windowEstimate.recommendation?.message }}
            </p>
            <p v-else-if="windowEstimateLoading" class="window-estimate-message">
              正在根据当前目标文档和参考文档估算窗口规模。
            </p>
            <p v-else-if="windowEstimateError" class="window-estimate-message window-estimate-message--warn">
              {{ windowEstimateError }}
            </p>
            <p v-else class="window-estimate-message">
              上传目标文档和参考文档后，这里会提示目标窗口、参考窗口和全量细检矩阵规模。
            </p>

            <div v-if="windowEstimate" class="window-recommendation">
              <span>系统建议</span>
              <strong>{{ windowEstimate.recommendation?.label }}</strong>
            </div>

            <p class="window-strategy-note">
              当前选择：{{ activeStrategy.eyebrow }}。如果快速模式最终候选覆盖全部参考文档，实际会与完整模式等价。
            </p>
          </article>
        </div>
      </div>

      <div v-if="modeModel === 'bert' && bgeStrategyModel === 'coarse_then_fine'" class="setting-group">
        <div class="setting-head">
          <p class="muted-label">Coarse Retrieval</p>
          <h3 class="setting-title">粗筛策略与参数</h3>
        </div>

        <div class="preset-band">
          <div class="preset-band__head">
            <div>
              <p class="muted-label">Strategy Presets</p>
              <h4 class="preset-band__title">粗筛策略预设</h4>
            </div>
            <span
              class="preset-state"
              :class="{ 'preset-state--custom': !activeCoarsePreset }"
            >
              当前档位: {{ activeCoarsePresetLabel }}
            </span>
          </div>

          <div class="preset-chip-row">
            <button
              v-for="preset in coarsePresetOptions"
              :key="preset.key"
              class="preset-chip"
              :class="{
                'preset-chip--active': activeCoarsePreset?.key === preset.key
              }"
              type="button"
              @click="applyCoarsePreset(preset)"
            >
              <strong>{{ preset.label }}</strong>
              <span>{{ preset.description }}</span>
            </button>
          </div>

          <article class="preset-guide">
            <div class="preset-guide__head">
              <div>
                <p class="muted-label">Preset Guide</p>
                <h5>{{ activeCoarsePresetGuide.label }}使用说明</h5>
              </div>
              <span class="preset-guide__focus">
                {{ activeCoarsePresetGuide.priority }}
              </span>
            </div>

            <p class="preset-guide__description">
              {{ activeCoarsePresetGuide.description }}
            </p>

            <div class="preset-guide__grid">
              <article class="preset-guide__card">
                <span>适合场景</span>
                <strong>{{ activeCoarsePresetGuide.bestFor }}</strong>
              </article>
              <article class="preset-guide__card">
                <span>策略取向</span>
                <strong>{{ activeCoarsePresetGuide.priority }}</strong>
              </article>
            </div>

            <div class="preset-guide__chips">
              <span
                v-for="change in activeCoarsePresetGuide.changes"
                :key="change"
                class="preset-guide__chip"
              >
                {{ change }}
              </span>
            </div>
          </article>
        </div>

        <details class="coarse-config-shell">
          <summary class="coarse-config-summary">
            <div>
              <strong>先选策略预设，再按需要微调候选池、阈值和同题扩容规则</strong>
              <span>如果你手动修改下面的任一参数，当前档位会自动切换成“自定义方案”。</span>
            </div>
            <button class="btn-ghost" type="button" @click.prevent="resetCoarseConfig">
              恢复默认
            </button>
          </summary>

          <div class="coarse-config-groups">
            <article
              v-for="group in coarseFieldGroups"
              :key="group.key"
              class="coarse-config-card"
            >
              <div class="coarse-config-card__head">
                <h4>{{ group.title }}</h4>
                <p>{{ group.description }}</p>
              </div>

              <div class="coarse-config-grid">
                <label
                  v-for="field in group.fields"
                  :key="field.key"
                  class="coarse-config-field"
                >
                  <span>{{ field.label }}</span>
                  <input
                    :value="props.coarseConfig[field.key]"
                    type="number"
                    :min="field.min"
                    :max="field.max"
                    :step="field.step"
                    @input="updateCoarseConfigField(field, $event)"
                  />
                </label>
              </div>
            </article>
          </div>
        </details>
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
        BGE 模式会按你选择的检测策略执行：快速模式先粗筛候选，完整模式则让所有参考文档进入细粒度匹配。
      </p>
    </section>

    <p v-if="notice" class="notice-ribbon">
      {{ notice }}
    </p>
  </section>
</template>
