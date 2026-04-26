<script setup>
import { computed } from "vue";
import { useRouter } from "vue-router";
import {
  ArrowLeft,
  ArrowRight,
  FileText,
  Gauge,
  Layers,
  Play,
  RotateCcw,
  SlidersHorizontal,
  Sparkles,
  Wand2
} from "lucide-vue-next";
import {
  coarseConfigGroups,
  coarseConfigPresets,
  detectCoarseConfigPreset
} from "../config/coarseRetrieval";
import { useTaskStore } from "../stores/task";

const task = useTaskStore();
const router = useRouter();

const modeOptions = [
  {
    value: "bert",
    label: "BGE 语义",
    description: "适合改写、洗稿和语义复核"
  },
  {
    value: "traditional",
    label: "传统白盒",
    description: "适合快速、可解释的词面比对"
  }
];

const profileOptions = [
  { value: "strict", label: "Strict", description: "低误报" },
  { value: "balanced", label: "Balanced", description: "日常默认" },
  { value: "recall", label: "Recall", description: "高召回" }
];

const strategyOptions = [
  {
    value: "coarse_then_fine",
    label: "粗筛后细检",
    description: "先召回候选，再做 BGE 窗口复核"
  },
  {
    value: "full_fine",
    label: "完整细检",
    description: "对全部参考文档做细粒度复核"
  }
];

const activePreset = computed(() => detectCoarseConfigPreset(task.coarseConfig));

const estimateMetrics = computed(() => {
  if (!task.windowEstimate) {
    return [];
  }

  return [
    ["目标窗口", task.windowEstimate.target_window_count],
    ["参考窗口", task.windowEstimate.reference_window_count],
    ["参考数量", task.windowEstimate.reference_count],
    ["窗口组合", task.windowEstimate.full_pair_count]
  ];
});

const formatInteger = (value) => Number(value || 0).toLocaleString("zh-CN");

const submit = async () => {
  const nextTaskId = await task.submitCheck();
  if (nextTaskId) {
    router.push("/results");
  }
};

const goUpload = () => {
  router.push("/upload");
};
</script>

<template>
  <section class="work-view configure-view">
    <header class="view-header">
      <div>
        <p class="view-kicker">Step 02</p>
        <h1>检测配置</h1>
      </div>
      <div class="header-actions">
        <button class="command-button" type="button" @click="goUpload">
          <ArrowLeft :size="18" />
          返回上传
        </button>
        <button class="command-button command-button--primary" type="button" :disabled="!task.canSubmit" @click="submit">
          <Play :size="18" />
          开始检测
        </button>
      </div>
    </header>

    <section v-if="!task.hasUploads" class="empty-action">
      <FileText />
      <h2>还没有完整的检测文件</h2>
      <p>需要一份待检测文档和至少一份参考文档，才能进入检测任务。</p>
      <button class="command-button command-button--primary" type="button" @click="goUpload">
        <ArrowRight :size="18" />
        去上传文档
      </button>
    </section>

    <template v-else>
      <div class="config-layout">
        <section class="config-main">
          <article class="config-section">
            <div class="section-titleline">
              <Sparkles :size="20" />
              <div>
                <span>检测引擎</span>
                <h2>选择分析方式</h2>
              </div>
            </div>

            <div class="segmented-grid">
              <button
                v-for="option in modeOptions"
                :key="option.value"
                class="option-tile"
                :class="{ 'option-tile--active': task.mode === option.value }"
                type="button"
                @click="task.setMode(option.value)"
              >
                <strong>{{ option.label }}</strong>
                <span>{{ option.description }}</span>
              </button>
            </div>
          </article>

          <article v-if="task.mode === 'bert'" class="config-section">
            <div class="section-titleline">
              <Gauge :size="20" />
              <div>
                <span>BGE Profile</span>
                <h2>阈值风格</h2>
              </div>
            </div>

            <div class="segmented-grid segmented-grid--three">
              <button
                v-for="option in profileOptions"
                :key="option.value"
                class="option-tile option-tile--compact"
                :class="{ 'option-tile--active': task.bertProfile === option.value }"
                type="button"
                @click="task.setBertProfile(option.value)"
              >
                <strong>{{ option.label }}</strong>
                <span>{{ option.description }}</span>
              </button>
            </div>
          </article>

          <article v-if="task.mode === 'bert'" class="config-section">
            <div class="section-titleline">
              <Layers :size="20" />
              <div>
                <span>检索策略</span>
                <h2>控制细检范围</h2>
              </div>
            </div>

            <div class="segmented-grid">
              <button
                v-for="option in strategyOptions"
                :key="option.value"
                class="option-tile"
                :class="{ 'option-tile--active': task.bgeStrategy === option.value }"
                type="button"
                @click="task.setBgeStrategy(option.value)"
              >
                <strong>{{ option.label }}</strong>
                <span>{{ option.description }}</span>
              </button>
            </div>
          </article>

          <article v-if="task.mode === 'bert' && task.bgeStrategy === 'coarse_then_fine'" class="config-section">
            <div class="section-titleline">
              <SlidersHorizontal :size="20" />
              <div>
                <span>粗筛参数</span>
                <h2>预设和高级参数</h2>
              </div>
            </div>

            <div class="preset-strip">
              <button
                v-for="preset in coarseConfigPresets"
                :key="preset.key"
                class="preset-button"
                :class="{ 'preset-button--active': activePreset?.key === preset.key }"
                type="button"
                @click="task.applyCoarseConfig(preset.config)"
              >
                {{ preset.label }}
              </button>
              <button class="preset-button preset-button--reset" type="button" @click="task.resetCoarseConfig">
                <RotateCcw :size="15" />
                重置
              </button>
            </div>

            <details class="advanced-config">
              <summary>
                <Wand2 :size="16" />
                展开高级参数
              </summary>
              <div class="config-fields">
                <section v-for="group in coarseConfigGroups" :key="group.key" class="field-group">
                  <h3>{{ group.title }}</h3>
                  <label v-for="field in group.fields" :key="field.key" class="number-field">
                    <span>{{ field.label }}</span>
                    <input
                      :type="field.type === 'int' ? 'number' : 'number'"
                      :min="field.min"
                      :max="field.max"
                      :step="field.step"
                      :value="task.coarseConfig[field.key]"
                      @input="task.updateCoarseConfigField(field.key, $event.target.value)"
                    />
                  </label>
                </section>
              </div>
            </details>
          </article>
        </section>

        <aside class="config-inspector">
          <section class="inspector-block">
            <span>输入摘要</span>
            <h2>{{ task.targetFile?.name }}</h2>
            <p>{{ task.refFiles.length }} 份参考文档</p>
          </section>

          <section class="inspector-block">
            <span>正文清洗</span>
            <button
              class="toggle-row"
              type="button"
              :class="{ 'toggle-row--on': task.bodyMode }"
              @click="task.setBodyMode(!task.bodyMode)"
            >
              <strong>{{ task.bodyMode ? "已开启" : "已关闭" }}</strong>
              <small>过滤封面、目录、图表和公式噪声</small>
            </button>
          </section>

          <section v-if="task.mode === 'bert'" class="inspector-block">
            <div class="inspector-line">
              <span>窗口估算</span>
              <button class="text-command" type="button" @click="task.requestWindowEstimate">
                重新估算
              </button>
            </div>

            <p v-if="task.windowEstimateLoading" class="estimate-state">正在估算窗口规模...</p>
            <p v-else-if="task.windowEstimateError" class="estimate-state estimate-state--error">
              {{ task.windowEstimateError }}
            </p>

            <div v-if="estimateMetrics.length" class="estimate-grid">
              <article v-for="[label, value] in estimateMetrics" :key="label">
                <span>{{ label }}</span>
                <strong>{{ formatInteger(value) }}</strong>
              </article>
            </div>
          </section>

          <button class="submit-wide" type="button" :disabled="!task.canSubmit" @click="submit">
            <Play :size="18" />
            {{ task.loading ? task.pollStatusMessage : "提交检测任务" }}
          </button>
        </aside>
      </div>
    </template>
  </section>
</template>
