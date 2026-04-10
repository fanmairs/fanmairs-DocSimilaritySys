<script setup>
import { computed } from "vue";

const props = defineProps({
  targetFile: {
    type: Object,
    default: null
  },
  refFiles: {
    type: Array,
    default: () => []
  },
  results: {
    type: Array,
    default: null
  },
  resultSummary: {
    type: Object,
    default: null
  },
  mode: {
    type: String,
    default: "bert"
  },
  loading: {
    type: Boolean,
    default: false
  },
  bertProfile: {
    type: String,
    default: "balanced"
  }
});

const modeLabel = computed(() => {
  return props.mode === "bert" ? "BGE 深度语义" : "TF-IDF + LSA";
});

const profileLabel = computed(() => {
  if (props.mode !== "bert") {
    return "传统比对模式";
  }
  return {
    strict: "Strict 低误报",
    balanced: "Balanced 日常模式",
    recall: "Recall 高召回"
  }[props.bertProfile] || "Balanced 日常模式";
});

const sessionLabel = computed(() => {
  if (props.loading) {
    return "分析进行中";
  }
  if (Array.isArray(props.results)) {
    return props.results.length ? "结果已生成" : "任务已完成";
  }
  return "等待启动";
});

const targetLabel = computed(() => {
  return props.targetFile?.name || "尚未选择待检测文档";
});

const topScore = computed(() => {
  if (props.mode === "bert" && props.resultSummary) {
    return props.resultSummary.global_score ?? 0;
  }

  if (!Array.isArray(props.results) || !props.results.length) {
    return 0.82;
  }

  const top = props.results[0];
  if (props.mode === "bert") {
    return top.sim_bert ?? top.sim_lsa ?? 0;
  }
  return top.risk_score ?? top.sim_hybrid ?? top.sim_lsa ?? 0;
});

const scoreLabel = computed(() => `${(topScore.value * 100).toFixed(2)}%`);

const signalBars = computed(() => {
  if (!Array.isArray(props.results) || !props.results.length) {
    return [0.36, 0.54, 0.7, 0.48, 0.82, 0.58];
  }

  const rowBars = props.results
    .slice(0, 6)
    .map((item) => {
      if (props.mode === "bert") {
        return item.sim_bert ?? item.sim_lsa ?? 0;
      }
      return item.risk_score ?? item.sim_hybrid ?? item.sim_lsa ?? 0;
    });

  if (props.mode === "bert" && props.resultSummary) {
    return [
      props.resultSummary.global_score ?? 0,
      props.resultSummary.global_coverage_effective ?? 0,
      props.resultSummary.global_confidence ?? 0,
      props.resultSummary.global_source_diversity ?? 0,
      ...rowBars,
    ].slice(0, 6);
  }

  return rowBars;
});
</script>

<template>
  <section class="hero-slab">
    <div class="hero-slab__glow hero-slab__glow--amber"></div>
    <div class="hero-slab__glow hero-slab__glow--mint"></div>
    <div class="hero-slab__mesh"></div>
    <div class="hero-sakura" aria-hidden="true">
      <span v-for="petal in 12" :key="petal" class="hero-sakura__petal"></span>
    </div>

    <div class="hero-layout">
      <div class="hero-copy">
        <p class="hero-kicker">Document Similarity Studio</p>
        <h1 class="hero-title">把文档查重做成一台有反馈、有节奏的语义扫描仪</h1>
        <p class="hero-summary">
          上传待检测文档与参考库后，系统会自动完成相似度排序、风险判断和命中片段回溯，让复核流程集中在同一块工作台里完成。
        </p>

        <div class="hero-chip-row">
          <span class="hero-chip">{{ sessionLabel }}</span>
          <span class="hero-chip hero-chip--soft">{{ modeLabel }}</span>
          <span class="hero-chip hero-chip--soft">{{ profileLabel }}</span>
        </div>

        <div class="hero-steps">
          <article class="hero-step">
            <span>01</span>
            <div>
              <p>上传待检测文档</p>
              <small>支持 txt / docx / pdf</small>
            </div>
          </article>
          <article class="hero-step">
            <span>02</span>
            <div>
              <p>切换引擎与阈值策略</p>
              <small>支持 strict / balanced / recall</small>
            </div>
          </article>
          <article class="hero-step">
            <span>03</span>
            <div>
              <p>复核结果与命中片段</p>
              <small>从排序直接进入局部证据</small>
            </div>
          </article>
        </div>
      </div>

      <aside class="hero-console">
        <div class="hero-console__scanline"></div>

        <div class="hero-console__header">
          <div>
            <p class="hero-console__eyebrow">Live Session</p>
            <h2 class="hero-console__title">当前检测场景</h2>
          </div>
          <span class="hero-console__badge">{{ sessionLabel }}</span>
        </div>

        <div class="hero-console__grid">
          <article class="console-metric console-metric--wide">
            <span>待检测文档</span>
            <strong>{{ targetLabel }}</strong>
          </article>
          <article class="console-metric">
            <span>参考文档数</span>
            <strong>{{ refFiles.length }}</strong>
          </article>
          <article class="console-metric">
            <span>当前模式</span>
            <strong>{{ props.mode === "bert" ? "BERT" : "传统" }}</strong>
          </article>
        </div>

        <div class="signal-wall">
          <div class="signal-wall__meta">
            <div>
              <p class="signal-wall__eyebrow">Signal Overview</p>
              <strong>{{ scoreLabel }}</strong>
            </div>
            <p>当前屏幕上最强的一路结果信号</p>
          </div>

          <div class="signal-wall__bars">
            <span
              v-for="(bar, index) in signalBars"
              :key="index"
              class="signal-wall__bar"
              :style="{ height: `${Math.max(22, bar * 100)}%`, animationDelay: `${index * 90}ms` }"
            ></span>
          </div>
        </div>
      </aside>
    </div>
  </section>
</template>
