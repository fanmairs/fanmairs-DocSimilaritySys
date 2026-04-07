<script setup>
import { computed } from "vue";
import SimilarityBadge from "./SimilarityBadge.vue";

const props = defineProps({
  results: {
    type: Array,
    default: null
  },
  mode: {
    type: String,
    default: "bert"
  },
  costTime: {
    type: Number,
    default: 0
  },
  loading: {
    type: Boolean,
    default: false
  },
  pollStatusMessage: {
    type: String,
    default: ""
  }
});

const hasResults = computed(() => Array.isArray(props.results));
const resultRows = computed(() => props.results || []);
const topResult = computed(() => resultRows.value[0] || null);

const highestScore = computed(() => {
  if (!topResult.value) {
    return 0;
  }

  if (props.mode === "bert") {
    return topResult.value.sim_bert ?? topResult.value.sim_lsa ?? 0;
  }
  return topResult.value.risk_score ?? topResult.value.sim_hybrid ?? topResult.value.sim_lsa ?? 0;
});

const summaryText = computed(() => {
  if (highestScore.value > 0.7) {
    return "检测到高风险信号，建议优先复核首位参考文档与命中片段。";
  }
  if (highestScore.value > 0.35) {
    return "存在中等强度重合信号，建议结合覆盖率与片段证据继续判断。";
  }
  return "当前结果整体较平稳，暂未发现明显高危相似内容。";
});

const formatScore = (value) => `${((value || 0) * 100).toFixed(2)}%`;

const heroMetrics = computed(() => {
  if (!topResult.value) {
    return [];
  }

  if (props.mode === "bert") {
    return [
      {
        label: "总分",
        value: formatScore(topResult.value.sim_bert ?? topResult.value.sim_lsa ?? 0)
      },
      {
        label: "风险分",
        value: formatScore(topResult.value.sim_bert_risk ?? topResult.value.sim_bert ?? 0)
      },
      {
        label: "覆盖率",
        value: formatScore(topResult.value.sim_bert_coverage_effective ?? topResult.value.sim_bert_coverage ?? 0)
      },
      {
        label: "置信度",
        value: formatScore(topResult.value.sim_bert_confidence ?? 0)
      }
    ];
  }

  return [
    {
      label: "风险分",
      value: formatScore(topResult.value.risk_score ?? topResult.value.sim_hybrid ?? 0)
    },
    {
      label: "Hybrid",
      value: formatScore(topResult.value.sim_hybrid ?? topResult.value.sim_lsa ?? 0)
    },
    {
      label: "LSA",
      value: formatScore(topResult.value.sim_lsa ?? 0)
    },
    {
      label: "TF-IDF",
      value: formatScore(topResult.value.sim_tfidf ?? 0)
    }
  ];
});

const getRowMetrics = (item) => {
  if (props.mode === "bert") {
    return [
      { label: "总分", value: item.sim_bert ?? item.sim_lsa ?? 0 },
      { label: "风险", value: item.sim_bert_risk ?? item.sim_bert ?? 0 },
      {
        label: "覆盖率",
        value: item.sim_bert_coverage_effective ?? item.sim_bert_coverage ?? 0
      },
      { label: "置信度", value: item.sim_bert_confidence ?? 0 }
    ];
  }

  return [
    { label: "风险", value: item.risk_score ?? item.sim_hybrid ?? 0 },
    { label: "Hybrid", value: item.sim_hybrid ?? item.sim_lsa ?? 0 },
    { label: "LSA", value: item.sim_lsa ?? 0 },
    { label: "TF-IDF", value: item.sim_tfidf ?? 0 }
  ];
};

const getBadges = (item) => {
  if (props.mode === "bert") {
    return [
      { label: "总分", score: item.sim_bert ?? item.sim_lsa ?? 0 },
      { label: "风险", score: item.sim_bert_risk ?? item.sim_bert ?? 0 }
    ];
  }

  return [
    { label: "风险", score: item.risk_score ?? item.sim_hybrid ?? 0 },
    { label: "Hybrid", score: item.sim_hybrid ?? item.sim_lsa ?? 0 }
  ];
};

const getRowSummary = (item) => {
  const fragments = item.plagiarized_parts?.length || 0;
  if (props.mode === "bert") {
    return `命中 ${fragments} 个片段，覆盖率 ${formatScore(
      item.sim_bert_coverage_effective ?? item.sim_bert_coverage ?? 0
    )}。`;
  }
  return `命中 ${fragments} 个片段，Hybrid ${formatScore(item.sim_hybrid ?? item.sim_lsa ?? 0)}。`;
};
</script>

<template>
  <section class="surface-panel results-surface">
    <header class="results-header">
      <div>
        <p class="badge-kicker">Results Flow</p>
        <h2 class="panel-title">检测结果与证据流</h2>
        <p class="panel-subtitle">
          这里优先展示最值得复核的文档，再展开覆盖率、置信度和片段证据，避免结果只剩下一排离散卡片。
        </p>
      </div>

      <div class="results-header__side">
        <span class="state-pill" :class="{ 'state-pill--live': loading }">
          {{ loading ? pollStatusMessage || "分析中" : `耗时 ${costTime.toFixed(2)}s` }}
        </span>
        <span v-if="hasResults" class="results-count">
          {{ resultRows.length }} 条结果
        </span>
      </div>
    </header>

    <article v-if="!hasResults" class="results-empty">
      <div class="results-empty__bars">
        <span v-for="bar in 6" :key="bar" class="results-empty__bar"></span>
      </div>
      <h3>结果区已准备好</h3>
      <p>上传文档并发起检测后，这里会按优先级呈现排序结果、证据指标和命中片段。</p>
    </article>

    <article v-else-if="resultRows.length === 0" class="results-empty">
      <div class="results-empty__bars">
        <span v-for="bar in 6" :key="bar" class="results-empty__bar"></span>
      </div>
      <h3>本次没有可展示结果</h3>
      <p>请检查文档内容是否为空，或重新上传格式更稳定的文本文件后再试。</p>
    </article>

    <template v-else>
      <section class="results-hero">
        <div class="results-hero__lead">
          <p class="muted-label">Current Highlight</p>
          <div class="score-flare">
            <strong>{{ formatScore(highestScore) }}</strong>
            <span>{{ props.mode === "bert" ? "BERT 总分" : "风险分数" }}</span>
          </div>
          <p class="text-note">{{ summaryText }}</p>
        </div>

        <div class="results-metrics">
          <article v-for="metric in heroMetrics" :key="metric.label" class="metric-slab">
            <span>{{ metric.label }}</span>
            <strong>{{ metric.value }}</strong>
          </article>
        </div>
      </section>

      <ol class="result-stream">
        <li
          v-for="(item, index) in resultRows"
          :key="`${item.file}-${index}`"
          class="result-row animate-rise-in"
          :style="{ animationDelay: `${Math.min(index * 80, 400)}ms` }"
        >
          <div class="result-row__head">
            <div class="result-row__title">
              <span class="result-rank">TOP {{ index + 1 }}</span>
              <h3>{{ item.file }}</h3>
              <p>{{ getRowSummary(item) }}</p>
            </div>

            <div class="result-badges">
              <SimilarityBadge
                v-for="badge in getBadges(item)"
                :key="badge.label"
                :label="badge.label"
                :score="badge.score"
              />
            </div>
          </div>

          <div class="result-metrics-grid">
            <article
              v-for="metric in getRowMetrics(item)"
              :key="metric.label"
              class="mini-metric"
            >
              <span>{{ metric.label }}</span>
              <strong>{{ formatScore(metric.value) }}</strong>
            </article>
          </div>

          <details
            v-if="item.plagiarized_parts && item.plagiarized_parts.length"
            class="fragment-shell"
          >
            <summary>
              查看 {{ item.plagiarized_parts.length }} 个命中片段
            </summary>

            <div class="fragment-list">
              <article
                v-for="(part, partIndex) in item.plagiarized_parts.slice(0, 8)"
                :key="partIndex"
                class="fragment-pair"
              >
                <div class="fragment-side">
                  <p class="fragment-label">待检测文档片段</p>
                  <p>{{ part.target_part }}</p>
                </div>

                <div class="fragment-side fragment-side--accent">
                  <div class="fragment-side__head">
                    <p class="fragment-label">参考文档片段</p>
                    <span>{{ formatScore(part.score || 0) }}</span>
                  </div>
                  <p>{{ part.ref_part }}</p>
                </div>
              </article>
            </div>
          </details>

          <p v-else class="result-ok">
            当前参考文档未返回显著连续片段，建议先以整体分数作为初步参考。
          </p>
        </li>
      </ol>
    </template>
  </section>
</template>
