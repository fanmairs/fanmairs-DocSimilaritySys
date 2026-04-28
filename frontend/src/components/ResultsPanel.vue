<script setup>
import { computed } from "vue";
import SimilarityBadge from "./SimilarityBadge.vue";

const props = defineProps({
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
const activeSummary = computed(() => (props.mode === "bert" ? props.resultSummary : null));

const highestScore = computed(() => {
  if (activeSummary.value) {
    return activeSummary.value.global_score ?? 0;
  }

  if (!topResult.value) {
    return 0;
  }

  if (props.mode === "bert") {
    return topResult.value.sim_bert ?? topResult.value.sim_lsa ?? 0;
  }
  return topResult.value.risk_score ?? topResult.value.sim_hybrid ?? topResult.value.sim_lsa ?? 0;
});

const narrativeSignal = computed(() => {
  if (activeSummary.value) {
    return Math.max(
      activeSummary.value.global_score ?? 0,
      activeSummary.value.global_coverage_effective ?? 0
    );
  }

  if (!topResult.value) {
    return 0;
  }

  if (props.mode === "bert") {
    return Math.max(
      topResult.value.sim_bert ?? 0,
      topResult.value.sim_bert_risk ?? 0
    );
  }

  return highestScore.value;
});

const summaryText = computed(() => {
  if (activeSummary.value) {
    if (isCoarseStrategyExhausted.value) {
      return "本次选择了快速模式，但粗筛候选已经覆盖全部参考文档，因此实际执行效果与完整模式一致；全局总分基于所有参考文档的细检证据计算。";
    }
    if (
      (activeSummary.value.global_source_diversity ?? 0) > 0.45 &&
      (activeSummary.value.global_verified_source_count ?? 0) >= 2 &&
      (activeSummary.value.global_coverage_effective ?? 0) > 0.08
    ) {
      return "当前全局总分已经综合了多个参考来源的细检证据，存在多来源覆盖同一目标文档的风险，建议优先复核命中片段与来源分布。";
    }
    if (narrativeSignal.value > 0.7) {
      return "当前全局总分处于高风险区间，说明候选细检证据在去重后仍形成了较强覆盖与连续命中，建议优先复核。";
    }
    if (narrativeSignal.value > 0.35) {
      return "当前全局总分处于中等风险区间，建议结合去重覆盖率、证据置信度和多来源分布继续判断。";
    }
    return "当前全局总分整体较低，细检证据在去重后尚未形成明显高风险覆盖，可继续参考单篇结果做补充复核。";
  }

  if (
    props.mode === "bert" &&
    topResult.value &&
    (topResult.value.sim_bert_doc ?? 0) > 0.85 &&
    (topResult.value.sim_bert_coverage_effective ?? topResult.value.sim_bert_coverage ?? 0) < 0.05
  ) {
    return "文档级语义接近，但当前局部覆盖仍有限，建议结合风险分与命中片段继续复核。";
  }
  if (narrativeSignal.value > 0.7) {
    return "检测到高风险信号，建议优先复核首位参考文档与命中片段。";
  }
  if (narrativeSignal.value > 0.35) {
    return "存在中等强度重合信号，建议结合覆盖率与片段证据继续判断。";
  }
  return "当前结果整体较平稳，暂未发现明显高危相似内容。";
});

const formatScore = (value) => `${((value || 0) * 100).toFixed(2)}%`;

const formatInteger = (value) => Number(value || 0).toLocaleString("zh-CN");

const traditionalSemanticStatus = computed(() => {
  if (props.mode !== "traditional" || !topResult.value) {
    return null;
  }

  const mode = topResult.value.traditional_semantic_mode || "unknown";
  const enabled = Boolean(topResult.value.traditional_semantic_enabled);
  const vectorCoverage = Number(topResult.value.traditional_semantic_vector_coverage || 0);
  const vectorHits = Number(topResult.value.traditional_semantic_vector_hits || 0);
  const vocabSize = Number(topResult.value.traditional_semantic_vocab_size || 0);
  const synonymCount = Number(topResult.value.traditional_semantic_synonym_count || 0);
  const embeddingsConfigured = Boolean(topResult.value.traditional_semantic_embeddings_configured);
  const embeddingsFound = Boolean(topResult.value.traditional_semantic_embeddings_found);

  if (mode === "vector") {
    return {
      label: vectorCoverage >= 0.03 ? "词向量已生效" : "词向量覆盖较低",
      value: `${formatScore(vectorCoverage)} 覆盖`,
      tone: vectorCoverage >= 0.03 ? "good" : "warn",
      description: `当前语料词表 ${formatInteger(vocabSize)} 个词，词向量命中 ${formatInteger(vectorHits)} 个；Soft 分会参与同义替换和近义表达补偿。`
    };
  }

  if (mode === "synonym") {
    return {
      label: "同义词增强已生效",
      value: `${formatInteger(synonymCount)} 条映射`,
      tone: "good",
      description: "未使用词向量文件，但同义词表可用；Soft 分主要通过同义词归一化补偿改写表达。"
    };
  }

  if (!enabled && embeddingsConfigured && !embeddingsFound) {
    return {
      label: "词向量文件未找到",
      value: "软语义未启用",
      tone: "warn",
      description: "系统配置了词向量路径，但本地文件不存在；当前传统模式主要依赖 TF-IDF 与 LSA。"
    };
  }

  return {
    label: "软语义未启用",
    value: "仅数学基线",
    tone: "neutral",
    description: "当前没有可用词向量或同义词表；传统模式仍会使用 TF-IDF、LSA 和窗口片段检测。"
  };
});

const traditionalMethodCards = computed(() => {
  if (props.mode !== "traditional") {
    return [];
  }

  const requestedLsaComponents = Number(topResult.value?.traditional_lsa_components || 3);
  const effectiveLsaComponents = Number(
    topResult.value?.traditional_lsa_components_effective || requestedLsaComponents
  );
  const lsaComponentText =
    requestedLsaComponents === effectiveLsaComponents
      ? `${effectiveLsaComponents}D`
      : `${effectiveLsaComponents}/${requestedLsaComponents}D`;

  return [
    {
      label: "TF-IDF",
      value: formatScore(topResult.value?.sim_tfidf ?? 0),
      tone: "neutral",
      description: "衡量关键词和专业词的字面重合，适合发现直接复制、术语复用和局部照搬。"
    },
    {
      label: "LSA",
      value: formatScore(topResult.value?.sim_lsa ?? 0),
      tone: "neutral",
      description: "通过 SVD 把 TF-IDF 矩阵压缩到低维主题空间，捕捉同一主题下的潜在语义相似。"
    },
    {
      label: "Soft",
      value: formatScore(topResult.value?.sim_soft ?? 0),
      tone: traditionalSemanticStatus.value?.tone === "warn" ? "warn" : "neutral",
      description: "基于同义词表或词向量做软匹配：即使两个词没有完全相同，也会尝试判断它们是否属于近义表达。"
    },
    {
      label: "LSA 维度",
      value: lsaComponentText,
      tone: "neutral",
      description: "表示本次 SVD 压缩后保留的潜在主题数量；维度越高越细，维度越低越概括。"
    },
    {
      label: "Hybrid",
      value: formatScore(topResult.value?.sim_hybrid ?? 0),
      tone: "neutral",
      description: "融合 TF-IDF、LSA 与 Soft 软语义，避免只靠某一个指标误判。"
    },
    {
      label: "风险分",
      value: formatScore(topResult.value?.risk_score ?? 0),
      tone: "warn",
      description: "面向人工复核的最终排序信号，综合整体相似度和片段证据强度。"
    },
    {
      label: "软语义资源",
      value: traditionalSemanticStatus.value?.value || "未返回",
      tone: traditionalSemanticStatus.value?.tone || "neutral",
      description: traditionalSemanticStatus.value?.description || "等待后端返回词向量或同义词资源状态。"
    }
  ];
});

const isCoarseStrategyExhausted = computed(() => {
  if (!activeSummary.value || activeSummary.value.retrieval_strategy !== "coarse_then_fine") {
    return false;
  }

  const candidateCount = Number(activeSummary.value.global_candidate_count || 0);
  const referenceCount = Number(activeSummary.value.global_reference_count || 0);
  return referenceCount > 0 && candidateCount >= referenceCount;
});

const formatRetrievalStrategy = (summary) => {
  const candidateCount = Number(summary?.global_candidate_count || 0);
  const referenceCount = Number(summary?.global_reference_count || 0);
  const scope = referenceCount > 0 ? `${candidateCount}/${referenceCount}` : "";

  if (summary?.retrieval_strategy === "full_fine") {
    return scope ? `全部细检 ${scope}` : "全部细检";
  }

  if (isCoarseStrategyExhausted.value) {
    return scope ? `候选细检 ${scope}，等同全部` : "候选细检，等同全部";
  }

  return scope ? `候选细检 ${scope}` : "候选细检";
};

const heroMetrics = computed(() => {
  if (activeSummary.value) {
    return [
      {
        label: "检测范围",
        value: formatRetrievalStrategy(activeSummary.value)
      },
      {
        label: "全局总分",
        value: formatScore(activeSummary.value.global_score ?? 0)
      },
      {
        label: "去重覆盖",
        value: formatScore(activeSummary.value.global_coverage_effective ?? 0)
      },
      {
        label: "证据置信",
        value: formatScore(activeSummary.value.global_confidence ?? 0)
      },
      {
        label: "连续命中",
        value: formatScore(activeSummary.value.global_continuity_top3 ?? 0)
      },
      {
        label: "来源分散",
        value: formatScore(activeSummary.value.global_source_diversity ?? 0)
      }
    ];
  }

  if (!topResult.value) {
    return [];
  }

  if (props.mode === "bert") {
    return [
      {
        label: "综合相似",
        value: formatScore(topResult.value.sim_bert ?? topResult.value.sim_lsa ?? 0)
      },
      {
        label: "文档语义",
        value: formatScore(topResult.value.sim_bert_doc ?? 0)
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
    },
    {
      label: "Soft",
      value: formatScore(topResult.value.sim_soft ?? 0)
    },
    {
      label: "软语义资源",
      value: traditionalSemanticStatus.value?.value || "未返回"
    }
  ];
});

const getRowMetrics = (item) => {
  if (props.mode === "bert") {
    if (item.retrieval_stage === "coarse_only") {
      return [
        { label: "粗筛分", value: item.sim_bert_coarse ?? item.sim_bert ?? 0 },
        { label: "全文语义", value: item.sim_bert_coarse_doc ?? item.sim_bert_doc ?? 0 },
        { label: "段落热点", value: item.sim_bert_coarse_para ?? 0 },
        { label: "词面锚点", value: item.sim_bert_coarse_lex ?? 0 }
      ];
    }

    return [
      { label: "综合相似", value: item.sim_bert ?? item.sim_lsa ?? 0 },
      { label: "文档语义", value: item.sim_bert_doc ?? 0 },
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
    { label: "TF-IDF", value: item.sim_tfidf ?? 0 },
    { label: "Soft", value: item.sim_soft ?? 0 }
  ];
};

const getBadges = (item) => {
  if (props.mode === "bert") {
    if (item.retrieval_stage === "coarse_only") {
      return [
        { label: "粗筛", score: item.sim_bert_coarse ?? item.sim_bert ?? 0 },
        { label: "语义", score: item.sim_bert_coarse_doc ?? item.sim_bert_doc ?? 0 },
        { label: "热点", score: item.sim_bert_coarse_para ?? 0 }
      ];
    }

    return [
      { label: "综合", score: item.sim_bert ?? item.sim_lsa ?? 0 },
      { label: "语义", score: item.sim_bert_doc ?? 0 },
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
    if (item.retrieval_stage === "coarse_only") {
      return `粗筛排序结果，未进入细粒度复核。全文语义 ${formatScore(
        item.sim_bert_coarse_doc ?? item.sim_bert_doc ?? 0
      )}，段落热点 ${formatScore(item.sim_bert_coarse_para ?? 0)}，词面锚点 ${formatScore(
        item.sim_bert_coarse_lex ?? 0
      )}。`;
    }

    return `命中 ${fragments} 个片段，文档语义 ${formatScore(
      item.sim_bert_doc ?? 0
    )}，覆盖率 ${formatScore(item.sim_bert_coverage_effective ?? item.sim_bert_coverage ?? 0)}。`;
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
          <p class="muted-label">{{ activeSummary ? "Global Summary" : "Current Highlight" }}</p>
          <div class="score-flare">
            <strong>{{ formatScore(highestScore) }}</strong>
            <span>{{ activeSummary ? "全局总分" : props.mode === "bert" ? "综合相似" : "风险分数" }}</span>
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

      <section v-if="props.mode === 'traditional'" class="method-explainer">
        <article
          v-for="card in traditionalMethodCards"
          :key="card.label"
          class="method-card"
          :class="`method-card--${card.tone}`"
        >
          <span>{{ card.label }}</span>
          <strong>{{ card.value }}</strong>
          <p>{{ card.description }}</p>
        </article>
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
