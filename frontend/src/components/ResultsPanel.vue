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
  }
});

const highestScore = computed(() => {
  if (!props.results || !props.results.length) {
    return 0;
  }
  const top = props.results[0];
  if (props.mode === "bert") {
    return top.sim_bert ?? top.sim_lsa ?? 0;
  }
  return top.risk_score ?? top.sim_hybrid ?? top.sim_lsa ?? 0;
});

const summaryText = computed(() => {
  if (highestScore.value > 0.7) {
    return "结论：存在高度相似内容，建议重点人工复核。";
  }
  if (highestScore.value > 0.35) {
    return "结论：存在中等程度语义重合，建议进行片段级核查。";
  }
  return "结论：整体风险较低，未发现明显高危相似内容。";
});
</script>

<template>
  <section v-if="results" class="space-y-5">
    <header class="surface-card px-6 py-5">
      <div class="flex flex-wrap items-center justify-between gap-3">
        <h2 class="panel-title">查重报告</h2>
        <p class="rounded-full bg-ink-900 px-3 py-1 text-xs font-semibold tracking-wide text-white">
          耗时 {{ costTime.toFixed(2) }}s
        </p>
      </div>
      <p class="mt-3 text-note">{{ summaryText }}</p>
      <p class="mt-2 text-xs text-ink-900/65">
        指标备注：`风险`=告警分数（用于结论）；`总分`=Hybrid 综合相似度（用于最终排序）；`LSA`=潜在语义相似度；`TF-IDF`=字面词频相似度；`Soft`=词向量/同义词软匹配相似度（未加载词向量时常为 0）。
      </p>
    </header>

    <article v-if="results.length === 0" class="surface-card px-6 py-6 text-note">
      本次未生成可展示结果，请检查文档内容是否为空或格式异常。
    </article>

    <article
      v-for="(item, index) in results"
      :key="`${item.file}-${index}`"
      class="surface-card animate-rise-in overflow-hidden"
      :style="{ animationDelay: `${Math.min(index * 80, 400)}ms` }"
    >
      <div class="border-b border-mint-500/15 bg-paper-50/80 px-6 py-4">
        <div class="flex flex-wrap items-center justify-between gap-3">
          <h3 class="font-display text-lg font-extrabold text-ink-900">
            TOP {{ index + 1 }} · {{ item.file }}
          </h3>
          <div class="flex flex-wrap gap-2">
            <SimilarityBadge v-if="mode === 'bert'" label="BERT" :score="item.sim_bert ?? item.sim_lsa ?? 0" />
            <SimilarityBadge v-if="mode !== 'bert'" label="风险" :score="item.risk_score ?? item.sim_hybrid ?? item.sim_lsa ?? 0" />
            <SimilarityBadge v-if="mode !== 'bert'" label="总分(Hybrid)" :score="item.sim_hybrid ?? item.sim_lsa ?? 0" />
            <SimilarityBadge v-if="mode !== 'bert'" label="LSA" :score="item.sim_lsa ?? 0" />
            <SimilarityBadge v-if="mode !== 'bert'" label="TF-IDF" :score="item.sim_tfidf ?? 0" />
            <SimilarityBadge v-if="mode !== 'bert'" label="Soft" :score="item.sim_soft ?? 0" />
          </div>
        </div>
      </div>

      <div class="px-6 py-5">
        <div v-if="item.plagiarized_parts && item.plagiarized_parts.length" class="space-y-3">
          <details class="group rounded-2xl border border-amber-300/45 bg-white/80 p-4">
            <summary class="flex cursor-pointer list-none items-center justify-between gap-3">
              <div class="space-y-1">
                <p class="text-sm font-semibold text-ink-900">
                  命中 {{ item.plagiarized_parts.length }} 个高风险片段（展示前 12 个）
                </p>
                <p class="text-xs text-ink-900/65">
                  最高相似度 {{ ((item.plagiarized_parts[0]?.score || 0) * 100).toFixed(2) }}%
                </p>
              </div>
              <div class="inline-flex items-center gap-2 rounded-full border border-mint-500/25 bg-mint-500/5 px-3 py-1.5 text-xs font-semibold text-mint-600">
                点击展开
                <svg
                  class="h-3.5 w-3.5 transition group-open:rotate-180"
                  viewBox="0 0 20 20"
                  fill="currentColor"
                  aria-hidden="true"
                >
                  <path fill-rule="evenodd" d="M5.23 7.21a.75.75 0 011.06.02L10 11.168l3.71-3.938a.75.75 0 111.08 1.04l-4.25 4.51a.75.75 0 01-1.08 0l-4.25-4.51a.75.75 0 01.02-1.06z" clip-rule="evenodd" />
                </svg>
              </div>
            </summary>

            <div class="mt-4 space-y-3">
              <div
                v-for="(part, partIndex) in item.plagiarized_parts.slice(0, 12)"
                :key="partIndex"
                class="grid gap-0 rounded-2xl border border-amber-300/45 bg-white md:grid-cols-2"
              >
                <div class="border-b border-amber-200/70 p-4 md:border-b-0 md:border-r">
                  <p class="mb-2 text-xs font-bold uppercase tracking-wider text-ink-900/60">待测文档</p>
                  <p class="font-prose text-sm leading-7 text-ink-900">{{ part.target_part }}</p>
                </div>
                <div class="bg-amber-50/70 p-4">
                  <div class="mb-2 flex items-center justify-between gap-2">
                    <p class="text-xs font-bold uppercase tracking-wider text-amber-700">参考文档片段</p>
                    <span class="rounded-full bg-white px-2 py-1 text-xs font-semibold text-amber-700">
                      {{ ((part.score || 0) * 100).toFixed(2) }}%
                    </span>
                  </div>
                  <p class="font-prose text-sm leading-7 text-amber-900">{{ part.ref_part }}</p>
                </div>
              </div>
            </div>
          </details>
        </div>

        <p v-else class="rounded-xl border border-emerald-200 bg-emerald-100/50 px-4 py-3 text-sm font-semibold text-emerald-700">
          当前参考文档未发现明显连续高相似片段。
        </p>
      </div>
    </article>
  </section>
</template>
