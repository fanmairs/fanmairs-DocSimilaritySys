<script setup>
import { computed, ref, watch } from "vue";
import { ArrowLeft, ArrowRight, FileSearch, LocateFixed } from "lucide-vue-next";

const props = defineProps({
  results: {
    type: Array,
    default: null
  },
  mode: {
    type: String,
    default: "bert"
  }
});

const selectedKey = ref("");

const formatScore = (value) => `${((Number(value) || 0) * 100).toFixed(2)}%`;

const evidenceItems = computed(() => {
  if (!Array.isArray(props.results)) {
    return [];
  }

  return props.results.flatMap((result, resultIndex) => {
    const parts = Array.isArray(result.plagiarized_parts) ? result.plagiarized_parts : [];
    return parts.map((part, partIndex) => ({
      key: `${result.file}-${resultIndex}-${partIndex}`,
      file: result.file || "unknown",
      resultIndex,
      partIndex,
      score: Number(part.score || part.raw_score || 0),
      confidence: Number(part.confidence || part.score || 0),
      matchType: part.match_type || (props.mode === "bert" ? "semantic_window" : "traditional_window"),
      targetText: part.target_part || "",
      referenceText: part.ref_part || part.reference_part || "",
      targetStart: part.target_start,
      targetEnd: part.target_end,
      refStart: part.ref_start ?? part.reference_start,
      refEnd: part.ref_end ?? part.reference_end,
      ruleFlags: Array.isArray(part.rule_flags) ? part.rule_flags : []
    }));
  });
});

const selectedIndex = computed(() => {
  const index = evidenceItems.value.findIndex((item) => item.key === selectedKey.value);
  return index >= 0 ? index : 0;
});

const selectedItem = computed(() => evidenceItems.value[selectedIndex.value] || null);

const evidenceStats = computed(() => {
  const items = evidenceItems.value;
  if (!items.length) {
    return [
      { label: "命中片段", value: "0" },
      { label: "最高分", value: "0.00%" },
      { label: "平均置信", value: "0.00%" }
    ];
  }

  const topScore = Math.max(...items.map((item) => item.score));
  const avgConfidence =
    items.reduce((total, item) => total + item.confidence, 0) / Math.max(1, items.length);

  return [
    { label: "命中片段", value: String(items.length) },
    { label: "最高分", value: formatScore(topScore) },
    { label: "平均置信", value: formatScore(avgConfidence) }
  ];
});

const sourceGroups = computed(() => {
  const groups = new Map();
  evidenceItems.value.forEach((item) => {
    const current = groups.get(item.file) || {
      file: item.file,
      count: 0,
      topScore: 0
    };
    current.count += 1;
    current.topScore = Math.max(current.topScore, item.score);
    groups.set(item.file, current);
  });
  return Array.from(groups.values()).sort((left, right) => right.topScore - left.topScore);
});

const currentSourceItems = computed(() => {
  if (!selectedItem.value) {
    return [];
  }
  return evidenceItems.value.filter((item) => item.file === selectedItem.value.file);
});

const selectItem = (item) => {
  selectedKey.value = item.key;
};

const selectSource = (file) => {
  const first = evidenceItems.value.find((item) => item.file === file);
  if (first) {
    selectItem(first);
  }
};

const moveSelection = (direction) => {
  if (!evidenceItems.value.length) {
    return;
  }

  const nextIndex =
    (selectedIndex.value + direction + evidenceItems.value.length) % evidenceItems.value.length;
  selectedKey.value = evidenceItems.value[nextIndex].key;
};

watch(
  evidenceItems,
  (items) => {
    if (!items.length) {
      selectedKey.value = "";
      return;
    }

    if (!items.some((item) => item.key === selectedKey.value)) {
      selectedKey.value = items[0].key;
    }
  },
  { immediate: true }
);
</script>

<template>
  <section class="evidence-review">
    <header class="evidence-review__header">
      <div>
        <p class="view-kicker">Evidence Desk</p>
        <h2>证据对照复核</h2>
      </div>

      <div class="evidence-stats">
        <article v-for="stat in evidenceStats" :key="stat.label">
          <span>{{ stat.label }}</span>
          <strong>{{ stat.value }}</strong>
        </article>
      </div>
    </header>

    <section v-if="!evidenceItems.length" class="evidence-empty">
      <FileSearch :size="32" />
      <h3>暂无可对照片段</h3>
      <p>当前结果没有返回具体命中片段，可以先查看整体分数和来源排序。</p>
    </section>

    <section v-else class="evidence-grid">
      <aside class="evidence-sources">
        <p class="evidence-column-label">来源文档</p>
        <button
          v-for="source in sourceGroups"
          :key="source.file"
          class="evidence-source"
          :class="{ 'evidence-source--active': selectedItem?.file === source.file }"
          type="button"
          @click="selectSource(source.file)"
        >
          <strong>{{ source.file }}</strong>
          <span>{{ source.count }} 个片段 · {{ formatScore(source.topScore) }}</span>
        </button>
      </aside>

      <section class="evidence-reader">
        <div class="evidence-reader__toolbar">
          <div>
            <span>当前片段</span>
            <strong>{{ selectedIndex + 1 }} / {{ evidenceItems.length }}</strong>
          </div>

          <div class="evidence-reader__actions">
            <button class="icon-command" type="button" title="上一个片段" @click="moveSelection(-1)">
              <ArrowLeft :size="17" />
            </button>
            <button class="icon-command" type="button" title="下一个片段" @click="moveSelection(1)">
              <ArrowRight :size="17" />
            </button>
          </div>
        </div>

        <div class="evidence-compare">
          <article class="evidence-pane">
            <div class="evidence-pane__head">
              <span>待检测文本</span>
              <small v-if="selectedItem?.targetStart !== null && selectedItem?.targetStart !== undefined">
                {{ selectedItem.targetStart }} - {{ selectedItem.targetEnd }}
              </small>
            </div>
            <p>{{ selectedItem?.targetText }}</p>
          </article>

          <article class="evidence-pane evidence-pane--reference">
            <div class="evidence-pane__head">
              <span>参考文本</span>
              <small v-if="selectedItem?.refStart !== null && selectedItem?.refStart !== undefined">
                {{ selectedItem.refStart }} - {{ selectedItem.refEnd }}
              </small>
            </div>
            <p>{{ selectedItem?.referenceText }}</p>
          </article>
        </div>

        <footer class="evidence-meta">
          <span>
            <LocateFixed :size="15" />
            {{ selectedItem?.matchType }}
          </span>
          <strong>片段分 {{ formatScore(selectedItem?.score) }}</strong>
          <strong>置信度 {{ formatScore(selectedItem?.confidence) }}</strong>
        </footer>
      </section>

      <aside class="evidence-fragments">
        <p class="evidence-column-label">同来源片段</p>
        <button
          v-for="item in currentSourceItems"
          :key="item.key"
          class="fragment-index"
          :class="{ 'fragment-index--active': selectedItem?.key === item.key }"
          type="button"
          @click="selectItem(item)"
        >
          <span>#{{ item.partIndex + 1 }}</span>
          <strong>{{ formatScore(item.score) }}</strong>
          <small>{{ item.targetText.slice(0, 46) || "空片段" }}</small>
        </button>
      </aside>
    </section>
  </section>
</template>
