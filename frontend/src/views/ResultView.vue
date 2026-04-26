<script setup>
import { useRouter } from "vue-router";
import { ArrowLeft, FileSearch, Play, UploadCloud } from "lucide-vue-next";
import EvidenceReview from "../components/EvidenceReview.vue";
import ResultsPanel from "../components/ResultsPanel.vue";
import { useTaskStore } from "../stores/task";

const task = useTaskStore();
const router = useRouter();

const submitAgain = async () => {
  const nextTaskId = await task.submitCheck();
  if (nextTaskId) {
    router.push("/results");
  }
};
</script>

<template>
  <section class="work-view result-view">
    <header class="view-header">
      <div>
        <p class="view-kicker">Step 03</p>
        <h1>结果复核</h1>
      </div>
      <div class="header-actions">
        <button class="command-button" type="button" @click="router.push('/upload')">
          <UploadCloud :size="18" />
          重新上传
        </button>
        <button class="command-button" type="button" @click="router.push('/detect')">
          <ArrowLeft :size="18" />
          调整配置
        </button>
        <button class="command-button command-button--primary" type="button" :disabled="!task.canSubmit" @click="submitAgain">
          <Play :size="18" />
          重新检测
        </button>
      </div>
    </header>

    <section v-if="!task.hasUploads" class="empty-action">
      <FileSearch :size="34" />
      <h2>还没有可复核的任务</h2>
      <p>上传目标文档和参考文档后，结果会在这里按风险和证据强度展开。</p>
      <button class="command-button command-button--primary" type="button" @click="router.push('/upload')">
        <UploadCloud :size="18" />
        去上传文档
      </button>
    </section>

    <template v-else>
      <EvidenceReview
        v-if="Array.isArray(task.results) && task.results.length"
        class="result-focus"
        :results="task.results"
        :mode="task.mode"
      />

      <ResultsPanel
        :results="task.results"
        :result-summary="task.resultSummary"
        :mode="task.mode"
        :cost-time="task.costTime"
        :loading="task.loading"
        :poll-status-message="task.pollStatusMessage"
      />
    </template>
  </section>
</template>
