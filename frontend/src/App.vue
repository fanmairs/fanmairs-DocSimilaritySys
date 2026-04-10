<script setup>
import { onBeforeUnmount, onMounted, ref } from "vue";
import { api } from "./api/client";
import {
  cloneCoarseConfig,
  coarseConfigDefaults,
  sanitizeCoarseConfig
} from "./config/coarseRetrieval";
import HeroBanner from "./components/HeroBanner.vue";
import ControlPanel from "./components/ControlPanel.vue";
import ResultsPanel from "./components/ResultsPanel.vue";
import PreviewModal from "./components/PreviewModal.vue";

const targetFile = ref(null);
const refFiles = ref([]);
const mode = ref("bert");
const bertProfile = ref("balanced");
const bodyMode = ref(true);
const defaultCoarseConfig = ref(cloneCoarseConfig(coarseConfigDefaults));
const coarseConfig = ref(cloneCoarseConfig(coarseConfigDefaults));
const loading = ref(false);
const pollStatusMessage = ref("等待任务调度");
const notice = ref("");
const results = ref(null);
const resultSummary = ref(null);
const costTime = ref(0);

const previewVisible = ref(false);
const previewTitle = ref("");
const previewContent = ref("");
const previewLoading = ref(false);

let pollTimer = null;

const stopPolling = () => {
  if (pollTimer) {
    clearInterval(pollTimer);
    pollTimer = null;
  }
};

const setNotice = (message) => {
  notice.value = message;
  if (message) {
    setTimeout(() => {
      if (notice.value === message) {
        notice.value = "";
      }
    }, 4000);
  }
};

const onTargetSelected = (file) => {
  targetFile.value = file;
};

const onRefsSelected = (files) => {
  refFiles.value = files;
};

const clearTarget = () => {
  targetFile.value = null;
};

const removeRef = (index) => {
  refFiles.value.splice(index, 1);
};

const resetCoarseConfig = () => {
  coarseConfig.value = cloneCoarseConfig(defaultCoarseConfig.value);
};

const hydrateCoarseConfigDefaults = async () => {
  try {
    const { data } = await api.get("/api/coarse_config_defaults");
    if (data.status === "success" && data.defaults) {
      const hydratedDefaults = sanitizeCoarseConfig(data.defaults);
      defaultCoarseConfig.value = hydratedDefaults;
      coarseConfig.value = cloneCoarseConfig(hydratedDefaults);
    }
  } catch (error) {
    console.warn("Failed to load coarse retrieval defaults.", error);
  }
};

const closePreview = () => {
  previewVisible.value = false;
  previewTitle.value = "";
  previewContent.value = "";
  previewLoading.value = false;
};

const previewFile = async (file) => {
  if (!file) {
    return;
  }

  previewVisible.value = true;
  previewTitle.value = file.name;
  previewContent.value = "";
  previewLoading.value = true;

  const formData = new FormData();
  formData.append("file", file);

  try {
    const { data } = await api.post("/api/preview_document", formData, {
      headers: {
        "Content-Type": "multipart/form-data"
      }
    });

    if (data.status === "success") {
      previewContent.value = data.content || "文档内容为空。";
    } else {
      previewContent.value = `解析失败：${data.message || "未知错误"}`;
    }
  } catch (error) {
    previewContent.value = "网络异常，无法预览文档，请确认后端服务是否正常运行。";
    console.error(error);
  } finally {
    previewLoading.value = false;
  }
};

const pollTask = async (taskId) => {
  try {
    const { data } = await api.get(`/api/task_status/${taskId}`);
    if (data.status !== "success") {
      throw new Error(data.message || "任务查询失败");
    }

    const taskStatus = data.task_status;
    if (taskStatus === "processing") {
      pollStatusMessage.value = "语义引擎正在分析";
      return;
    }

    if (taskStatus === "pending") {
      pollStatusMessage.value = "任务排队中";
      return;
    }

    if (taskStatus === "completed") {
      stopPolling();
      loading.value = false;
      if (Array.isArray(data.data)) {
        results.value = data.data;
        resultSummary.value = null;
      } else {
        results.value = data.data?.items || [];
        resultSummary.value = data.data?.summary || null;
      }
      costTime.value = data.cost_time || 0;
      setNotice("检测完成，结果已更新。");
      return;
    }

    if (taskStatus === "failed") {
      stopPolling();
      loading.value = false;
      costTime.value = data.cost_time || 0;
      setNotice(`任务失败：${data.message || "未知错误"}`);
    }
  } catch (error) {
    stopPolling();
    loading.value = false;
    setNotice("任务轮询失败，请稍后重试。");
    console.error(error);
  }
};

const submitCheck = async () => {
  if (!targetFile.value) {
    setNotice("请先上传待检测文档。");
    return;
  }

  if (!refFiles.value.length) {
    setNotice("请至少上传一份参考文档。");
    return;
  }

  stopPolling();
  loading.value = true;
  results.value = null;
  resultSummary.value = null;
  costTime.value = 0;
  pollStatusMessage.value = "任务提交中";

  const formData = new FormData();
  formData.append("target_file", targetFile.value);
  refFiles.value.forEach((file) => formData.append("reference_files", file));
  formData.append("mode", mode.value);
  formData.append("bert_profile", bertProfile.value);
  formData.append("body_mode", bodyMode.value);
  if (mode.value === "bert") {
    formData.append(
      "coarse_config",
      JSON.stringify(sanitizeCoarseConfig(coarseConfig.value))
    );
  }

  try {
    const { data } = await api.post("/api/submit_task", formData, {
      headers: {
        "Content-Type": "multipart/form-data"
      }
    });

    if (data.status !== "success") {
      throw new Error(data.message || "任务提交失败");
    }

    const taskId = data.task_id;
    pollStatusMessage.value = data.message || "任务已提交，等待调度。";
    pollTimer = setInterval(() => {
      pollTask(taskId);
    }, 2000);
  } catch (error) {
    loading.value = false;
    setNotice(`提交失败：${error.message || "后端服务不可用"}`);
    console.error(error);
  }
};

onMounted(() => {
  hydrateCoarseConfigDefaults();
});

onBeforeUnmount(() => {
  stopPolling();
});
</script>

<template>
  <main class="app-shell">
    <HeroBanner
      :target-file="targetFile"
      :ref-files="refFiles"
      :results="results"
      :result-summary="resultSummary"
      :mode="mode"
      :loading="loading"
      :bert-profile="bertProfile"
    />

    <section class="workspace-stage">
      <div class="workspace-stage__intro">
        <div class="workspace-stage__copy">
          <p class="badge-kicker">Semantic Workbench</p>
          <h2 class="stage-title">把上传、检测、复核放进同一块扫描工作台</h2>
          <p class="text-note">
            左侧处理文档与检测参数，右侧集中查看排序结果、风险指标和命中片段，不再把信息切成零散小卡片。
          </p>
        </div>

        <div class="stage-strip">
          <span class="stage-pill">多格式文档输入</span>
          <span class="stage-pill">异步队列检测</span>
          <span class="stage-pill">片段级复核</span>
        </div>
      </div>

      <section class="workspace-layout">
        <ControlPanel
          class="workspace-control"
          v-model:mode="mode"
          v-model:bertProfile="bertProfile"
          v-model:bodyMode="bodyMode"
          v-model:coarseConfig="coarseConfig"
          :default-coarse-config="defaultCoarseConfig"
          :target-file="targetFile"
          :ref-files="refFiles"
          :loading="loading"
          :poll-status-message="pollStatusMessage"
          :notice="notice"
          @reset-coarse-config="resetCoarseConfig"
          @target-selected="onTargetSelected"
          @refs-selected="onRefsSelected"
          @clear-target="clearTarget"
          @remove-ref="removeRef"
          @preview-file="previewFile"
          @submit="submitCheck"
        />

        <ResultsPanel
          :results="results"
          :result-summary="resultSummary"
          :mode="mode"
          :cost-time="costTime"
          :loading="loading"
          :poll-status-message="pollStatusMessage"
        />
      </section>
    </section>

    <section class="operating-strip">
      <article class="operating-note">
        <p class="operating-note__eyebrow">Dual Engines</p>
        <h3>深度语义与传统规则同步可用</h3>
        <p>需要查洗稿时切到 BGE，需要快速初筛时使用 TF-IDF + LSA。</p>
      </article>

      <article class="operating-note">
        <p class="operating-note__eyebrow">Review Ready</p>
        <h3>结果不是一个分数，而是一条复核链</h3>
        <p>从排序、覆盖率、置信度到命中片段，都可以在同一屏完成判断。</p>
      </article>

      <article class="operating-note">
        <p class="operating-note__eyebrow">Queue Driven</p>
        <h3>长文本检测也保持稳定节奏</h3>
        <p>前端实时感知任务状态，避免把等待过程做成无反馈的白屏。</p>
      </article>
    </section>

    <PreviewModal
      :visible="previewVisible"
      :title="previewTitle"
      :content="previewContent"
      :loading="previewLoading"
      @close="closePreview"
    />
  </main>
</template>
