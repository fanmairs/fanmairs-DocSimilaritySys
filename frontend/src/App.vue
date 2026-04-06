<script setup>
import { onBeforeUnmount, ref } from "vue";
import { api } from "./api/client";
import HeroBanner from "./components/HeroBanner.vue";
import ControlPanel from "./components/ControlPanel.vue";
import ResultsPanel from "./components/ResultsPanel.vue";
import PreviewModal from "./components/PreviewModal.vue";

const targetFile = ref(null);
const refFiles = ref([]);
const mode = ref("bert");
const bertProfile = ref("balanced");
const bodyMode = ref(true);
const loading = ref(false);
const pollStatusMessage = ref("排队中...");
const notice = ref("");
const results = ref(null);
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
    previewContent.value = "网络异常，无法预览文档。请确认后端服务是否可用。";
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
      pollStatusMessage.value = "模型计算中...";
      return;
    }

    if (taskStatus === "pending") {
      pollStatusMessage.value = "任务排队中...";
      return;
    }

    if (taskStatus === "completed") {
      stopPolling();
      loading.value = false;
      results.value = data.data || [];
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
    setNotice("请至少上传一篇参考文档。");
    return;
  }

  stopPolling();
  loading.value = true;
  results.value = null;
  costTime.value = 0;
  pollStatusMessage.value = "任务提交中...";

  const formData = new FormData();
  formData.append("target_file", targetFile.value);
  refFiles.value.forEach((file) => formData.append("reference_files", file));
  formData.append("mode", mode.value);
  formData.append("bert_profile", bertProfile.value);
  formData.append("body_mode", bodyMode.value);

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
    pollStatusMessage.value = data.message || "任务排队中...";
    pollTimer = setInterval(() => {
      pollTask(taskId);
    }, 2000);
  } catch (error) {
    loading.value = false;
    setNotice(`提交失败：${error.message || "后端服务不可用"}`);
    console.error(error);
  }
};

onBeforeUnmount(() => {
  stopPolling();
});
</script>

<template>
  <main class="mx-auto min-h-screen max-w-6xl px-4 py-8 sm:px-6 lg:px-8">
    <HeroBanner />
    <ControlPanel
      v-model:mode="mode"
      v-model:bertProfile="bertProfile"
      v-model:bodyMode="bodyMode"
      :target-file="targetFile"
      :ref-files="refFiles"
      :loading="loading"
      :poll-status-message="pollStatusMessage"
      :notice="notice"
      @target-selected="onTargetSelected"
      @refs-selected="onRefsSelected"
      @clear-target="clearTarget"
      @remove-ref="removeRef"
      @preview-file="previewFile"
      @submit="submitCheck"
    />
    <ResultsPanel :results="results" :mode="mode" :cost-time="costTime" />

    <PreviewModal
      :visible="previewVisible"
      :title="previewTitle"
      :content="previewContent"
      :loading="previewLoading"
      @close="closePreview"
    />
  </main>
</template>
