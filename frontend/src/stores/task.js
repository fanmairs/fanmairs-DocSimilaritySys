import { computed, ref } from "vue";
import { defineStore } from "pinia";
import { api } from "../api/client";
import {
  cloneCoarseConfig,
  coarseConfigDefaults,
  sanitizeCoarseConfig
} from "../config/coarseRetrieval";

export const useTaskStore = defineStore("task", () => {
  const targetFile = ref(null);
  const refFiles = ref([]);
  const mode = ref("bert");
  const bertProfile = ref("balanced");
  const bgeStrategy = ref("coarse_then_fine");
  const bodyMode = ref(true);
  const defaultCoarseConfig = ref(cloneCoarseConfig(coarseConfigDefaults));
  const coarseConfig = ref(cloneCoarseConfig(coarseConfigDefaults));

  const loading = ref(false);
  const taskId = ref("");
  const pollStatusMessage = ref("等待任务调度");
  const notice = ref("");
  const results = ref(null);
  const resultSummary = ref(null);
  const costTime = ref(0);

  const windowEstimate = ref(null);
  const windowEstimateLoading = ref(false);
  const windowEstimateError = ref("");

  const previewVisible = ref(false);
  const previewTitle = ref("");
  const previewContent = ref("");
  const previewLoading = ref(false);

  let pollTimer = null;
  let estimateRequestSeq = 0;

  const hasUploads = computed(() => Boolean(targetFile.value) && refFiles.value.length > 0);
  const hasResults = computed(() => Array.isArray(results.value));
  const canSubmit = computed(() => hasUploads.value && !loading.value);
  const canEstimateWindows = computed(
    () => mode.value === "bert" && hasUploads.value && !loading.value
  );

  const activeTopScore = computed(() => {
    if (mode.value === "bert" && resultSummary.value) {
      return Number(resultSummary.value.global_score || 0);
    }

    const first = Array.isArray(results.value) ? results.value[0] : null;
    if (!first) {
      return 0;
    }

    if (mode.value === "bert") {
      return Number(first.sim_bert || first.sim_lsa || 0);
    }
    return Number(first.risk_score || first.sim_hybrid || first.sim_lsa || 0);
  });

  const setNotice = (message) => {
    notice.value = message;
    if (!message) {
      return;
    }

    window.setTimeout(() => {
      if (notice.value === message) {
        notice.value = "";
      }
    }, 4200);
  };

  const resetResultState = () => {
    results.value = null;
    resultSummary.value = null;
    costTime.value = 0;
  };

  const stopPolling = () => {
    if (pollTimer) {
      window.clearInterval(pollTimer);
      pollTimer = null;
    }
  };

  const clearWindowEstimate = () => {
    estimateRequestSeq += 1;
    windowEstimate.value = null;
    windowEstimateError.value = "";
    windowEstimateLoading.value = false;
  };

  const estimateBgeWindowCost = async () => {
    if (!canEstimateWindows.value) {
      clearWindowEstimate();
      return;
    }

    const requestSeq = ++estimateRequestSeq;
    windowEstimateLoading.value = true;
    windowEstimateError.value = "";

    const formData = new FormData();
    formData.append("target_file", targetFile.value);
    refFiles.value.forEach((file) => formData.append("reference_files", file));
    formData.append("body_mode", bodyMode.value);

    try {
      const { data } = await api.post("/api/bge_window_estimate", formData, {
        headers: {
          "Content-Type": "multipart/form-data"
        }
      });

      if (requestSeq !== estimateRequestSeq) {
        return;
      }

      if (data.status !== "success") {
        throw new Error(data.message || "窗口规模估算失败");
      }

      windowEstimate.value = data;
    } catch (error) {
      if (requestSeq !== estimateRequestSeq) {
        return;
      }
      windowEstimate.value = null;
      windowEstimateError.value =
        error.response?.data?.detail ||
        error.message ||
        "窗口规模估算失败，请稍后重试";
    } finally {
      if (requestSeq === estimateRequestSeq) {
        windowEstimateLoading.value = false;
      }
    }
  };

  const markInputsChanged = () => {
    resetResultState();
    clearWindowEstimate();
  };

  const setTargetFile = (file) => {
    targetFile.value = file || null;
    markInputsChanged();
  };

  const setReferenceFiles = (files) => {
    refFiles.value = Array.from(files || []);
    markInputsChanged();
  };

  const clearTarget = () => {
    setTargetFile(null);
  };

  const removeRef = (index) => {
    refFiles.value.splice(index, 1);
    markInputsChanged();
  };

  const setMode = (value) => {
    mode.value = value === "traditional" ? "traditional" : "bert";
    markInputsChanged();
  };

  const setBertProfile = (value) => {
    bertProfile.value = ["strict", "balanced", "recall"].includes(value)
      ? value
      : "balanced";
  };

  const setBgeStrategy = (value) => {
    bgeStrategy.value = value === "full_fine" ? "full_fine" : "coarse_then_fine";
  };

  const setBodyMode = (value) => {
    bodyMode.value = Boolean(value);
    markInputsChanged();
  };

  const applyCoarseConfig = (value) => {
    coarseConfig.value = sanitizeCoarseConfig(value);
    markInputsChanged();
  };

  const updateCoarseConfigField = (key, value) => {
    const nextConfig = sanitizeCoarseConfig({
      ...coarseConfig.value,
      [key]: value
    });
    coarseConfig.value = nextConfig;
    markInputsChanged();
  };

  const resetCoarseConfig = () => {
    applyCoarseConfig(defaultCoarseConfig.value);
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

  const pollTask = async (nextTaskId) => {
    try {
      const { data } = await api.get(`/api/task_status/${nextTaskId}`);
      if (data.status !== "success") {
        throw new Error(data.message || "任务查询失败");
      }

      if (data.task_status === "processing") {
        pollStatusMessage.value = "语义引擎正在分析";
        return;
      }

      if (data.task_status === "pending") {
        pollStatusMessage.value = "任务排队中";
        return;
      }

      if (data.task_status === "completed") {
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
        setNotice("检测完成，结果已更新");
        return;
      }

      if (data.task_status === "failed") {
        stopPolling();
        loading.value = false;
        costTime.value = data.cost_time || 0;
        setNotice(`任务失败：${data.message || "未知错误"}`);
      }
    } catch (error) {
      stopPolling();
      loading.value = false;
      setNotice("任务轮询失败，请稍后重试");
      console.error(error);
    }
  };

  const submitCheck = async () => {
    if (!targetFile.value) {
      setNotice("请先上传待检测文档");
      return null;
    }

    if (!refFiles.value.length) {
      setNotice("请至少上传一份参考文档");
      return null;
    }

    stopPolling();
    loading.value = true;
    resetResultState();
    pollStatusMessage.value = "任务提交中";

    const formData = new FormData();
    formData.append("target_file", targetFile.value);
    refFiles.value.forEach((file) => formData.append("reference_files", file));
    formData.append("mode", mode.value);
    formData.append("bert_profile", bertProfile.value);
    formData.append("body_mode", bodyMode.value);

    if (mode.value === "bert") {
      formData.append("bge_strategy", bgeStrategy.value);
      if (bgeStrategy.value === "coarse_then_fine") {
        formData.append("coarse_config", JSON.stringify(sanitizeCoarseConfig(coarseConfig.value)));
      }
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

      taskId.value = data.task_id;
      pollStatusMessage.value = data.message || "任务已提交，等待调度";
      pollTimer = window.setInterval(() => {
        pollTask(data.task_id);
      }, 2000);
      return data.task_id;
    } catch (error) {
      loading.value = false;
      setNotice(`提交失败：${error.message || "后端服务不可用"}`);
      console.error(error);
      return null;
    }
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
        previewContent.value = data.content || "文档内容为空";
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

  const closePreview = () => {
    previewVisible.value = false;
    previewTitle.value = "";
    previewContent.value = "";
    previewLoading.value = false;
  };

  const requestWindowEstimate = () => {
    estimateBgeWindowCost();
  };

  return {
    targetFile,
    refFiles,
    mode,
    bertProfile,
    bgeStrategy,
    bodyMode,
    defaultCoarseConfig,
    coarseConfig,
    loading,
    taskId,
    pollStatusMessage,
    notice,
    results,
    resultSummary,
    costTime,
    windowEstimate,
    windowEstimateLoading,
    windowEstimateError,
    previewVisible,
    previewTitle,
    previewContent,
    previewLoading,
    hasUploads,
    hasResults,
    canSubmit,
    canEstimateWindows,
    activeTopScore,
    applyCoarseConfig,
    clearTarget,
    closePreview,
    hydrateCoarseConfigDefaults,
    previewFile,
    removeRef,
    requestWindowEstimate,
    resetCoarseConfig,
    setBertProfile,
    setBgeStrategy,
    setBodyMode,
    setMode,
    setNotice,
    setReferenceFiles,
    setTargetFile,
    stopPolling,
    submitCheck,
    updateCoarseConfigField
  };
});
