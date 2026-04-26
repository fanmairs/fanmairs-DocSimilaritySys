<script setup>
import { computed, ref } from "vue";
import { useRouter } from "vue-router";
import {
  ArrowRight,
  Eye,
  FileText,
  FolderOpen,
  Library,
  Trash2,
  UploadCloud
} from "lucide-vue-next";
import { useTaskStore } from "../stores/task";

const task = useTaskStore();
const router = useRouter();
const targetInputRef = ref(null);
const refInputRef = ref(null);
const targetDragActive = ref(false);
const refDragActive = ref(false);
const acceptTypes = ".txt,.doc,.docx,.pdf";

const referenceSize = computed(() =>
  task.refFiles.reduce((total, file) => total + Number(file.size || 0), 0)
);

const formatFileSize = (bytes = 0) => {
  if (!bytes) {
    return "0 KB";
  }

  const units = ["B", "KB", "MB", "GB"];
  let value = Number(bytes);
  let unitIndex = 0;
  while (value >= 1024 && unitIndex < units.length - 1) {
    value /= 1024;
    unitIndex += 1;
  }
  return `${value.toFixed(unitIndex === 0 ? 0 : 1)} ${units[unitIndex]}`;
};

const openTargetPicker = () => {
  targetInputRef.value?.click();
};

const openRefPicker = () => {
  refInputRef.value?.click();
};

const onTargetChange = (event) => {
  task.setTargetFile(event.target.files?.[0] || null);
  event.target.value = "";
};

const onRefChange = (event) => {
  task.setReferenceFiles(event.target.files || []);
  event.target.value = "";
};

const onTargetDrop = (event) => {
  targetDragActive.value = false;
  task.setTargetFile(event.dataTransfer?.files?.[0] || null);
};

const onRefDrop = (event) => {
  refDragActive.value = false;
  task.setReferenceFiles(event.dataTransfer?.files || []);
};

const continueToConfig = () => {
  if (!task.hasUploads) {
    task.setNotice("请先上传待检测文档和至少一份参考文档");
    return;
  }
  router.push("/detect");
};
</script>

<template>
  <section class="work-view upload-view">
    <header class="view-header">
      <div>
        <p class="view-kicker">Step 01</p>
        <h1>文档上传</h1>
      </div>
      <button class="command-button command-button--primary" type="button" @click="continueToConfig">
        <ArrowRight :size="18" />
        进入检测配置
      </button>
    </header>

    <div class="upload-matrix">
      <section
        class="drop-lane drop-lane--target"
        :class="{ 'drop-lane--active': targetDragActive }"
        @dragenter.prevent="targetDragActive = true"
        @dragover.prevent
        @dragleave.prevent="targetDragActive = false"
        @drop.prevent="onTargetDrop"
      >
        <input
          ref="targetInputRef"
          class="sr-only"
          type="file"
          :accept="acceptTypes"
          @change="onTargetChange"
        />

        <div class="lane-icon">
          <FileText :size="24" />
        </div>
        <div class="lane-copy">
          <p class="lane-label">待检测文档</p>
          <h2>{{ task.targetFile?.name || "选择目标文档" }}</h2>
          <span v-if="targetDragActive">松开后设为待检测文档</span>
          <span v-else>{{ task.targetFile ? formatFileSize(task.targetFile.size) : "支持 TXT / DOCX / PDF" }}</span>
        </div>

        <div class="lane-actions">
          <button class="icon-command" type="button" title="选择目标文档" @click="openTargetPicker">
            <UploadCloud :size="18" />
          </button>
          <button
            class="icon-command"
            type="button"
            title="预览目标文档"
            :disabled="!task.targetFile"
            @click="task.previewFile(task.targetFile)"
          >
            <Eye :size="18" />
          </button>
          <button
            class="icon-command icon-command--danger"
            type="button"
            title="移除目标文档"
            :disabled="!task.targetFile"
            @click="task.clearTarget"
          >
            <Trash2 :size="18" />
          </button>
        </div>
      </section>

      <section
        class="drop-lane"
        :class="{ 'drop-lane--active': refDragActive }"
        @dragenter.prevent="refDragActive = true"
        @dragover.prevent
        @dragleave.prevent="refDragActive = false"
        @drop.prevent="onRefDrop"
      >
        <input
          ref="refInputRef"
          class="sr-only"
          type="file"
          multiple
          :accept="acceptTypes"
          @change="onRefChange"
        />

        <div class="lane-icon lane-icon--warm">
          <Library :size="24" />
        </div>
        <div class="lane-copy">
          <p class="lane-label">参考文档库</p>
          <h2>{{ task.refFiles.length ? `${task.refFiles.length} 份参考文档` : "选择参考文档" }}</h2>
          <span v-if="refDragActive">松开后替换参考文档库</span>
          <span v-else>{{ task.refFiles.length ? formatFileSize(referenceSize) : "可一次选择多份文件" }}</span>
        </div>

        <div class="lane-actions">
          <button class="icon-command" type="button" title="选择并替换参考文档" @click="openRefPicker">
            <FolderOpen :size="18" />
          </button>
        </div>
      </section>
    </div>

    <section class="file-ledger" v-if="task.targetFile || task.refFiles.length">
      <article v-if="task.targetFile" class="ledger-row ledger-row--target">
        <div>
          <span>目标</span>
          <strong>{{ task.targetFile.name }}</strong>
          <small>{{ formatFileSize(task.targetFile.size) }}</small>
        </div>
        <button class="icon-command" type="button" title="预览" @click="task.previewFile(task.targetFile)">
          <Eye :size="17" />
        </button>
      </article>

      <article v-for="(file, index) in task.refFiles" :key="`${file.name}-${index}`" class="ledger-row">
        <div>
          <span>参考 {{ index + 1 }}</span>
          <strong>{{ file.name }}</strong>
          <small>{{ formatFileSize(file.size) }}</small>
        </div>
        <div class="ledger-actions">
          <button class="icon-command" type="button" title="预览" @click="task.previewFile(file)">
            <Eye :size="17" />
          </button>
          <button class="icon-command icon-command--danger" type="button" title="移除" @click="task.removeRef(index)">
            <Trash2 :size="17" />
          </button>
        </div>
      </article>
    </section>
  </section>
</template>
