<script setup>
import { computed, onMounted } from "vue";
import { RouterLink, RouterView, useRoute } from "vue-router";
import {
  Activity,
  Database,
  FileStack,
  FileText,
  Gauge,
  LayoutDashboard
} from "lucide-vue-next";
import PreviewModal from "./components/PreviewModal.vue";
import { routes } from "./router";
import { useTaskStore } from "./stores/task";

const task = useTaskStore();
const route = useRoute();

const navItems = routes
  .filter((item) => item.name)
  .map((item) => ({
    name: item.name,
    path: item.path,
    step: item.meta.step,
    label: item.meta.label
  }));

const sessionStats = computed(() => [
  {
    label: "目标文档",
    value: task.targetFile ? "1" : "0",
    icon: FileText
  },
  {
    label: "参考文档",
    value: String(task.refFiles.length),
    icon: FileStack
  },
  {
    label: "检测模式",
    value: task.mode === "bert" ? "BGE" : "传统",
    icon: Gauge
  },
  {
    label: "任务状态",
    value: task.loading ? "运行中" : task.hasResults ? "已完成" : "待提交",
    icon: Activity
  }
]);

const activeRouteLabel = computed(
  () => navItems.find((item) => item.name === route.name)?.label || "工作台"
);

onMounted(() => {
  task.hydrateCoarseConfigDefaults();
});
</script>

<template>
  <main class="workflow-shell">
    <aside class="workflow-sidebar">
      <RouterLink class="brand-mark" to="/upload">
        <LayoutDashboard :size="24" />
        <span>
          <strong>DocSimilaritySys</strong>
          <small>文档查重分析台</small>
        </span>
      </RouterLink>

      <nav class="workflow-nav" aria-label="检测流程">
        <RouterLink
          v-for="item in navItems"
          :key="item.name"
          class="workflow-nav__item"
          :class="{ 'workflow-nav__item--active': route.name === item.name }"
          :to="item.path"
        >
          <span>{{ item.step }}</span>
          <strong>{{ item.label }}</strong>
        </RouterLink>
      </nav>

      <section class="sidebar-status">
        <Database :size="18" />
        <div>
          <span>当前任务</span>
          <strong>{{ task.taskId || "尚未提交" }}</strong>
        </div>
      </section>
    </aside>

    <section class="workflow-main">
      <header class="workflow-topbar">
        <div>
          <p>当前视图</p>
          <h2>{{ activeRouteLabel }}</h2>
        </div>

        <div class="session-strip">
          <article v-for="stat in sessionStats" :key="stat.label" class="session-stat">
            <component :is="stat.icon" :size="17" />
            <span>{{ stat.label }}</span>
            <strong>{{ stat.value }}</strong>
          </article>
        </div>
      </header>

      <p v-if="task.notice" class="toast-notice">{{ task.notice }}</p>

      <RouterView v-slot="{ Component }">
        <Transition name="page-fade" mode="out-in">
          <component :is="Component" />
        </Transition>
      </RouterView>
    </section>

    <PreviewModal
      :visible="task.previewVisible"
      :title="task.previewTitle"
      :content="task.previewContent"
      :loading="task.previewLoading"
      @close="task.closePreview"
    />
  </main>
</template>
