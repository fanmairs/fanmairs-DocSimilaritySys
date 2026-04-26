import { createRouter, createWebHistory } from "vue-router";
import UploadView from "../views/UploadView.vue";
import ConfigureView from "../views/ConfigureView.vue";
import ResultView from "../views/ResultView.vue";

export const routes = [
  {
    path: "/",
    redirect: "/upload"
  },
  {
    path: "/upload",
    name: "upload",
    component: UploadView,
    meta: {
      step: "01",
      label: "文档上传"
    }
  },
  {
    path: "/detect",
    name: "detect",
    component: ConfigureView,
    meta: {
      step: "02",
      label: "检测配置"
    }
  },
  {
    path: "/results",
    name: "results",
    component: ResultView,
    meta: {
      step: "03",
      label: "结果复核"
    }
  }
];

export const router = createRouter({
  history: createWebHistory(),
  routes,
  scrollBehavior() {
    return { top: 0 };
  }
});
