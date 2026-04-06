<template>
  <section class="relative mb-5 overflow-hidden rounded-[2rem] border border-white/70 bg-night-900 px-6 py-8 text-white shadow-soft sm:px-10 sm:py-9">
    <div class="pointer-events-none absolute -left-14 top-0 h-40 w-40 rounded-full bg-amber-500/40 blur-3xl"></div>
    <div class="pointer-events-none absolute -right-16 bottom-0 h-48 w-48 rounded-full bg-mint-500/30 blur-3xl animate-ambient-float"></div>
    <div class="pointer-events-none absolute inset-0 opacity-45 [background:linear-gradient(120deg,rgba(255,255,255,0.12),rgba(255,255,255,0)_45%)]"></div>

    <div class="relative grid gap-7 lg:grid-cols-[minmax(0,1.25fr)_minmax(0,0.75fr)] lg:items-end">
      <div>
        <p class="mb-3 inline-flex rounded-full border border-white/25 bg-white/10 px-3 py-1 text-[11px] font-semibold uppercase tracking-[0.2em] text-white/90">
          Document Similarity Studio
        </p>
        <h1 class="font-display text-3xl font-extrabold leading-tight text-white sm:text-[2.55rem]">
          智能文档查重系统
        </h1>
        <p class="mt-4 max-w-2xl text-sm leading-7 text-white/80 sm:text-base">
          面向论文、作业与项目报告的一站式相似性分析工作台。上传文档后，系统会自动比对、排序并给出可复核的风险片段。
        </p>
        <div class="mt-5 flex flex-wrap gap-2">
          <span class="rounded-full border border-white/25 bg-white/10 px-3 py-1 text-xs font-semibold text-white/85">语义引擎 + 传统引擎</span>
          <span class="rounded-full border border-white/25 bg-white/10 px-3 py-1 text-xs font-semibold text-white/85">多文档并行比对</span>
          <span class="rounded-full border border-white/25 bg-white/10 px-3 py-1 text-xs font-semibold text-white/85">片段级风险定位</span>
        </div>
      </div>

      <aside class="glass-panel border-white/20 bg-white/10 p-4 sm:p-5">
        <p class="text-xs font-semibold uppercase tracking-[0.2em] text-white/70">工作流程</p>
        <ol class="mt-3 space-y-3">
          <li class="flex items-start gap-3 text-sm text-white/85">
            <span class="mt-0.5 inline-flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-white/15 font-display text-xs font-bold text-white">1</span>
            选择待检测文档与参考文档库
          </li>
          <li class="flex items-start gap-3 text-sm text-white/85">
            <span class="mt-0.5 inline-flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-white/15 font-display text-xs font-bold text-white">2</span>
            设置引擎与阈值档位并提交任务
          </li>
          <li class="flex items-start gap-3 text-sm text-white/85">
            <span class="mt-0.5 inline-flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-white/15 font-display text-xs font-bold text-white">3</span>
            查看排序结果与高风险片段详情
          </li>
        </ol>
      </aside>
    </div>
  </section>
</template>
