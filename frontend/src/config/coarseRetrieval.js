export const coarseConfigDefaults = Object.freeze({
  min_candidates: 8,
  max_candidates: 20,
  base_candidate_ratio: 0.15,
  concentrated_min_candidates: 12,
  concentrated_max_candidates: 40,
  concentrated_candidate_ratio: 0.3,
  coarse_threshold: 0.58,
  lexical_threshold: 0.24,
  paragraph_hotspot_threshold: 0.8,
  per_paragraph_top_m: 1,
  paragraph_min_chars: 80,
  paragraph_max_count: 14,
  paragraph_score_top_k: 3,
  lexical_top_terms: 96,
  topic_mean_threshold: 0.8,
  topic_std_threshold: 0.03
});

export const coarseConfigPresets = Object.freeze([
  {
    key: "balanced",
    label: "平衡推荐",
    description: "默认推荐档，兼顾召回、速度与误报控制。",
    bestFor: "大多数日常查重任务，尤其适合参考库规模中等、希望稳定启用两阶段检索时。",
    priority: "平衡召回、速度与误报控制",
    changes: ["候选池规模适中", "阈值设置均衡", "适合作为基线方案"],
    config: coarseConfigDefaults
  },
  {
    key: "conservative",
    label: "低误报",
    description: "收紧候选池和阈值，更适合先压低误报。",
    bestFor: "复核成本较高、参考库噪声较多，或更在意误报而不是漏报的场景。",
    priority: "优先压低误报",
    changes: ["候选池更小", "粗筛阈值更高", "词面和热点要求更严"],
    config: {
      min_candidates: 6,
      max_candidates: 14,
      base_candidate_ratio: 0.1,
      concentrated_min_candidates: 10,
      concentrated_max_candidates: 28,
      concentrated_candidate_ratio: 0.22,
      coarse_threshold: 0.66,
      lexical_threshold: 0.3,
      paragraph_hotspot_threshold: 0.84,
      per_paragraph_top_m: 1,
      paragraph_min_chars: 100,
      paragraph_max_count: 12,
      paragraph_score_top_k: 2,
      lexical_top_terms: 72,
      topic_mean_threshold: 0.82,
      topic_std_threshold: 0.025
    }
  },
  {
    key: "recall",
    label: "高召回",
    description: "扩大候选池并放宽阈值，优先多抓可疑来源。",
    bestFor: "初筛阶段、担心漏掉改写洗稿、希望先尽量把可疑来源抓全时。",
    priority: "优先减少漏检",
    changes: ["候选池更大", "粗筛阈值更低", "每段补召回更多来源"],
    config: {
      min_candidates: 12,
      max_candidates: 28,
      base_candidate_ratio: 0.22,
      concentrated_min_candidates: 18,
      concentrated_max_candidates: 52,
      concentrated_candidate_ratio: 0.4,
      coarse_threshold: 0.5,
      lexical_threshold: 0.18,
      paragraph_hotspot_threshold: 0.74,
      per_paragraph_top_m: 2,
      paragraph_min_chars: 70,
      paragraph_max_count: 18,
      paragraph_score_top_k: 4,
      lexical_top_terms: 120,
      topic_mean_threshold: 0.78,
      topic_std_threshold: 0.04
    }
  },
  {
    key: "same_topic",
    label: "同题文库",
    description: "针对主题高度接近的参考库，尽量避免过早裁掉候选。",
    bestFor: "同课程作业、同方向论文、同项目材料等参考文档主题天然接近的文库。",
    priority: "在同主题参考库中尽量保召回",
    changes: ["同主题扩容更积极", "主题集中判断更宽", "词面窗口更大"],
    config: {
      min_candidates: 14,
      max_candidates: 32,
      base_candidate_ratio: 0.25,
      concentrated_min_candidates: 20,
      concentrated_max_candidates: 60,
      concentrated_candidate_ratio: 0.45,
      coarse_threshold: 0.54,
      lexical_threshold: 0.2,
      paragraph_hotspot_threshold: 0.76,
      per_paragraph_top_m: 2,
      paragraph_min_chars: 70,
      paragraph_max_count: 18,
      paragraph_score_top_k: 4,
      lexical_top_terms: 128,
      topic_mean_threshold: 0.76,
      topic_std_threshold: 0.05
    }
  }
]);

export const coarseConfigPresetMap = Object.freeze(
  Object.fromEntries(
    coarseConfigPresets.map((preset) => [preset.key, preset])
  )
);

export const coarseConfigGroups = [
  {
    key: "candidate_pool",
    title: "候选池规模",
    description: "控制常规候选数和同主题文档库扩容范围。",
    fields: [
      {
        key: "min_candidates",
        label: "常规最小候选数",
        type: "int",
        min: 1,
        step: 1
      },
      {
        key: "max_candidates",
        label: "常规最大候选数",
        type: "int",
        min: 1,
        step: 1
      },
      {
        key: "base_candidate_ratio",
        label: "常规候选比例",
        type: "float",
        min: 0,
        max: 1,
        step: 0.01
      },
      {
        key: "concentrated_min_candidates",
        label: "同主题最小候选数",
        type: "int",
        min: 1,
        step: 1
      },
      {
        key: "concentrated_max_candidates",
        label: "同主题最大候选数",
        type: "int",
        min: 1,
        step: 1
      },
      {
        key: "concentrated_candidate_ratio",
        label: "同主题候选比例",
        type: "float",
        min: 0,
        max: 1,
        step: 0.01
      }
    ]
  },
  {
    key: "candidate_rules",
    title: "候选命中规则",
    description: "控制粗筛主分阈值、词面锚点阈值与段落热点补召回。",
    fields: [
      {
        key: "coarse_threshold",
        label: "粗筛阈值",
        type: "float",
        min: 0,
        max: 1,
        step: 0.01
      },
      {
        key: "lexical_threshold",
        label: "词面锚点阈值",
        type: "float",
        min: 0,
        max: 1,
        step: 0.01
      },
      {
        key: "paragraph_hotspot_threshold",
        label: "段落热点阈值",
        type: "float",
        min: 0,
        max: 1,
        step: 0.01
      },
      {
        key: "per_paragraph_top_m",
        label: "每段补召回数",
        type: "int",
        min: 1,
        step: 1
      }
    ]
  },
  {
    key: "feature_window",
    title: "段落与词面特征",
    description: "控制段落切分粒度、段落热点聚合和词面权重裁剪。",
    fields: [
      {
        key: "paragraph_min_chars",
        label: "段落最小字数",
        type: "int",
        min: 1,
        step: 1
      },
      {
        key: "paragraph_max_count",
        label: "最大段落数",
        type: "int",
        min: 1,
        step: 1
      },
      {
        key: "paragraph_score_top_k",
        label: "段落热点 TopK",
        type: "int",
        min: 1,
        step: 1
      },
      {
        key: "lexical_top_terms",
        label: "词面特征 TopN",
        type: "int",
        min: 1,
        step: 1
      }
    ]
  },
  {
    key: "topic_detection",
    title: "同主题识别",
    description: "判断参考库是否主题过于集中，并据此放宽粗筛裁剪。",
    fields: [
      {
        key: "topic_mean_threshold",
        label: "同主题均值阈值",
        type: "float",
        min: 0,
        max: 1,
        step: 0.01
      },
      {
        key: "topic_std_threshold",
        label: "同主题标准差阈值",
        type: "float",
        min: 0,
        max: 1,
        step: 0.01
      }
    ]
  }
];

export function sanitizeCoarseConfig(rawConfig = {}) {
  const nextConfig = {};

  Object.entries(coarseConfigDefaults).forEach(([key, defaultValue]) => {
    const rawValue = rawConfig[key];
    const numericValue =
      rawValue === "" || rawValue === null || rawValue === undefined
        ? Number.NaN
        : Number(rawValue);

    nextConfig[key] = Number.isFinite(numericValue) ? numericValue : defaultValue;
  });

  return nextConfig;
}

export function cloneCoarseConfig(rawConfig = coarseConfigDefaults) {
  return sanitizeCoarseConfig(rawConfig);
}

export function areCoarseConfigsEqual(leftConfig = {}, rightConfig = {}) {
  const left = sanitizeCoarseConfig(leftConfig);
  const right = sanitizeCoarseConfig(rightConfig);

  return Object.keys(coarseConfigDefaults).every(
    (key) => Number(left[key]) === Number(right[key])
  );
}

export function detectCoarseConfigPreset(
  config = {},
  presetList = coarseConfigPresets
) {
  return (
    presetList.find((preset) =>
      areCoarseConfigsEqual(config, preset.config)
    ) || null
  );
}

export const customCoarseConfigGuide = Object.freeze({
  key: "custom",
  label: "自定义方案",
  description: "当前参数已偏离预设档位，系统会按你手动设置的数值执行粗筛。",
  bestFor: "你已经明确知道要调哪些参数，或正在做参数实验与效果对比时。",
  priority: "以当前手动参数为准",
  changes: ["不会自动回到某个预设", "适合做实验比对", "建议和平衡推荐对照观察"]
});
