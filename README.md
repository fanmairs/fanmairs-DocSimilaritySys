# DocSimilaritySys 文档相似度与查重系统

DocSimilaritySys 是一个面向中文文档的相似度比对与疑似抄袭检测系统。项目同时保留两类检测思路：

- **传统白盒检测**：基于分词、TF-IDF、LSA/SVD、余弦相似度、软语义词向量和滑动窗口，强调可解释、轻量、便于教学展示。
- **深度语义检测**：基于 BGE/SentenceTransformer 做文档级语义、段落级语义和细粒度窗口匹配，并通过粗筛、证据聚合和评分规则降低误报。

系统提供 FastAPI 后端、前端构建产物托管，以及命令行传统检测入口。

## 核心能力

- 支持 TXT、DOCX、PDF 文档读取。
- 支持论文正文清洗，过滤目录、引用、图表、公式、元信息等噪声。
- 支持传统 TF-IDF + LSA 查重。
- 支持同义词和词向量软语义评分。
- 支持 BGE 深度语义细检。
- 支持粗筛后细检，减少大批量参考文档的等待时间。
- 支持全局证据聚合，给出覆盖率、置信度、来源多样性和综合风险。
- 支持异步任务队列，避免多个请求同时抢占 GPU。
- 支持前端构建产物由 FastAPI 统一服务。

## 项目结构

```text
DocSimilaritySys/
├── api.py                         # FastAPI 后端入口
├── api_bge_helpers.py             # API 层 BGE 参数、复核和窗口估算辅助
├── coarse_retrieval.py            # BGE 粗筛召回模块
├── deep_semantic.py               # BGE 深度语义主引擎
├── deep_semantic_evidence.py      # BGE 证据覆盖率和最终评分
├── deep_semantic_profiles.py      # BGE 阈值配置
├── deep_semantic_text.py          # BGE 文本、段落、实体等辅助函数
├── deep_semantic_window_scoring.py# BGE 窗口候选打分
├── detector.py                    # 传统模式滑动窗口检测
├── document_processing.py         # 文档读取和正文清洗
├── frontend_static.py             # FastAPI 前端静态文件服务
├── global_evidence.py             # 多来源全局证据聚合
├── lsa_svd.py                     # 白盒 LSA/SVD
├── main.py                        # 传统查重系统和命令行入口
├── preprocess.py                  # 文本预处理、分词、停用词、同义词
├── similarity.py                  # 余弦相似度
├── soft_semantic.py               # 软语义词向量评分
├── task_store.py                  # SQLite 任务状态存储
├── traditional_scoring.py         # 传统模式分数融合
├── vectorize.py                   # 白盒 TF-IDF
├── data/                          # 示例文档和参考文档
├── dicts/                         # 停用词、同义词、词向量等资源
├── frontend/                      # 前端项目
├── temp_uploads/                  # 上传临时文件目录
└── tests/                         # 单元测试
```

## 每个 Python 文件的职责

| 文件 | 主要职责 |
| --- | --- |
| `api.py` | FastAPI 后端主入口。负责 CORS、任务提交接口、任务状态接口、文档预览接口、BGE 窗口规模估算接口、后台 GPU worker 启动和任务队列消费。 |
| `api_bge_helpers.py` | API 层 BGE 辅助逻辑。负责解析 BGE 策略、解析粗筛配置、执行单篇参考文档的 BGE 细检复核、构造 BERT/BGE 返回结果、估算窗口数量和给出粗筛/全量细检建议。 |
| `coarse_retrieval.py` | 深度语义粗筛模块。负责粗筛配置、目标文档上下文、参考文档上下文、文档语义分、词面锚点分、段落热点分、候选参考文档选择，以及粗筛/细检结果元数据构造。 |
| `deep_semantic.py` | 深度语义主引擎。负责加载 BGE 模型、管理 tokenizer、长文本编码、构建语义窗口、执行滑动窗口细检、计算文档对综合评分，并调用拆分出的 helper 模块完成具体规则。 |
| `deep_semantic_evidence.py` | BGE 证据评分模块。负责目标文本命中区间收集、原始覆盖率、置信度加权覆盖率、匹配置信度、有效覆盖率、连续性特征和最终现实分数计算。 |
| `deep_semantic_profiles.py` | BGE 阈值配置模块。集中存放 `strict`、`balanced`、`recall` 三套阈值、权重和评分门控配置，并负责按名称解析 profile。 |
| `deep_semantic_text.py` | BGE 文本辅助模块。负责文本归一化、段落归一化、span 构造、句子切分、段落提取、区间合并、实体提取、关键词标签、句法骨架和公式说明判断。 |
| `deep_semantic_window_scoring.py` | BGE 窗口候选评分模块。负责 top-k 候选索引选择、统计异常阈值计算，以及窗口候选的实体一致性、标签一致性、模板化、数字/英文密集等反误报规则。 |
| `detector.py` | 传统模式细粒度检测模块。通过滑动窗口切分文本片段，对目标和参考文本的局部片段做 TF-IDF、LSA 和软语义比对，输出疑似片段。 |
| `document_processing.py` | 文档处理模块。负责读取 TXT、DOC/DOCX、PDF，并提供论文正文清洗逻辑，过滤学术噪声、目录、引用、图表、公式和异常段落。 |
| `frontend_static.py` | 前端静态服务模块。负责从 `frontend/dist` 提供构建后的前端页面和资源，并处理前端路由 fallback。 |
| `global_evidence.py` | 全局证据聚合模块。把多个参考来源的 BGE 结果合并，去重重叠区间，计算全局覆盖率、全局置信度、来源多样性、来源支持度和最终全局风险。 |
| `lsa_svd.py` | 白盒 LSA 模块。用 NumPy 对 TF-IDF 矩阵做 SVD 降维，得到低维潜在语义表示。 |
| `main.py` | 传统查重系统主流程。负责装配预处理器、TF-IDF、LSA、软语义、滑动窗口检测器，执行传统查重流程，并提供命令行入口和终端报告输出。 |
| `preprocess.py` | 文本预处理模块。负责加载停用词和同义词，完成文本清洗、jieba 分词、停用词过滤和同义词归一化。 |
| `similarity.py` | 相似度基础模块。提供余弦相似度计算。 |
| `soft_semantic.py` | 软语义评分模块。负责加载同义词和词向量，按 TF-IDF 词项选择重要词，计算词项之间的近义关系，并输出软语义相似度。 |
| `task_store.py` | 任务存储模块。负责 SQLite 表初始化、任务创建、任务状态更新、任务查询、服务重启后的僵尸任务处理和历史任务清理。 |
| `traditional_scoring.py` | 传统模式评分融合模块。负责融合 LSA、TF-IDF、软语义分数，并计算风险分，避免 LSA 单项虚高造成误报。 |
| `vectorize.py` | 白盒 TF-IDF 模块。负责构建词表、计算词频、逆文档频率和 TF-IDF 矩阵。 |
| `tests/test_coarse_retrieval.py` | 粗筛模块单元测试。覆盖候选数量、主题集中度、配置归一化、候选选择规则和粗筛结果元数据。 |
| `tests/test_deep_semantic_helpers.py` | BGE helper 单元测试。覆盖有效覆盖率、技术片段降权、现实分数和连续性特征。 |
| `tests/test_global_evidence.py` | 全局证据聚合单元测试。覆盖来源区间去重、低覆盖证据封顶和无坐标命中的聚合逻辑。 |

## 后端运行

建议在项目根目录执行。

```bash
py -m uvicorn api:app --host 127.0.0.1 --port 8000
```

### PDF 解析后端

默认 PDF 检测使用 `hybrid` 后端：PyMuPDF 读取文本块，pdfplumber 识别表格区域，再由项目规则过滤表格、图表数字、页眉页脚、公式和图注。

```powershell
$env:DOCSIM_PDF_BACKEND="hybrid"
py -m uvicorn api:app --host 127.0.0.1 --port 8000
```

如果要使用 Docling：

```powershell
$env:DOCSIM_PDF_BACKEND="docling"
py -m uvicorn api:app --host 127.0.0.1 --port 8000
```

如果要使用 GROBID 学术结构解析，需要先启动 GROBID 服务。GROBID 后端会优先抽取 TEI XML 中的正文 `body`，跳过表格、图、公式和参考文献；服务不可用时会自动回退到 `hybrid`。

```powershell
docker run --rm --init -p 8070:8070 lfoppiano/grobid:0.8.2
```

另开一个终端启动后端：

```powershell
$env:DOCSIM_PDF_BACKEND="grobid"
$env:GROBID_URL="http://127.0.0.1:8070"
py -m uvicorn api:app --host 127.0.0.1 --port 8000
```

也可以直接运行：

```bash
py api.py
```

启动后访问：

```text
http://127.0.0.1:8000
```

如果前端页面提示 `Frontend build not found`，需要先构建前端：

```bash
cd frontend
npm install
npm run build
cd ..
py -m uvicorn api:app --host 127.0.0.1 --port 8000
```

## 命令行传统检测

默认使用 `data/` 目录中的示例文档：

```bash
py main.py
```

指定目标文档：

```bash
py main.py data/检测文档.txt
```

指定目标文档和参考文档目录：

```bash
py main.py data/检测文档.txt data/
```

## 检测流程说明

### BGE 深度语义模式

1. 前端上传目标文档和参考文档。
2. `api.py` 保存上传文件，并把任务写入 `task_store.py`。
3. 后台 worker 读取任务，使用 `document_processing.py` 提取文本。
4. 如果选择粗筛策略，`coarse_retrieval.py` 先筛出候选参考文档。
5. `deep_semantic.py` 对候选参考文档执行 BGE 细粒度窗口匹配。
6. `deep_semantic_evidence.py` 计算覆盖率、置信度和最终分数。
7. `global_evidence.py` 聚合多个来源，形成全局结论。
8. 结果写回 SQLite，前端轮询任务状态并展示报告。

### 传统白盒模式

1. `main.py` 读取目标文档和参考文档。
2. `document_processing.py` 可选执行正文清洗。
3. `preprocess.py` 分词、去停用词、同义词归一。
4. `vectorize.py` 构建 TF-IDF 矩阵。
5. `lsa_svd.py` 对矩阵做 SVD 降维。
6. `similarity.py` 计算 TF-IDF 和 LSA 相似度。
7. `soft_semantic.py` 补充软语义相似度。
8. `traditional_scoring.py` 融合分数并输出风险。
9. `detector.py` 对高风险参考文档做局部片段定位。

## 测试

运行单元测试：

```bash
$env:PYTHONDONTWRITEBYTECODE = "1"
py -m unittest discover -s tests
```

当前测试覆盖：

- 粗筛候选选择和配置归一化。
- BGE 覆盖率、连续性、现实分数等 helper。
- 全局证据聚合和低证据封顶。

## 常见问题

### 1. `git push` 被拒绝，提示 `fetch first`

说明远程仓库有你本地没有的新提交。先同步远程：

```bash
git pull --rebase origin main
```

处理完冲突后再推送：

```bash
git push origin main
```

### 2. 浏览器提示 `ERR_CONNECTION_REFUSED`

说明后端服务没有启动，或者访问了错误端口。先启动：

```bash
py -m uvicorn api:app --host 127.0.0.1 --port 8000
```

然后访问：

```text
http://127.0.0.1:8000
```

### 3. 前端静态资源 404

说明 `frontend/dist` 不存在或不是最新构建。重新构建前端：

```bash
cd frontend
npm install
npm run build
cd ..
```

## 设计原则

- 将粗筛、细检、证据聚合、文本处理、任务存储拆成独立模块，便于查阅和维护。
- 保留传统白盒算法，方便解释 TF-IDF、LSA、余弦相似度和滑动窗口的检测逻辑。
- 深度语义模式优先使用粗筛减少计算量，再用 BGE 细检保证结果质量。
- 分数不只看语义相似度，还结合覆盖率、置信度、连续性和来源多样性，降低同主题误报。
