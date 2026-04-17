# DocSimilaritySys 文档相似度与查重系统

DocSimilaritySys 是一个面向中文文档的相似度比对与疑似抄袭检测系统。项目保留两条检测路线：

- 传统白盒检测：分词、TF-IDF、LSA/SVD、余弦相似度、软语义词向量评分和局部窗口定位，适合轻量、稳定、可解释的查重。
- BGE 深度语义检测：使用 SentenceTransformer/BGE 做文档级、段落级和窗口级语义匹配，再通过粗筛、证据聚合和评分门控降低同主题误报。

后端使用 FastAPI，前端构建产物由 FastAPI 托管。根目录只保留应用入口和运行支撑代码，核心能力按流程拆到包目录中。

## 项目结构

```text
DocSimilaritySys/
├── api.py                         # FastAPI 后端入口
├── api_bge_helpers.py             # API 层 BGE 参数、复核和窗口估算辅助
├── frontend_static.py             # 前端静态文件托管
├── main.py                        # 传统检测 CLI 入口
├── task_store.py                  # SQLite 任务状态存储
├── document_readers/              # 文档读取模块
│   ├── factory.py                 # 按扩展名分发 TXT/DOCX/PDF 读取器
│   ├── txt/reader.py              # TXT 读取
│   ├── docx/reader.py             # DOCX 读取
│   └── pdf/
│       ├── reader.py              # PDF hybrid/pymupdf/docling/grobid 统一入口
│       ├── pymupdf_backend.py     # PyMuPDF 轻量读取
│       ├── hybrid.py              # hybrid 后端导出
│       ├── docling_backend.py     # Docling 后端导出
│       └── grobid_backend.py      # GROBID TEI 正文解析
├── text_processing/               # 文本处理模块
│   ├── cleaners/                  # 学术噪声、数字表格噪声清洗
│   ├── normalizers/               # 通用/PDF 文本归一化
│   ├── segmenters/                # 句子、段落切分
│   └── tokenizers/                # jieba 分词、停用词、同义词归一
├── engines/                       # 检测引擎模块
│   ├── factory.py                 # 检测引擎工厂
│   ├── traditional/               # 传统白盒引擎
│   │   ├── system.py              # 传统检测主流程
│   │   ├── tfidf_backend.py       # TF-IDF
│   │   ├── lsa_backend.py         # LSA/SVD
│   │   ├── similarity.py          # 余弦相似度
│   │   ├── soft_semantic.py       # 软语义评分
│   │   ├── scoring.py             # 分数融合与风险分
│   │   └── window_detector.py     # 局部疑似片段定位
│   └── semantic/                  # BGE 深度语义引擎
│       ├── bge_backend.py         # BGE 主引擎
│       ├── coarse_retrieval.py    # 粗筛候选召回
│       ├── evidence.py            # 覆盖率、置信度、连续性证据
│       ├── global_evidence.py     # 多来源全局证据聚合
│       ├── profiles.py            # strict/balanced/recall 阈值配置
│       ├── text.py                # 语义文本辅助函数
│       └── window_scoring.py      # 窗口候选评分规则
├── frontend/                      # 前端项目
├── dicts/                         # 停用词、同义词、词向量资源
└── tests/                         # 单元测试
```

## PDF 读取后端

默认 PDF 检测使用 `hybrid`，也就是 PyMuPDF 读取文本块，pdfplumber 识别表格区域，再由项目规则过滤表格、图表数字、页眉页脚、公式和图注。这是当前最稳的日常方案。

```powershell
$env:DOCSIM_PDF_BACKEND="hybrid"
py -m uvicorn api:app --host 127.0.0.1 --port 8000
```

可选后端：

```powershell
$env:DOCSIM_PDF_BACKEND="pymupdf"  # 更轻量，只走 PyMuPDF 与基础过滤
$env:DOCSIM_PDF_BACKEND="docling"  # 高级文档解析，失败会回退到 hybrid
$env:DOCSIM_PDF_BACKEND="grobid"   # 学术结构解析，需要 GROBID 服务，失败会回退到 hybrid
```

GROBID 需要先启动服务：

```powershell
docker run --rm --init -p 8070:8070 lfoppiano/grobid:0.8.2
$env:DOCSIM_PDF_BACKEND="grobid"
$env:GROBID_URL="http://127.0.0.1:8070"
py -m uvicorn api:app --host 127.0.0.1 --port 8000
```

Docling 默认关闭 OCR 和表格结构识别，以降低本地硬件压力：

```powershell
$env:DOCSIM_PDF_BACKEND="docling"
$env:DOCSIM_DOCLING_OCR="0"
$env:DOCSIM_DOCLING_TABLE_STRUCTURE="0"
py -m uvicorn api:app --host 127.0.0.1 --port 8000
```

## 运行

后端：

```powershell
py -m uvicorn api:app --host 127.0.0.1 --port 8000
```

如果需要重新构建前端：

```powershell
cd frontend
npm install
npm run build
cd ..
py -m uvicorn api:app --host 127.0.0.1 --port 8000
```

传统 CLI：

```powershell
py main.py
py main.py data/target.txt data/
```

## 检测流程

BGE 深度语义模式：

1. `api.py` 接收上传文件，写入 `task_store.py` 任务表。
2. 后台 worker 通过 `document_readers.factory.read_document_by_type` 提取文本。
3. 如果选择粗筛，`engines.semantic.coarse_retrieval` 先筛出候选参考文档。
4. `engines.semantic.bge_backend` 对候选文档执行 BGE 窗口级细检。
5. `engines.semantic.evidence` 计算覆盖率、置信度、连续性和现实分数。
6. `engines.semantic.global_evidence` 聚合多个来源，形成全局结论。
7. 结果写回 SQLite，前端轮询任务状态并展示报告。

传统白盒模式：

1. `engines.traditional.system` 读取目标和参考文档。
2. `text_processing.cleaners.academic` 可选执行正文清洗。
3. `text_processing.tokenizers.TextPreprocessor` 分词、去停用词、同义词归一。
4. `engines.traditional.tfidf_backend` 构建 TF-IDF 矩阵。
5. `engines.traditional.lsa_backend` 对矩阵做 SVD 降维。
6. `engines.traditional.similarity` 计算 TF-IDF 和 LSA 相似度。
7. `engines.traditional.soft_semantic` 补充软语义相似度。
8. `engines.traditional.scoring` 融合分数并输出风险。
9. `engines.traditional.window_detector` 对高风险参考文档做局部片段定位。

## 测试

```powershell
$env:PYTHONDONTWRITEBYTECODE = "1"
py -m unittest discover -s tests
```

当前测试覆盖：

- PDF 后端分发和数字表格噪声过滤。
- 文档读取工厂与 TXT/DOCX/PDF 分支。
- 文本清洗、归一化、句子和段落切分。
- 传统与语义引擎工厂。
- BGE 粗筛、窗口证据、全局证据聚合。

## 设计原则

- 根目录保持清爽，只放应用入口和运行支撑。
- 按流程分包：文档读取、文本处理、检测引擎分别维护。
- 默认使用 `hybrid` PDF 读取，保证本地稳定性；Docling 和 GROBID 作为高级可选后端。
- 传统算法保留白盒可解释性，BGE 模式负责更强的语义匹配。
- 分数不只看语义相似度，还结合覆盖率、置信度、连续性和来源多样性。
