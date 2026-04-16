import os
import shutil
import uuid
import threading
import queue
import time
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from api_bge_helpers import (
    BGE_STRATEGY_COARSE,
    build_basic_bert_result,
    estimate_text_window_count,
    parse_coarse_config_payload,
    resolve_bge_strategy,
    run_bert_fine_verification,
    window_recommendation,
    window_scale_level,
)
from frontend_static import serve_frontend_path
from task_store import create_task, get_task, init_db, update_task
from engines.semantic.coarse_retrieval import CoarseRetriever, CoarseRetrievalConfig
from engines.semantic.global_evidence import GlobalEvidenceAggregator

# 导入我们项目核心类
from engines.semantic.bge_backend import DeepSemanticEngine
from engines.traditional.system import PlagiarismDetectorSystem

app = FastAPI(title="智能文档查重系统 (异步队列并发版)")

# 配置 CORS 允许跨域请求
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

TEMP_DIR = "temp_uploads"
os.makedirs(TEMP_DIR, exist_ok=True)

# ==========================================
# 1. 数据库初始化 (模拟企业级 MySQL)
# ==========================================
# 用于持久化存储任务的进度和结果

init_db()


# ==========================================
# 2. 全局任务队列与后台消费者 (模拟企业级 Celery + Redis)
# ==========================================
# 所有前端请求都会被塞进这个队列，而不是直接去抢占 GPU
task_queue = queue.Queue()

# 全局模型实例，由专属后台线程加载，避免多线程抢占显存
bert_engine = None
traditional_system = None
coarse_retriever = None
global_evidence_aggregator = None


def gpu_worker():
    """
    这是一个独立运行的后台守护线程。
    它的使命是：永远在后台排队取任务，确保同一时刻永远只有一个任务在占用 GPU。
    这就从根本上解决了 100 个人同时点击查重导致 3060 显存瞬间爆炸的问题！
    """
    global bert_engine, traditional_system, coarse_retriever, global_evidence_aggregator
    print(">>> [GPU 队列守护进程] 正在启动并独占加载深度学习模型...")
    bert_engine = DeepSemanticEngine()
    traditional_system = PlagiarismDetectorSystem(
        stopwords_path='dicts/stopwords.txt',
        lsa_components=3,
        synonyms_path='dicts/synonyms.txt',
        semantic_embeddings_path='dicts/embeddings/fasttext_zh.vec',
        semantic_threshold=0.55,
        semantic_weight=0.35
    )
    coarse_retriever = CoarseRetriever(bert_engine, traditional_system.preprocessor)
    global_evidence_aggregator = GlobalEvidenceAggregator(bert_engine)
    print(">>> [GPU 队列守护进程] 模型加载完毕，开始监听并发查重任务...")

    while True:
        # 如果队列为空，线程会在这里阻塞休眠，不消耗 CPU
        task = task_queue.get()
        task_id = task['id']
        target_path = task['target_path']
        ref_paths = task['ref_paths']
        mode = task['mode']
        body_mode = task['body_mode']
        bert_profile = task.get('bert_profile', 'balanced')
        bge_strategy = task.get('bge_strategy', BGE_STRATEGY_COARSE)
        coarse_config = task.get('coarse_config')
        session_dir = task['session_dir']

        print(f">>> [GPU Worker] 正在处理排队任务: {task_id}")
        update_task(task_id, "processing")

        # 记录开始时间，用于计算耗时
        start_time = time.time()

        try:
            results = []
            result_summary = None
            if mode == "bert":
                # 深度语义引擎检测逻辑
                target_text = traditional_system.read_document(target_path)
                if body_mode:
                    target_text = traditional_system.clean_academic_noise(target_text)

                reference_payloads = []
                reference_text_map = {}
                for ref_path in ref_paths:
                    ref_text = traditional_system.read_document(ref_path)
                    if body_mode:
                        ref_text = traditional_system.clean_academic_noise(ref_text)
                    reference_payloads.append({
                        "path": ref_path,
                        "text": ref_text,
                    })
                    reference_text_map[ref_path] = ref_text

                verified_results = []
                coarse_only_results = []
                candidate_count = len(reference_payloads)

                print(
                    ">>> [BGE][Strategy] "
                    f"strategy={bge_strategy} references={len(reference_payloads)}"
                )

                if bge_strategy == BGE_STRATEGY_COARSE:
                    task_coarse_retriever = coarse_retriever.with_config(coarse_config)
                    target_context = task_coarse_retriever.build_target_context(target_text)
                    reference_contexts = task_coarse_retriever.build_reference_contexts(reference_payloads)
                    ranked_refs, selection_meta = task_coarse_retriever.rank_references(
                        target_context,
                        reference_contexts,
                    )
                    ranked_ref_map = {item["path"]: item for item in ranked_refs}
                    candidate_ref_paths = [
                        item["path"]
                        for item in ranked_refs
                        if item.get("is_candidate", False)
                    ]
                    coarse_only_results = [
                        task_coarse_retriever.build_coarse_only_result(item, bert_profile)
                        for item in ranked_refs
                        if not item.get("is_candidate", False)
                    ]
                    for item in coarse_only_results:
                        item["retrieval_strategy"] = bge_strategy

                    candidate_count = int(selection_meta.get("candidate_count", len(candidate_ref_paths)))
                    print(
                        ">>> [BGE][Coarse] "
                        f"references={len(reference_payloads)} "
                        f"candidates={selection_meta['candidate_count']} "
                        f"candidate_limit={selection_meta['candidate_limit']} "
                        f"topic_concentrated={selection_meta['topic_concentrated']} "
                        f"theme_mean={selection_meta['theme_mean']:.4f} "
                        f"theme_std={selection_meta['theme_std']:.4f}"
                    )

                    for ref_path in candidate_ref_paths:
                        ref_text = reference_text_map[ref_path]
                        plagiarized_parts, score_breakdown = run_bert_fine_verification(
                            bert_engine,
                            ref_path,
                            target_text,
                            ref_text,
                            bert_profile,
                        )
                        verified_result = build_basic_bert_result(
                            ref_path,
                            bert_profile,
                            score_breakdown,
                            plagiarized_parts,
                        )
                        verified_result.update(
                            task_coarse_retriever.build_verified_result(
                                ranked_ref_map[ref_path],
                                bert_profile,
                                score_breakdown,
                                plagiarized_parts,
                            )
                        )
                        verified_result["retrieval_strategy"] = bge_strategy
                        results.append(verified_result)
                        verified_results.append(verified_result)
                else:
                    candidate_ref_paths = ref_paths
                    candidate_count = len(candidate_ref_paths)
                    print(
                        ">>> [BGE][FullFine] "
                        f"references={len(reference_payloads)} "
                        "candidates=all"
                    )

                    for index, ref_path in enumerate(candidate_ref_paths, start=1):
                        ref_text = reference_text_map[ref_path]
                        plagiarized_parts, score_breakdown = run_bert_fine_verification(
                            bert_engine,
                            ref_path,
                            target_text,
                            ref_text,
                            bert_profile,
                        )
                        verified_result = build_basic_bert_result(
                            ref_path,
                            bert_profile,
                            score_breakdown,
                            plagiarized_parts,
                        )
                        verified_result.update(
                            {
                                "sim_bert_candidate_rank": index,
                                "sim_bert_coarse_rank": None,
                                "retrieval_strategy": bge_strategy,
                                "retrieval_reason": "full_fine",
                                "retrieval_candidate_pool_size": len(candidate_ref_paths),
                                "retrieval_reference_count": len(reference_payloads),
                                "retrieval_theme_mean": 0.0,
                                "retrieval_theme_std": 0.0,
                                "retrieval_topic_concentrated": False,
                            }
                        )
                        results.append(verified_result)
                        verified_results.append(verified_result)

                result_summary = global_evidence_aggregator.aggregate(
                    target_text,
                    verified_results,
                    bert_profile=bert_profile,
                    reference_count=len(reference_payloads),
                    candidate_count=candidate_count,
                    retrieval_strategy=bge_strategy,
                )
                print(
                    ">>> [BGE][Global] "
                    f"score={result_summary['global_score']:.4f} "
                    f"coverage={result_summary['global_coverage_effective']:.4f} "
                    f"confidence={result_summary['global_confidence']:.4f} "
                    f"source_diversity={result_summary['global_source_diversity']:.4f} "
                    f"verified_sources={result_summary['global_verified_source_count']}"
                )

                results.extend(coarse_only_results)
                results.sort(key=lambda x: x['sim_bert'], reverse=True)

            else:
                # 传统字面引擎检测逻辑
                raw_results = traditional_system.check_similarity(target_path, ref_paths, body_mode=body_mode)
                for r in raw_results:
                    results.append({
                        "file": os.path.basename(r['file']).replace("ref_", ""),
                        "sim_lsa": float(r['sim_lsa']),
                                "sim_tfidf": float(r['sim_tfidf']),
                                "sim_soft": float(r.get('sim_soft', 0.0)),
                                "sim_hybrid": float(r.get('sim_hybrid', r['sim_lsa'])),
                                "risk_score": float(r.get('risk_score', r.get('sim_hybrid', r['sim_lsa']))),
                                "plagiarized_parts": r.get('plagiarized_parts', [])
                            })

            # 计算耗时并写入数据库
            cost_time = time.time() - start_time
            result_payload = {
                "items": results,
                "summary": result_summary,
            }
            update_task(task_id, "completed", result=result_payload, cost_time=cost_time)
            print(f">>> [GPU Worker] 任务圆满完成: {task_id} (耗时: {cost_time:.2f}秒)")

        except Exception as e:
            import traceback
            traceback.print_exc()
            cost_time = time.time() - start_time
            update_task(task_id, "failed", message=str(e), cost_time=cost_time)
            print(f">>> [GPU Worker] 任务计算崩溃: {task_id} (耗时: {cost_time:.2f}秒)")

        finally:
            # 清理当前任务产生的临时文件
            shutil.rmtree(session_dir, ignore_errors=True)
            # 通知队列当前任务处理完毕，可以拉取下一个任务了
            task_queue.task_done()

# 随系统启动后台消费者线程
threading.Thread(target=gpu_worker, daemon=True).start()

# ==========================================
# 3. FastAPI 路由接口 (非阻塞式)
# ==========================================
@app.post("/api/submit_task")
async def submit_task(
    target_file: UploadFile = File(...),
    reference_files: List[UploadFile] = File(...),
    mode: str = Form("bert"),
    body_mode: bool = Form(False),
    bert_profile: str = Form("balanced"),
    bge_strategy: str = Form(BGE_STRATEGY_COARSE),
    coarse_config: str = Form("")
):
    """
    【非阻塞提交接口】
    无论外面有多少个人同时点击“查重”，这个接口只负责把文件保存下来，
    生成一个任务号，塞进队列，然后立刻返回给前端。
    绝对不会卡死！
    """
    task_id = str(uuid.uuid4())
    session_dir = os.path.join(TEMP_DIR, task_id)
    os.makedirs(session_dir, exist_ok=True)

    # 1. 极速保存文件
    target_path = os.path.join(session_dir, f"target_{target_file.filename}")
    with open(target_path, "wb") as f: shutil.copyfileobj(target_file.file, f)

    ref_paths = []
    for ref in reference_files:
        ref_path = os.path.join(session_dir, f"ref_{ref.filename}")
        with open(ref_path, "wb") as f: shutil.copyfileobj(ref.file, f)
        ref_paths.append(ref_path)

    # 2. Register the queued task
    create_task(task_id)

    # 3. 塞入后台处理队列
    safe_profile = (bert_profile or "balanced").strip().lower()
    if safe_profile not in {"strict", "balanced", "recall"}:
        safe_profile = "balanced"
    safe_bge_strategy = resolve_bge_strategy(bge_strategy)
    safe_coarse_config = (
        parse_coarse_config_payload(coarse_config)
        if safe_bge_strategy == BGE_STRATEGY_COARSE
        else None
    )

    task_queue.put({
        'id': task_id,
        'target_path': target_path,
        'ref_paths': ref_paths,
        'mode': mode,
        'body_mode': body_mode,
        'bert_profile': safe_profile,
        'bge_strategy': safe_bge_strategy,
        'coarse_config': safe_coarse_config,
        'session_dir': session_dir
    })

    queue_length = task_queue.qsize()
    return {
        "status": "success",
        "task_id": task_id,
        "queue_length": queue_length,
        "message": f"任务已提交！前面还有 {queue_length - 1} 个人在排队..." if queue_length > 1 else "任务已提交！即将开始计算..."
    }

@app.get("/api/coarse_config_defaults")
async def coarse_config_defaults():
    return {
        "status": "success",
        "defaults": CoarseRetrievalConfig().normalized().to_dict(),
    }


@app.post("/api/bge_window_estimate")
async def bge_window_estimate(
    target_file: UploadFile = File(...),
    reference_files: List[UploadFile] = File(...),
    body_mode: bool = Form(False),
):
    if bert_engine is None or traditional_system is None:
        raise HTTPException(status_code=503, detail="BGE 模型仍在加载，请稍后再估算窗口规模")

    estimate_id = uuid.uuid4().hex[:10]
    estimate_dir = os.path.join(TEMP_DIR, f"estimate_{estimate_id}")
    os.makedirs(estimate_dir, exist_ok=True)

    try:
        target_path = os.path.join(estimate_dir, f"target_{target_file.filename}")
        with open(target_path, "wb") as f:
            shutil.copyfileobj(target_file.file, f)

        ref_paths = []
        for ref in reference_files:
            ref_path = os.path.join(estimate_dir, f"ref_{ref.filename}")
            with open(ref_path, "wb") as f:
                shutil.copyfileobj(ref.file, f)
            ref_paths.append(ref_path)

        target_text = traditional_system.read_document(target_path)
        if body_mode:
            target_text = traditional_system.clean_academic_noise(target_text)
        target_window_count = estimate_text_window_count(bert_engine, target_text)

        reference_summaries = []
        total_reference_windows = 0
        for ref_path in ref_paths:
            ref_text = traditional_system.read_document(ref_path)
            if body_mode:
                ref_text = traditional_system.clean_academic_noise(ref_text)
            window_count = estimate_text_window_count(bert_engine, ref_text)
            total_reference_windows += window_count
            reference_summaries.append(
                {
                    "file": os.path.basename(ref_path).replace("ref_", ""),
                    "window_count": int(window_count),
                }
            )

        full_pair_count = int(target_window_count * total_reference_windows)
        recommendation = window_recommendation(full_pair_count, len(ref_paths))
        scale_level = window_scale_level(full_pair_count)

        return {
            "status": "success",
            "target_window_count": int(target_window_count),
            "reference_window_count": int(total_reference_windows),
            "reference_count": int(len(ref_paths)),
            "average_reference_windows": float(
                total_reference_windows / len(ref_paths)
                if ref_paths else 0.0
            ),
            "full_pair_count": full_pair_count,
            "scale_level": scale_level,
            "recommendation": recommendation,
            "references": reference_summaries[:20],
        }
    finally:
        shutil.rmtree(estimate_dir, ignore_errors=True)


@app.get("/api/task_status/{task_id}")
async def check_task_status(task_id: str):
    """
    【状态轮询接口】
    前端每隔几秒钟来这里查一下进度。如果完成了，就把数据带回去。
    """
    task_info = get_task(task_id)
    if not task_info:
        return {"status": "error", "message": "任务不存在或已过期"}

    return {
        "status": "success",
        "task_status": task_info["status"],
        "data": task_info["result"],
        "message": task_info["message"],
        "cost_time": task_info["cost_time"]
    }

@app.post("/api/preview_document")
async def preview_document(file: UploadFile = File(...)):
    """
    提供给前端预览文档内容的接口。
    前端直接将文件 POST 过来，后端解析后返回纯文本内容。
    """
    temp_path = os.path.join(TEMP_DIR, f"preview_{uuid.uuid4().hex[:8]}_{file.filename}")
    try:
        with open(temp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # 实例化一个临时的传统引擎专门用来读取各种格式 (TXT/DOCX/PDF)
        temp_system = PlagiarismDetectorSystem(
            stopwords_path='dicts/stopwords.txt',
            lsa_components=3,
            synonyms_path='dicts/synonyms.txt',
            semantic_embeddings_path='dicts/embeddings/fasttext_zh.vec',
            semantic_threshold=0.55,
            semantic_weight=0.35
        )
        content = temp_system.read_document(temp_path, preview_mode=True)
        return {"status": "success", "content": content}
    except Exception as e:
        return {"status": "error", "message": str(e)}
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.get("/", include_in_schema=False)
async def serve_frontend_index():
    return serve_frontend_path()


@app.get("/{full_path:path}", include_in_schema=False)
async def serve_frontend_app(full_path: str):
    return serve_frontend_path(full_path)

if __name__ == "__main__":
    import uvicorn
    # 启动命令: uvicorn api:app --reload --host 127.0.0.1 --port 8000
    host = os.getenv("DOCSIM_HOST", "0.0.0.0")
    port = int(os.getenv("DOCSIM_PORT", "8000"))
    uvicorn.run(app, host=host, port=port)
