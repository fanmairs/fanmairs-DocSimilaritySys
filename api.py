import os
import shutil
import uuid
import sqlite3
import json
import threading
import queue
import time
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from typing import List

# 导入我们项目核心类
from main import PlagiarismDetectorSystem
from deep_semantic import DeepSemanticEngine

app = FastAPI(title="智能文档查重系统 (异步队列并发版)")

# 配置 CORS 允许跨域请求
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIST_DIR = BASE_DIR / "frontend" / "dist"
FRONTEND_INDEX_FILE = FRONTEND_DIST_DIR / "index.html"
TEMP_DIR = "temp_uploads"
os.makedirs(TEMP_DIR, exist_ok=True)

# ==========================================
# 1. 数据库初始化 (模拟企业级 MySQL)
# ==========================================
# 用于持久化存储任务的进度和结果
DB_FILE = "tasks.db"
def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS tasks
                 (id TEXT PRIMARY KEY, 
                  status TEXT, 
                  result TEXT, 
                  message TEXT,
                  cost_time REAL DEFAULT 0,
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    
    # 尝试为旧版表补充缺失字段（兼容老数据库）
    try:
        c.execute("ALTER TABLE tasks ADD COLUMN cost_time REAL DEFAULT 0")
    except sqlite3.OperationalError:
        pass
    try:
        c.execute("ALTER TABLE tasks ADD COLUMN created_at TIMESTAMP DEFAULT '1970-01-01 00:00:00'")
    except sqlite3.OperationalError:
        pass
        
    # 鲁棒性增强1：系统重启时，自动将上一次崩溃留下的“僵尸任务”标记为失败
    c.execute("UPDATE tasks SET status='failed', message='服务器重启，排队或计算中的任务已中断' WHERE status IN ('pending', 'processing')")
    
    # 鲁棒性增强2：自动清理 3 天前的陈旧任务，防止单机 SQLite 膨胀
    c.execute("DELETE FROM tasks WHERE created_at <= datetime('now', '-3 days')")
    
    conn.commit()
    conn.close()

init_db()

def update_task(task_id, status, result=None, message=None, cost_time=0):
    # 鲁棒性增强3：引入带退避的重试机制，防止 SQLite 在高并发读写时报错 "database is locked"
    for _ in range(5):
        try:
            conn = sqlite3.connect(DB_FILE, timeout=5.0)
            c = conn.cursor()
            c.execute("UPDATE tasks SET status=?, result=?, message=?, cost_time=? WHERE id=?", 
                      (status, result, message, cost_time, task_id))
            conn.commit()
            conn.close()
            break
        except sqlite3.OperationalError:
            time.sleep(0.2)

def get_task(task_id):
    conn = sqlite3.connect(DB_FILE, timeout=5.0)
    c = conn.cursor()
    c.execute("SELECT status, result, message, cost_time FROM tasks WHERE id=?", (task_id,))
    row = c.fetchone()
    conn.close()
    if row:
        return {
            "status": row[0], 
            "result": json.loads(row[1]) if row[1] else None, 
            "message": row[2],
            "cost_time": row[3]
        }
    return None

# ==========================================
# 2. 全局任务队列与后台消费者 (模拟企业级 Celery + Redis)
# ==========================================
# 所有前端请求都会被塞进这个队列，而不是直接去抢占 GPU
task_queue = queue.Queue()

# 全局模型实例，由专属后台线程加载，避免多线程抢占显存
bert_engine = None
traditional_system = None

def gpu_worker():
    """
    这是一个独立运行的后台守护线程。
    它的使命是：永远在后台排队取任务，确保同一时刻永远只有一个任务在占用 GPU。
    这就从根本上解决了 100 个人同时点击查重导致 3060 显存瞬间爆炸的问题！
    """
    global bert_engine, traditional_system
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
        session_dir = task['session_dir']
        
        print(f">>> [GPU Worker] 正在处理排队任务: {task_id}")
        update_task(task_id, "processing")
        
        # 记录开始时间，用于计算耗时
        start_time = time.time()
        
        try:
            results = []
            if mode == "bert":
                # 深度语义引擎检测逻辑
                target_text = traditional_system.read_document(target_path)
                if body_mode: target_text = traditional_system.clean_academic_noise(target_text)
                    
                for ref_path in ref_paths:
                    ref_text = traditional_system.read_document(ref_path)
                    if body_mode: ref_text = traditional_system.clean_academic_noise(ref_text)
                    
                    plagiarized_parts = bert_engine.sliding_window_check(
                        target_text,
                        ref_text,
                        threshold_profile=bert_profile
                    )
                    score_breakdown = bert_engine.score_document_pair(
                        target_text,
                        ref_text,
                        plagiarized_parts=plagiarized_parts,
                        threshold_profile=bert_profile
                    )
                    print(
                        ">>> [BGE][Score] "
                        f"file={os.path.basename(ref_path)} "
                        f"final={score_breakdown['final_score']:.4f} "
                        f"risk={score_breakdown.get('risk_score', score_breakdown['final_score']):.4f} "
                        f"doc={score_breakdown['doc_semantic']:.4f} "
                        f"doc_ex={score_breakdown.get('doc_semantic_excess', score_breakdown['doc_semantic']):.4f} "
                        f"cov={score_breakdown['coverage']:.4f} "
                        f"cov_eff={score_breakdown.get('coverage_effective', score_breakdown['coverage']):.4f} "
                        f"cov_w={score_breakdown.get('coverage_weighted', score_breakdown['coverage']):.4f} "
                        f"conf={score_breakdown['confidence']:.4f} "
                        f"base={score_breakdown['base_score']:.4f} "
                        f"gate={score_breakdown['gate']:.4f} "
                        f"hits={score_breakdown['hit_count']}"
                    )
                    
                    results.append({
                        "file": os.path.basename(ref_path).replace("ref_", ""),
                        # Backward-compatible field: now mapped to the composite BGE similarity score.
                        "sim_bert": float(score_breakdown["final_score"]),
                        "sim_bert_risk": float(score_breakdown.get("risk_score", score_breakdown["final_score"])),
                        "sim_bert_doc": float(score_breakdown["doc_semantic"]),
                        "sim_bert_doc_excess": float(score_breakdown.get("doc_semantic_excess", score_breakdown["doc_semantic"])),
                        "sim_bert_coverage": float(score_breakdown["coverage"]),
                        "sim_bert_coverage_raw": float(score_breakdown.get("coverage_raw", score_breakdown["coverage"])),
                        "sim_bert_coverage_weighted": float(score_breakdown.get("coverage_weighted", score_breakdown["coverage"])),
                        "sim_bert_coverage_effective": float(score_breakdown.get("coverage_effective", score_breakdown["coverage"])),
                        "sim_bert_confidence": float(score_breakdown["confidence"]),
                        "sim_bert_base": float(score_breakdown["base_score"]),
                        "sim_bert_gate": float(score_breakdown["gate"]),
                        "sim_bert_hits": int(score_breakdown["hit_count"]),
                        "sim_bert_semantic_signal": float(score_breakdown.get("semantic_signal", 0.0)),
                        "sim_bert_evidence": float(score_breakdown.get("evidence_score", 0.0)),
                        "sim_bert_continuity_bonus": float(score_breakdown.get("continuity_bonus", 0.0)),
                        "sim_bert_continuity_longest": float(score_breakdown.get("continuity_longest", 0.0)),
                        "sim_bert_continuity_top3": float(score_breakdown.get("continuity_top3", 0.0)),
                        "sim_bert_low_evidence_cap": float(score_breakdown.get("low_evidence_cap", 1.0)),
                        # Explicit macro coverage field for downstream display/debug.
                        "sim_bert_legacy_coverage": float(score_breakdown.get("coverage_raw", score_breakdown["coverage"])),
                        "bert_profile": bert_profile,
                        "plagiarized_parts": plagiarized_parts
                    })
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
            update_task(task_id, "completed", result=json.dumps(results), cost_time=cost_time)
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
    bert_profile: str = Form("balanced")
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
        
    # 2. 写入数据库初始状态
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("INSERT INTO tasks (id, status) VALUES (?, ?)", (task_id, "pending"))
    conn.commit()
    conn.close()
    
    # 3. 塞入后台处理队列
    safe_profile = (bert_profile or "balanced").strip().lower()
    if safe_profile not in {"strict", "balanced", "recall"}:
        safe_profile = "balanced"

    task_queue.put({
        'id': task_id,
        'target_path': target_path,
        'ref_paths': ref_paths,
        'mode': mode,
        'body_mode': body_mode,
        'bert_profile': safe_profile,
        'session_dir': session_dir
    })
    
    queue_length = task_queue.qsize()
    return {
        "status": "success", 
        "task_id": task_id, 
        "queue_length": queue_length,
        "message": f"任务已提交！前面还有 {queue_length - 1} 个人在排队..." if queue_length > 1 else "任务已提交！即将开始计算..."
    }

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


def _serve_frontend_path(request_path: str = ""):
    if not FRONTEND_INDEX_FILE.exists():
        raise HTTPException(
            status_code=503,
            detail="Frontend build not found. Run `npm run build` in the frontend directory first."
        )

    if request_path.startswith("api/"):
        raise HTTPException(status_code=404, detail="API route not found")

    if not request_path:
        return FileResponse(FRONTEND_INDEX_FILE)

    candidate = (FRONTEND_DIST_DIR / request_path).resolve()
    try:
        candidate.relative_to(FRONTEND_DIST_DIR.resolve())
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="Invalid frontend asset path") from exc

    if candidate.is_file():
        return FileResponse(candidate)

    return FileResponse(FRONTEND_INDEX_FILE)


@app.get("/", include_in_schema=False)
async def serve_frontend_index():
    return _serve_frontend_path()


@app.get("/{full_path:path}", include_in_schema=False)
async def serve_frontend_app(full_path: str):
    return _serve_frontend_path(full_path)

if __name__ == "__main__":
    import uvicorn
    # 启动命令: uvicorn api:app --reload --host 127.0.0.1 --port 8000
    host = os.getenv("DOCSIM_HOST", "0.0.0.0")
    port = int(os.getenv("DOCSIM_PORT", "8000"))
    uvicorn.run(app, host=host, port=port)
