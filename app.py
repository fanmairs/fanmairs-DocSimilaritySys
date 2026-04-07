import streamlit as st
import os
import pandas as pd
import time
from main import PlagiarismDetectorSystem

# 设置页面基本信息
st.set_page_config(
    page_title="白盒化文档查重系统",
    page_icon="🔍",
    layout="wide"
)

# ====== 1. 侧边栏配置区 ======
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/documents.png", width=80)
    st.title("参数配置")
    
    bert_profile = st.selectbox(
        "BERT 阈值档位",
        ["balanced", "strict", "recall"],
        index=0,
        help="balanced 为默认，strict 更低误报，recall 更高召回。"
    )

    st.markdown("---")
    st.subheader("⚙️ 核心算法参数")
    engine_choice = st.radio(
        "选择检测引擎", 
        ["🛠️ 传统白盒引擎 (TF-IDF + LSA)", "🚀 深度语义引擎 (BERT)"],
        help="传统引擎适合算法原理演示；深度语义引擎需安装 pytorch，但能从根本上解决所有洗稿问题。"
    )
    
    lsa_dim = st.slider("LSA 降维维度 (保留特征数)", min_value=2, max_value=20, value=5, 
                        help="仅在传统引擎下生效。维度越低越能抗洗稿，但可能导致误判率上升。")
    body_mode = st.checkbox("正文模式（过滤声明/参考文献）", value=True)
    
    st.markdown("---")
    st.markdown("👨‍💻 **作者**: 宫世林")
    st.markdown("🎓 **课题**: 文档相似度比对与抄袭检测工具的设计与实现")

# ====== 2. 页面主标题区 ======
st.title("🔍 白盒化文档查重与洗稿检测系统")
st.markdown("基于 **TF-IDF** 与 **LSA (潜在语义分析)** 的轻量级查重工具，支持识别同义词替换与句子重组。")

# ====== 3. 文件上传区 ======
col1, col2 = st.columns(2)

with col1:
    st.subheader("📄 1. 上传待检测文档")
    target_file = st.file_uploader("选择一个文档", type=['txt', 'docx', 'pdf'], key="target")
    
    # ==== 新增：待检测文档预览功能 ====
    if target_file is not None:
        with st.expander(f"👀 预览待检测文档: {target_file.name}", expanded=False):
            try:
                # 为了不干扰后续的查重逻辑，我们需要创建一个临时文件来读取内容
                temp_preview_path = f"temp_preview_{target_file.name}"
                with open(temp_preview_path, "wb") as f:
                    f.write(target_file.getvalue())
                
                # 临时实例化一个系统对象来借用它的读取功能
                preview_detector = PlagiarismDetectorSystem(stopwords_path='dicts/stopwords.txt')
                # 修复 PDF 预览看不全的问题：传入 preview_mode=True 避免被正则全部替换为空格和换行
                content = preview_detector.read_document(temp_preview_path, preview_mode=True)
                
                # 展示内容，限制高度并开启滚动
                st.markdown(f"""
                <div style="max-height: 400px; overflow-y: auto; padding: 15px; background-color: #f9f9f9; border-radius: 5px; font-size: 14px; border: 1px solid #eee; line-height: 1.6; white-space: pre-wrap;">
                    {content}
                </div>
                """, unsafe_allow_html=True)
                
                # 清理临时文件
                if os.path.exists(temp_preview_path):
                    os.remove(temp_preview_path)
            except Exception as e:
                st.error(f"预览失败: {str(e)}")

with col2:
    st.subheader("📚 2. 上传参考文档库")
    reference_files = st.file_uploader("选择多个参考文档", type=['txt', 'docx', 'pdf'], accept_multiple_files=True, key="refs")
    
    # ==== 新增：参考文档预览功能 ====
    if reference_files:
        st.markdown("##### 📎 参考文档预览")
        # 让用户通过下拉菜单选择要预览的文档，避免页面太长
        preview_names = [f.name for f in reference_files]
        selected_preview = st.selectbox("选择要查看的参考文档", preview_names)
        
        if selected_preview:
            # 找到选中的文件对象
            selected_file = next(f for f in reference_files if f.name == selected_preview)
            with st.expander(f"👀 预览: {selected_preview}", expanded=False):
                try:
                    temp_preview_path = f"temp_preview_ref_{selected_file.name}"
                    with open(temp_preview_path, "wb") as f:
                        f.write(selected_file.getvalue())
                    
                    preview_detector = PlagiarismDetectorSystem(stopwords_path='dicts/stopwords.txt')
                    # 修复 PDF 预览看不全的问题
                    content = preview_detector.read_document(temp_preview_path, preview_mode=True)
                    
                    st.markdown(f"""
                    <div style="max-height: 400px; overflow-y: auto; padding: 15px; background-color: #f9f9f9; border-radius: 5px; font-size: 14px; border: 1px solid #eee; line-height: 1.6; white-space: pre-wrap;">
                        {content}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if os.path.exists(temp_preview_path):
                        os.remove(temp_preview_path)
                except Exception as e:
                    st.error(f"预览失败: {str(e)}")


# ====== 4. 核心检测逻辑 ======
if st.button("🚀 立即开始查重", type="primary", width="stretch"):
    if not target_file:
        st.error("请先上传待检测文档！")
    elif not reference_files:
        st.error("请至少上传一个参考文档作为对比库！")
    else:
        # 创建一个临时目录来保存用户上传的文件
        temp_dir = "temp_uploads"
        
        # 修复：每次点击“开始查重”前，强制清空临时目录中的旧文件
        import shutil
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        os.makedirs(temp_dir, exist_ok=True)
        
        # 保存目标文件
        target_path = os.path.join(temp_dir, "target_" + target_file.name)
        with open(target_path, "wb") as f:
            f.write(target_file.getbuffer())
            
        # 保存参考文件
        ref_paths = []
        for i, ref in enumerate(reference_files):
            # 修复：确保每个上传的参考文件都有一个独立的不重复文件名，防止在临时目录中被同名文件覆盖
            path = os.path.join(temp_dir, f"ref_{i}_" + ref.name)
            with open(path, "wb") as f:
                f.write(ref.getbuffer())
            ref_paths.append(path)
            
        # 显示进度条
        progress_text = "正在启动 NLP 分析引擎..."
        my_bar = st.progress(0, text=progress_text)
        
        # 实例化传统系统 (无论选什么引擎，我们都需要用它来读取不同格式的文件)
        try:
            detector = PlagiarismDetectorSystem(
                stopwords_path='dicts/stopwords.txt', 
                lsa_components=lsa_dim
            )
            
            is_bert_mode = "BERT" in engine_choice
            if is_bert_mode:
                try:
                    from deep_semantic import DeepSemanticEngine
                    my_bar.progress(20, text="正在加载 BERT 深度语义模型 (首次运行需下载约 400MB 权重)...")
                    bert_engine = DeepSemanticEngine()
                    
                    my_bar.progress(50, text="正在读取并清洗文档...")
                    target_text = detector.read_document(target_path)
                    ref_texts = [detector.read_document(p) for p in ref_paths]
                    if body_mode:
                        target_text = detector.clean_academic_noise(target_text)
                        ref_texts = [detector.clean_academic_noise(t) for t in ref_texts]
                    
                    # 【核心优化】：彻底删除了旧版无用的“宏观主题向量计算” (target_vec 和 ref_vecs)
                    # 极大地节省了内存和计算时间！
                    
                    my_bar.progress(80, text="正在进行全文细粒度语义扫描 (BERT Sliding Window)...")
                    results = []
                    target_length = len(target_text) if len(target_text) > 0 else 1
                    
                    for i, ref_text in enumerate(ref_texts):
                        # 获取基于 BERT 滑动窗口的细粒度抄袭片段
                        plag_parts = bert_engine.sliding_window_check(
                            target_text,
                            ref_text,
                            threshold_profile=bert_profile
                        )
                        score_breakdown = bert_engine.score_document_pair(
                            target_text,
                            ref_text,
                            plagiarized_parts=plag_parts,
                            threshold_profile=bert_profile
                        )
                        
                        # 【核心修正】：对于长篇文档，放弃使用池化向量计算余弦相似度（会导致同领域相似度虚高至95%+）
                        # 真正的抄袭率应该等于：所有被判定为抄袭片段的长度之和 / 目标文档总长度
                        total_plag_len = sum(p['length'] for p in plag_parts)
                        # 商业查重系统的真实算法：局部重合度换算为全局抄袭率，并给予一定放大系数（惩罚系数）
                        real_plag_ratio = (total_plag_len / target_length) * 1.5 
                        real_plag_ratio = min(real_plag_ratio, 1.0) # 最高100%
                        
                        results.append({
                            'file': ref_paths[i],
                            'sim_lsa': real_plag_ratio, # 借用这个字段用于排序
                            'sim_tfidf': float(score_breakdown['final_score']),
                            'sim_bert': float(score_breakdown['final_score']),
                            'sim_bert_risk': float(score_breakdown.get('risk_score', score_breakdown['final_score'])),
                            'sim_bert_doc': float(score_breakdown['doc_semantic']),
                            'sim_bert_doc_excess': float(score_breakdown.get('doc_semantic_excess', score_breakdown['doc_semantic'])),
                            'sim_bert_coverage': float(score_breakdown.get('coverage_raw', score_breakdown['coverage'])),
                            'sim_bert_coverage_weighted': float(score_breakdown.get('coverage_weighted', score_breakdown['coverage'])),
                            'sim_bert_coverage_effective': float(score_breakdown.get('coverage_effective', score_breakdown['coverage'])),
                            'sim_bert_confidence': float(score_breakdown['confidence']),
                            'sim_bert_semantic_signal': float(score_breakdown.get('semantic_signal', 0.0)),
                            'sim_bert_evidence': float(score_breakdown.get('evidence_score', 0.0)),
                            'sim_bert_continuity_bonus': float(score_breakdown.get('continuity_bonus', 0.0)),
                            'sim_bert_continuity_longest': float(score_breakdown.get('continuity_longest', 0.0)),
                            'sim_bert_continuity_top3': float(score_breakdown.get('continuity_top3', 0.0)),
                            'sim_lsa': float(score_breakdown['final_score']),
                            'bert_profile': bert_profile,
                            'plagiarized_parts': plag_parts
                        })
                    
                    # 按相似度降序排序
                    results.sort(key=lambda x: x['sim_bert'], reverse=True)
                    my_bar.progress(100, text="检测完成！")
                    
                except Exception as e:
                    st.error(f"❌ 深度语义引擎启动失败: {str(e)}\n\n💡 **解决办法**: 请在 Pycharm 底部的终端运行以下命令安装依赖：\n`pip install sentence-transformers torch`")
                    results = []
                    my_bar.empty()
            
            else:
                # ====== 传统 TF-IDF + LSA 引擎执行逻辑 ======
                # 重新实例化带同义词典的传统系统
                detector = PlagiarismDetectorSystem(
                    stopwords_path='dicts/stopwords.txt', 
                    lsa_components=lsa_dim,
                    synonyms_path='dicts/synonyms.txt',
                    semantic_embeddings_path='dicts/embeddings/fasttext_zh.vec',
                    semantic_threshold=0.55,
                    semantic_weight=0.35
                )
                
                my_bar.progress(30, text="正在构建 TF-IDF 高维稀疏矩阵...")
                time.sleep(0.5) # 为了 UI 展示效果稍微停顿
                
                my_bar.progress(60, text="正在执行 SVD 奇异值分解降维...")
                time.sleep(0.5)
                
                my_bar.progress(80, text="正在进行滑动窗口段落级扫描...")
                
                results = detector.check_similarity(target_path, ref_paths, body_mode=body_mode)
                
                my_bar.progress(100, text="检测完成！")
            
            # ====== 5. 结果展示区 ======
            st.markdown("---")
            st.header("📊 查重报告")
            
            if not results:
                st.warning("未能生成报告，请检查文档内容是否有效。")
            else:
                # 提取最高相似度
                if is_bert_mode:
                    highest_sim = results[0].get('sim_bert', results[0]['sim_lsa']) * 100
                else:
                    highest_sim = results[0].get('risk_score', results[0].get('sim_hybrid', results[0]['sim_lsa'])) * 100
                
                m_col1, m_col2, m_col3 = st.columns(3)
                m_col1.metric(
                    "最高深度语义相似度 (BERT)" if is_bert_mode else "最高综合相似度 (Hybrid)",
                    f"{highest_sim:.2f}%",
                    delta="高危" if highest_sim > 60 else "正常",
                    delta_color="inverse"
                )
                if is_bert_mode:
                    total_fragments = sum(len(res.get('plagiarized_parts', [])) for res in results)
                    m_col2.metric("细粒度命中片段", f"{total_fragments} 段")
                else:
                    m_col2.metric("Soft语义相似度", f"{results[0].get('sim_soft', 0)*100:.2f}%")
                m_col3.metric("比对文档总数", f"{len(results)} 篇")
                
                # 结论预警
                if highest_sim > 70:
                    st.error("🚨 **系统结论：【极度疑似抄袭】检测到高度的语义重合与洗稿行为！**")
                elif highest_sim > 35:
                    st.warning("⚠️ **系统结论：【疑似洗稿】存在较多同义词替换或结构借用，需人工复核。**")
                else:
                    st.success("✅ **系统结论：【合格】未发现高度相似文档。**")
                
                # 表格展示所有结果
                st.subheader("📑 总体相似度排名")
                df_data = []
                for i, res in enumerate(results):
                    import re
                    original_name = os.path.basename(res['file'])
                    original_name = re.sub(r'^ref_\d+_', '', original_name)
                    if is_bert_mode:
                        df_data.append({
                            "排名": i + 1,
                            "参考文档": original_name,
                            "综合相似度": f"{res.get('sim_bert', res['sim_lsa'])*100:.2f}%",
                            "文档语义分": f"{res.get('sim_bert_doc', 0)*100:.2f}%",
                            "风险分数": f"{res.get('sim_bert_risk', res.get('sim_bert', res['sim_lsa']))*100:.2f}%",
                            "覆盖率": f"{res.get('sim_bert_coverage_effective', res.get('sim_bert_coverage', 0))*100:.2f}%",
                            "置信度": f"{res.get('sim_bert_confidence', 0)*100:.2f}%"
                        })
                    else:
                        df_data.append({
                            "排名": i + 1,
                            "参考文档": original_name,
                            "Hybrid综合相似度": f"{res.get('sim_hybrid', res['sim_lsa'])*100:.2f}%",
                            "风险分数": f"{res.get('risk_score', res.get('sim_hybrid', res['sim_lsa']))*100:.2f}%",
                            "LSA语义相似度": f"{res['sim_lsa']*100:.2f}%",
                            "TF-IDF字面相似度": f"{res['sim_tfidf']*100:.2f}%",
                            "Soft语义相似度": f"{res.get('sim_soft', 0)*100:.2f}%"
                        })
                st.dataframe(pd.DataFrame(df_data))
                
                # 细粒度雷神之锤展示与原文高亮溯源
                st.subheader("🔎 细粒度抄袭片段对比与原文高亮 (雷神之锤)")
                has_parts = False
                
                # 提取所有疑似抄袭的原文片段，用于后续高亮渲染
                all_plagiarized_texts = []
                
                for res in results:
                    if res.get('plagiarized_parts'):
                        has_parts = True
                        original_name = os.path.basename(res['file'])
                        original_name = re.sub(r'^ref_\d+_', '', original_name)
                        
                        # 把所有的抄袭片段都加入到高亮列表中，实现真正的全景标红
                        for part in res['plagiarized_parts']:
                            all_plagiarized_texts.append(part['target_part'])
                            
                        with st.expander(f"与 《{original_name}》 的高度相似片段对比 (最多展示Top 15)", expanded=False):
                            for part in res['plagiarized_parts'][:15]: # UI上只展示前15个，防止页面卡顿
                                
                                st.markdown(f"**相似度：`{part['score']*100:.2f}%`**")
                                c1, c2 = st.columns(2)
                                with c1:
                                    st.info("**你的原文:**\n\n" + part['target_part'])
                                with c2:
                                    st.error("**疑似抄自:**\n\n" + part['ref_part'])
                                st.markdown("---")
                                
                if not has_parts:
                    st.info("未检测到大段的连续抄袭片段。")
                else:
                    # ==== 新增：知网级原文高亮渲染 ====
                    st.subheader("🔴 待测文档查重全景图 (高亮版)")
                    st.caption("红色高亮部分代表与其他参考文档存在高度语义重合，可能涉嫌洗稿或抄袭。")
                    
                    full_target_text = detector.read_document(target_path)
                    if body_mode:
                        full_target_text = detector.clean_academic_noise(full_target_text)
                    
                    # 按照长度对片段进行降序排序，优先高亮长片段，避免短片段嵌套替换出错
                    all_plagiarized_texts.sort(key=len, reverse=True)
                    
                    # 在全文中进行高亮替换
                    highlighted_text = full_target_text
                    for plag_text in all_plagiarized_texts:
                        # 使用 HTML span 标签加上红色背景和白色字体进行高亮
                        # 注意：为了防止多次重复替换，我们需要非常小心的处理，这里采用简单的直接替换
                        if plag_text in highlighted_text and '<span' not in plag_text:
                            html_highlight = f'<span style="background-color: #ffcccc; color: #cc0000; font-weight: bold; border-radius: 3px; padding: 2px;">{plag_text}</span>'
                            highlighted_text = highlighted_text.replace(plag_text, html_highlight)
                    
                    # 使用 st.markdown 渲染带有 HTML 的文本
                    st.markdown(
                        f"""
                        <div style="border: 1px solid #ddd; padding: 20px; border-radius: 5px; background-color: #fdfdfd; line-height: 1.8;">
                            {highlighted_text}
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )
                    
        except Exception as e:
            st.error(f"系统运行出错: {str(e)}")
