import os
import time
import argparse
import glob
from document_readers.factory import read_document_by_type as read_document_file
from text_processing.cleaners.academic import clean_academic_noise as clean_academic_noise_text
from .scoring import calculate_risk_score, fuse_similarity_scores

# 导入我们亲手打造的四大核心模块
from text_processing.tokenizers import TextPreprocessor
from .tfidf_backend import WhiteBoxTFIDF
from .lsa_backend import WhiteBoxLSA
from .similarity import calculate_cosine_similarity
from .window_detector import WindowDetector
from .soft_semantic import SoftSemanticScorer


class PlagiarismDetectorSystem:
    def __init__(
        self,
        stopwords_path,
        lsa_components=3,
        synonyms_path=None,
        semantic_embeddings_path=None,
        semantic_threshold=0.55,
        semantic_weight=0.35
    ):
        """
        系统初始化：装配所有核心组件
        """
        print(">>> 正在启动【白盒化文档查重与洗稿检测系统】...")
        self.preprocessor = TextPreprocessor(stopwords_path, synonyms_path)
        self.vectorizer = WhiteBoxTFIDF()
        self.lsa = WhiteBoxLSA(n_components=lsa_components)
        self.semantic_weight = max(0.0, min(float(semantic_weight), 0.6))
        self.semantic_scorer = SoftSemanticScorer(
            embeddings_path=semantic_embeddings_path,
            synonyms_path=synonyms_path,
            similarity_threshold=semantic_threshold
        )
        self.window_detector = WindowDetector(
            window_size=50,
            step=25,
            synonyms_path=synonyms_path,
            semantic_embeddings_path=semantic_embeddings_path,
            semantic_threshold=semantic_threshold,
            semantic_weight=min(0.35, self.semantic_weight)
        ) # 引入段落级检测器

        # 内置一个极简的背景语料库，用于辅助 LSA 理解同义词关联
        self.background_corpus = [
            "互联网和网络科技推动了数字化时代的发展，产生了大量电子文档。",
            "知识产权与版权保护在科研领域和学术界都至关重要。",
            "爆炸式增长的数量和指数级上升的数目，都带来了严峻的挑战与困难。"
        ]

    def _fuse_similarity_scores(
        self,
        sim_lsa,
        sim_tfidf,
        sim_soft,
        target_token_len=0,
        ref_token_len=0
    ):
        return fuse_similarity_scores(
            sim_lsa,
            sim_tfidf,
            sim_soft,
            target_token_len=target_token_len,
            ref_token_len=ref_token_len,
            semantic_weight=self.semantic_weight,
        )

    @staticmethod
    def _calculate_risk_score(sim_hybrid, sim_lsa, sim_tfidf, sim_soft):
        return calculate_risk_score(sim_hybrid, sim_lsa, sim_tfidf, sim_soft)

    def clean_academic_noise(self, text):
        return clean_academic_noise_text(text)

    def read_document(self, filepath, preview_mode=False):
        return read_document_file(filepath, preview_mode=preview_mode)

    def check_similarity(self, target_file, reference_files, body_mode=False):
        """
        核心功能：将目标文档与一组参考文档进行比对
        """
        print("\n" + "=" * 50)
        print("               🚀 开始执行查重任务               ")
        print("=" * 50)

        start_time = time.time()

        # 1. 加载目标文档
        print(f"[1/5] 读取待检测文档: {os.path.basename(target_file)}")
        try:
            target_text = self.read_document(target_file)
            if body_mode:
                target_text = self.clean_academic_noise(target_text)
        except Exception as e:
            print(f"❌ 读取待检测文档失败: {e}")
            return []

        # 2. 加载参考文档
        print(f"[1/5] 读取参考文档库 (共 {len(reference_files)} 个文件)...")
        ref_texts = []
        valid_ref_files = []
        for ref_file in reference_files:
            try:
                text = self.read_document(ref_file)
                if body_mode:
                    text = self.clean_academic_noise(text)
                if text.strip(): # 忽略空文件
                    ref_texts.append(text)
                    valid_ref_files.append(ref_file)
                else:
                    print(f"⚠️ 警告: 文档 {os.path.basename(ref_file)} 未提取到有效文本(可能是纯图片扫描件)，已自动过滤。")
            except Exception as e:
                print(f"⚠️ 警告: 无法读取参考文档 {os.path.basename(ref_file)}: {e}")

        if not ref_texts:
            print("❌ 没有有效的参考文档，无法进行查重。")
            return []

        # 3. 构建全量语料库 (目标 + 参考 + 背景)
        # 索引说明: 0=目标文档, 1~N=参考文档, N+1~M=背景文档
        all_texts = [target_text] + ref_texts + self.background_corpus

        # 4. 文本预处理 (清洗+分词)
        print("[2/5] 启动 NLP 预处理与分词引擎...")
        all_words = [self.preprocessor.clean_and_cut(text) for text in all_texts]

        # 5. 提取数学特征 (TF-IDF)
        print("[3/5] 构建高维稀疏特征矩阵 (TF-IDF)...")
        tfidf_matrix = self.vectorizer.fit_transform(all_words)
        self.semantic_scorer.prepare_vocab(self.vectorizer.vocab_list)
        if self.semantic_scorer.last_vocab_size >= 80 and self.semantic_scorer.last_vector_coverage < 0.03:
            print("⚠️ 语义向量覆盖率过低，可能存在词向量语言不匹配或文件错误。")

        # 6. 施展降维打击 (LSA / SVD)
        # 注意：如果文档总数少于 n_components，SVD 会报错或自动调整，这里简单处理
        n_docs = len(all_texts)
        effective_n_components = min(self.lsa.n_components, n_docs)
        if effective_n_components != self.lsa.n_components:
            print(f"⚠️ 文档数量较少，自动调整 LSA 维度为 {effective_n_components}")
            self.lsa.n_components = effective_n_components
            
        print("[4/5] 执行 SVD 矩阵分解，映射至潜在语义空间...")
        lsa_matrix = self.lsa.fit_transform(tfidf_matrix)

        # 7. 批量计算相似度
        print("[5/5] 计算空间向量余弦夹角...")
        results = []
        
        target_vec_tfidf = tfidf_matrix[0]
        target_vec_lsa = lsa_matrix[0]
        target_token_len = len(all_words[0]) if all_words else 0

        for i, ref_file in enumerate(valid_ref_files):
            # 参考文档在矩阵中的索引是从 1 开始的
            ref_idx = i + 1
            
            ref_vec_tfidf = tfidf_matrix[ref_idx]
            ref_vec_lsa = lsa_matrix[ref_idx]

            sim_tfidf = calculate_cosine_similarity(target_vec_tfidf, ref_vec_tfidf)
            sim_lsa = calculate_cosine_similarity(target_vec_lsa, ref_vec_lsa)
            sim_soft = self.semantic_scorer.score(
                target_vec_tfidf,
                ref_vec_tfidf,
                self.vectorizer.vocab_list
            )
            ref_token_len = len(all_words[ref_idx]) if ref_idx < len(all_words) else 0
            sim_hybrid = self._fuse_similarity_scores(
                sim_lsa,
                sim_tfidf,
                sim_soft,
                target_token_len=target_token_len,
                ref_token_len=ref_token_len
            )
            risk_score = self._calculate_risk_score(sim_hybrid, sim_lsa, sim_tfidf, sim_soft)

            results.append({
                'file': ref_file,
                'sim_tfidf': sim_tfidf,
                'sim_lsa': sim_lsa,
                'sim_soft': sim_soft,
                'sim_hybrid': sim_hybrid,
                'risk_score': risk_score
            })

        end_time = time.time()
        print(f"⏱️ 总耗时: {end_time - start_time:.4f} 秒")

        # 按组合相似度降序排列
        results.sort(key=lambda x: x.get('sim_hybrid', x['sim_lsa']), reverse=True)

        # ====== 鏂板锛氬鏋滄槸楂樺害鐩镐技鐨勬枃妗ｏ紝鎵ц缁嗙矑搴︽钀芥娴?======
        print("[6/5] 姝ｅ湪瀵归珮鍗辨枃妗ｆ墽琛屾钀界骇鎵弿...")
        for res in results:
            # 鏀惧缁嗙矑搴︽娴嬬殑闂ㄦ锛氬彧瑕?LSA 澶т簬 30%锛屾垨鑰?TF-IDF 澶т簬 30%锛岄兘杩涜缁嗙矑搴︽壂鎻?
            # 鍥犱负鍗充娇鏁寸瘒鏂囩珷鐩镐技搴﹀彧鏈?40%锛屼篃鍙兘鎰忓懗鐫€鍏朵腑鏈夋暣鏁翠竴娈垫槸瀹屽叏鎶勮鐨勶紒
            if res.get('risk_score', 0) > 0.3 or res.get('sim_hybrid', 0) > 0.3 or res['sim_lsa'] > 0.3 or res['sim_tfidf'] > 0.3:
                try:
                    ref_text = self.read_document(res['file'])
                    if body_mode:
                        ref_text = self.clean_academic_noise(ref_text)
                    res['plagiarized_parts'] = self.window_detector.check(target_text, ref_text)
                except Exception as e:
                    res['plagiarized_parts'] = []
            else:
                res['plagiarized_parts'] = []

        return results

    def print_report(self, target_file, results, top_n=5):
        """
        打印查重报告
        """
        print("\n" + "=" * 20 + " 📊 最终检测报告 " + "=" * 20)
        print(f"📄 待检文档: {os.path.basename(target_file)}")
        print(f"📚 比对库规模: {len(results)} 个文档")
        print("-" * 102)
        print(f"{'排名':<6}{'风险分':<12}{'综合分':<12}{'LSA':<12}{'TF-IDF':<12}{'Soft':<12}{'参考文档'}")
        print("-" * 102)

        for i, res in enumerate(results[:top_n]):
            rank = i + 1
            hybrid_score = res.get('sim_hybrid', res['sim_lsa'])
            risk_score = res.get('risk_score', hybrid_score)
            lsa_score = res['sim_lsa']
            tfidf_score = res['sim_tfidf']
            soft_score = res.get('sim_soft', 0.0)
            filename = os.path.basename(res['file'])
            
            # 标记高危结果
            mark = ""
            if risk_score > 0.7: mark = "🚨"
            elif risk_score > 0.35: mark = "⚠️"
            
            print(
                f"{rank:<6}"
                f"{risk_score*100:6.2f}% {mark:<3}"
                f"{hybrid_score*100:6.2f}%   "
                f"{lsa_score*100:8.2f}%"
                f"{tfidf_score*100:8.2f}%"
                f"{soft_score*100:8.2f}%"
                f"   {filename}"
            )

        print("-" * 102)
        
        # ====== 打印细粒度抄袭片段 ======
        for res in results[:top_n]:
            if res.get('plagiarized_parts'):
                filename = os.path.basename(res['file'])
                print(f"\n[🔍 细粒度分析] 目标文档在与 《{filename}》 的比对中发现高度疑似抄袭片段：")
                for part_info in res['plagiarized_parts']:
                    print(f"  > 相似度: {part_info['score']*100:.2f}%")
                    print(f"  > 你的原文: {part_info['target_part'][:100]}...")
                    print(f"  > 疑似抄自: {part_info['ref_part'][:100]}...")
                    print("  " + "-"*40)
        
        # 总体结论
        top_risk_score = results[0].get('risk_score', results[0].get('sim_hybrid', results[0]['sim_lsa'])) if results else 0
        if results and top_risk_score > 0.7:
            print("🚨 结论：【极度疑似抄袭】检测到高度的语义重合与洗稿行为！")
        elif results and top_risk_score > 0.35:
            print("⚠️ 结论：【疑似洗稿】存在较多同义词替换或结构借用，需人工复核。")
        else:
            print("✅ 结论：【合格】未发现高度相似文档。")
        print("=" * 60 + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="文档相似度比对与抄袭检测工具")
    parser.add_argument('target', nargs='?', help="待检测的文档路径 (例如: data/检测文档.txt)")
    parser.add_argument('reference', nargs='?', help="参考文档路径或目录 (例如: data/doc2.txt 或 data/)")
    parser.add_argument('--stopwords', default='dicts/stopwords.txt', help="停用词表路径")
    parser.add_argument('--lsa_dim', type=int, default=3, help="LSA 降维维度")
    parser.add_argument(
        '--semantic_embeddings',
        default='dicts/embeddings/fasttext_zh.vec',
        help="词向量模型路径(.vec)，用于传统模式同义词穿透"
    )
    parser.add_argument('--semantic_threshold', type=float, default=0.55, help="词向量相似阈值")
    parser.add_argument('--semantic_weight', type=float, default=0.35, help="语义分数融合权重(0~0.6)")

    args = parser.parse_args()

    # 默认配置 (如果用户在 PyCharm 中直接点击运行，没有传参数，默认用这个)
    # 你可以在这里修改默认的待检测文档！
    target_file = args.target if args.target else 'data/AI医疗_原文.txt'
    reference_input = args.reference if args.reference else 'data/'
    stopwords_file = args.stopwords

    # 处理参考文档列表
    reference_files = []
    
    # 修改这里的逻辑：如果用户什么参数都没传，默认扫描 data/ 目录
    if not args.reference:
        reference_input = "data/"
        
    if os.path.isdir(reference_input):
        # 如果是目录，扫描所有 .txt 文件
        reference_files = glob.glob(os.path.join(reference_input, "*.txt"))
        # 排除目标文档自己
        reference_files = [f for f in reference_files if os.path.abspath(f) != os.path.abspath(target_file)]
    else:
        # 如果是单个文件
        if os.path.exists(reference_input):
            reference_files = [reference_input]
        else:
            print(f"❌ 参考文档路径不存在: {reference_input}")
            # 如果指定的文件不存在，再尝试作为目录扫描
            print("尝试扫描默认 data 目录...")
            reference_files = glob.glob(os.path.join("data", "*.txt"))
            reference_files = [f for f in reference_files if os.path.abspath(f) != os.path.abspath(target_file)]

    if not os.path.exists(target_file):
        print(f"❌ 待检测文档不存在: {target_file}")
        print("请提供正确的文件路径，或确保 data/检测文档.txt 存在。")
        exit(1)

    if not reference_files:
        print("❌ 未找到任何参考文档。")
        exit(1)

    # 实例化系统并执行
    detector = PlagiarismDetectorSystem(
        stopwords_path=stopwords_file,
        lsa_components=args.lsa_dim,
        synonyms_path='dicts/synonyms.txt',
        semantic_embeddings_path=args.semantic_embeddings,
        semantic_threshold=args.semantic_threshold,
        semantic_weight=args.semantic_weight
    )
    results = detector.check_similarity(target_file, reference_files)
    detector.print_report(target_file, results)
