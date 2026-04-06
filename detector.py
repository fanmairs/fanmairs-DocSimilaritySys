import re
from preprocess import TextPreprocessor
from vectorize import WhiteBoxTFIDF
from similarity import calculate_cosine_similarity
from soft_semantic import SoftSemanticScorer


class WindowDetector:
    def __init__(
        self,
        window_size=30,
        step=15,
        synonyms_path=None,
        semantic_embeddings_path='dicts/embeddings/fasttext_zh.vec',
        semantic_threshold=0.55,
        semantic_weight=0.25
    ):
        # 保留原来的参数以防其他地方调用报错，但我们现在改用自然句切分
        self.window_size = window_size  
        self.step = step  
        self.synonyms_path = synonyms_path
        self.semantic_weight = max(0.0, min(float(semantic_weight), 0.5))
        self.semantic_scorer = SoftSemanticScorer(
            embeddings_path=semantic_embeddings_path,
            synonyms_path=synonyms_path,
            similarity_threshold=semantic_threshold,
            max_terms=140
        )

    def sliding_window(self, text):
        """
        进阶版：滑动自然句窗口 (Overlapping Sentences)
        将连续的 N 个自然句拼接成一个窗口，滑动步长为 1 句。
        这能有效应对抄袭者“拆分/合并句子”的洗稿行为，并保留足够的上下文特征。
        """
        # 1. 先切分出所有的自然句 (保留标点)
        raw_sentences = re.split(r'([。？！；\n]+)', text)
        sentences = []
        for i in range(0, len(raw_sentences)-1, 2):
            sentence = raw_sentences[i] + raw_sentences[i+1]
            sentence = sentence.strip()
            if len(sentence) > 5:  # 放宽单句过滤条件
                sentences.append(sentence)
                
        if len(raw_sentences) % 2 != 0 and len(raw_sentences[-1].strip()) > 5:
            sentences.append(raw_sentences[-1].strip())

        # 2. 组装“滑动句群窗口” (N-gram sentences)
        # 例如：N=2，则 [句1+句2, 句2+句3, 句3+句4]
        windows = []
        N = 2  # 每次将 2 个自然句捆绑在一起
        
        if len(sentences) < N:
            return sentences  # 如果文章特别短，就直接返回原句
            
        for i in range(len(sentences) - N + 1):
            # 将 N 个相邻的句子拼接成一个大的检测窗口
            combined_window = " ".join(sentences[i : i+N])
            windows.append(combined_window)
            
        return windows

    def check(self, text1, text2):
        """细粒度对比：窗口 vs 窗口"""
        pre = TextPreprocessor(stopwords_path='dicts/stopwords.txt', synonyms_path=self.synonyms_path)

        # 1. 切片
        win1 = self.sliding_window(text1)
        win2 = self.sliding_window(text2)

        # 2. 预处理所有窗口
        all_words = [pre.clean_and_cut(w) for w in win1 + win2]

        # 3. 向量化并降维 (调用之前的逻辑)
        vec = WhiteBoxTFIDF()
        tfidf = vec.fit_transform(all_words)
        self.semantic_scorer.prepare_vocab(vec.vocab_list)

        # ====== 修复核心点：避免在窗口对比时过度降维 ======
        # 因为窗口切分后会产生几十个小片段，如果强行降到 2 维，
        # 所有句子都会被挤压在同一个狭窄的平面里，导致随便两句话相似度都是 100%
        # 但如果是跨格式文档对比，短小的标题也可能被切成窗口，导致它们完全匹配
        # 不要降维太多！细粒度的句子本来词汇就少，如果降维会导致误判。
        # 我们在这里直接使用 TF-IDF 的余弦相似度来进行细粒度匹配，或者使用极高的维度保留所有特征。
        # 在句子级别，TF-IDF 字面匹配往往比 LSA 更精准，因为不需要“大平层”概念了。
        
        # 4. 逐个窗口比对
        n1 = len(win1)
        results = []
        for i in range(n1):
            max_tfidf_sim = 0
            best_match_idx = -1
            
            # 只有当第二篇文章也有窗口时才进行比较
            if len(win2) > 0:
                # 注意：我们这里改为直接计算 TF-IDF 的相似度！
                # 抛弃 LSA 降维在极短文本上的误判（100% 现象）
                for j in range(n1, len(tfidf)):
                    # tfidf 是稀疏矩阵，直接转为数组计算
                    vec1 = tfidf[i].toarray()[0] if hasattr(tfidf[i], 'toarray') else tfidf[i]
                    vec2 = tfidf[j].toarray()[0] if hasattr(tfidf[j], 'toarray') else tfidf[j]
                    
                    sim_tfidf = calculate_cosine_similarity(vec1, vec2)
                    if sim_tfidf > max_tfidf_sim: 
                        max_tfidf_sim = sim_tfidf
                        best_match_idx = j - n1 # 记录在 win2 中的索引

            max_soft_sim = 0.0
            max_sim = max_tfidf_sim
            # 仅在最佳候选上补算 soft，相比全对全更快
            if best_match_idx != -1:
                j_best = best_match_idx + n1
                vec1 = tfidf[i].toarray()[0] if hasattr(tfidf[i], 'toarray') else tfidf[i]
                vec2 = tfidf[j_best].toarray()[0] if hasattr(tfidf[j_best], 'toarray') else tfidf[j_best]
                max_soft_sim = self.semantic_scorer.score(vec1, vec2, vec.vocab_list)
                max_sim = (1 - self.semantic_weight) * max_tfidf_sim + self.semantic_weight * max_soft_sim

            # 我们将阈值设定在一个合理的范围（字面相似度超过 8% 即判定疑似）
            # 过滤掉极短的废话匹配（必须大于40个字符，屏蔽掉姓名、学院、小标题等）
            if max_sim > 0.08 and len(win1[i]) > 40:  
                if not win1[i].strip() or (best_match_idx != -1 and not win2[best_match_idx].strip()):
                    continue
                results.append({
                    'target_part': win1[i],
                    'ref_part': win2[best_match_idx] if best_match_idx != -1 else "",
                    'score': max_sim,
                    'score_tfidf': max_tfidf_sim,
                    'score_soft': max_soft_sim,
                    'length': len(win1[i])
                })

        # 【核心优化】：按“综合威胁度”降序排列，而不仅仅是相似度！
        # 威胁度 = 相似度 * log(文本长度)
        # 这样可以保证：长篇大论的洗稿（相似度20%）排在 简短标题的复制（相似度100%）前面
        import math
        results.sort(key=lambda x: x['score'] * math.log(x['length'] + 1), reverse=True)
        
        # 去重逻辑：如果某一段包含在另一段里，只保留最长的那段
        filtered_results = []
        seen_texts = set()
        for res in results:
            if res['target_part'] not in seen_texts:
                filtered_results.append(res)
                seen_texts.add(res['target_part'])
                
        # 防止输出太多，只返回最相似的前 N 个片段
        return filtered_results[:10]


if __name__ == '__main__':
    text1 = "随着互联网技术的飞速发展，电子文档的数量呈爆炸式增长。无论是在学术界还是工业界，版权保护都面临着严峻的挑战。"
    text2 = "伴随着网络科技的极速进步，数字化文档的数目呈现出指数级上升。"

    wd = WindowDetector()
    plagiarized_parts = wd.check(text1, text2)

    print("\n🔍【细粒度查重扫描结果】:")
    for part, score in plagiarized_parts:
        print(f"疑似抄袭片段: {part} | 相似度: {score * 100:.2f}%")
