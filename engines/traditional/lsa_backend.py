import numpy as np


class WhiteBoxLSA:
    def __init__(self, n_components=2):
        """
        n_components: 我们要降到多少维（保留多少个潜在语义/大平层）
        """
        self.n_components = n_components
        self.U = None
        self.S = None
        self.Vt = None
        self.effective_components = n_components

    def fit_transform(self, tfidf_matrix):
        """
        核心数学过程：利用 SVD 奇异值分解进行降维
        矩阵分解公式: X = U * Sigma * V^T
        """
        print(f"\n【SVD 降维中...】原始 TF-IDF 矩阵形状: {tfidf_matrix.shape}")

        # 当文档数量极少时，强制降低维度以防止噪声主导
        n_docs, n_features = tfidf_matrix.shape
        actual_components = min(self.n_components, n_docs - 1, n_features - 1)
        if actual_components < 1:
            actual_components = 1

        # 1. 调用 numpy 的底层线性代数库进行 SVD 分解
        # full_matrices=False 表示使用经济型 SVD，节省内存
        U, S, Vt = np.linalg.svd(tfidf_matrix, full_matrices=False)

        # 2. 截断 (Truncation)：只保留前 k 个最大的奇异值
        max_possible_k = min(tfidf_matrix.shape)
        k = self.n_components
        
        # 核心修复：如果是少量文档测试，且特征极少，才进行限制
        if k > max_possible_k:
            k = max_possible_k
             
        # 【极其关键的修复】
        # 之前我们为了避免过度平滑，强制 k = max_possible_k - 1。
        # 但在处理长度差异极大的文档（比如短的开题报告，长的AI医疗）混合在一起时，
        # 如果维度不够低，长文档的噪声特征会主导整个矩阵空间，导致完全无关的长文档 LSA 相似度飙升。
        # 解决办法：严格按照用户设定的 k 值（通常为 5-10）进行截断，强行把高频特征挤压到低维空间，滤除噪声。
        if k < 1:
            k = 1
        self.effective_components = int(k)

        U_k = U[:, :k]  # 文档在潜在语义空间的分布
        S_k = np.diag(S[:k])  # 奇异值对角矩阵（代表每个语义维度的重要性）
        Vt_k = Vt[:k, :]  # 词汇在潜在语义空间的分布

        # 3. 计算降维后的文档向量矩阵 (U_k * S_k)
        lsa_matrix = np.dot(U_k, S_k)

        print(f"【SVD 降维完成】降维后的文档矩阵形状: {lsa_matrix.shape} (保留了 {k} 维核心语义特征)")
        return lsa_matrix


if __name__ == '__main__':
    from text_processing.tokenizers import TextPreprocessor
    from .tfidf_backend import WhiteBoxTFIDF
    from .similarity import calculate_cosine_similarity

    # ==========================================
    # 🌟 极其重要的一步：构建“背景语料库”
    # SVD 是一种统计学方法，如果只给它两句话，它无法学习到词和词之间的关系。
    # 所以我们需要给它喂几条包含这些词汇的“背景文章”，让它学习到“互联网”和“网络科技”是经常一起出现的。
    # 在真实的工程中，这个语料库是几万篇论文。这里我们用几句话模拟。
    # ==========================================
    corpus_texts = [
        "随着互联网技术的飞速发展，电子文档的数量呈爆炸式增长。",  # 文档0 (被查重文档)
        "伴随着网络科技的极速进步，数字化文档的数目呈现出指数级上升。",  # 文档1 (涉嫌洗稿文档)
        "互联网和网络科技推动了数字化时代的发展。",  # 背景学习材料
        "电子文档的爆炸式增长是技术进步的体现。",  # 背景学习材料
        "数量的增长和数目的上升描述了同样的趋势。"  # 背景学习材料
    ]

    # 1. 预处理与分词
    preprocessor = TextPreprocessor(stopwords_path='dicts/stopwords.txt')
    corpus_words = [preprocessor.clean_and_cut(text) for text in corpus_texts]

    # 2. TF-IDF 向量化
    vectorizer = WhiteBoxTFIDF()
    tfidf_matrix = vectorizer.fit_transform(corpus_words)

    # 我们先看看传统 TF-IDF 的查重结果（提取第0句和第1句）
    vec1_tfidf = tfidf_matrix[0]
    vec2_tfidf = tfidf_matrix[1]
    sim_tfidf = calculate_cosine_similarity(vec1_tfidf, vec2_tfidf)
    print(f"\n---> 【传统 TF-IDF】查重结果: {sim_tfidf * 100 :.2f} %")

    # ----------------------------------------------------
    # 🚀 3. 施展 LSA 降维打击！
    # 假设我们原本有几十个词汇维度，现在强行压缩到 3 维（3个核心概念）
    # ----------------------------------------------------
    lsa = WhiteBoxLSA(n_components=3)
    lsa_matrix = lsa.fit_transform(tfidf_matrix)

    # 提取降维后的第0句和第1句的新向量
    vec1_lsa = lsa_matrix[0]
    vec2_lsa = lsa_matrix[1]

    # 4. 再次计算相似度
    sim_lsa = calculate_cosine_similarity(vec1_lsa, vec2_lsa)

    print("\n===============================")
    print(f"原句1: {corpus_texts[0]}")
    print(f"原句2: {corpus_texts[1]}")
    print("===============================")
    print(f"🔥🔥🔥 【引入 LSA 降维后】查重结果: {sim_lsa * 100 :.2f} %")

    if sim_lsa > 0.6:  # 阈值设为 60%
        print("结论：系统成功识破洗稿，判定为【高度抄袭】！")
