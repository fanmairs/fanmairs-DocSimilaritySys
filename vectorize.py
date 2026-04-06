import numpy as np
import math


class WhiteBoxTFIDF:
    def __init__(self):
        self.vocab = {}  # 词汇表：存储 "词" -> "索引" 的映射
        self.idf_weights = {}  # 存储每个词的 IDF 权重
        self.vocab_list = []  # 存储词汇的列表，方便按索引查看词语

    def fit_transform(self, corpus_words):
        """
        核心方法：输入所有文档的分词结果，输出 TF-IDF 矩阵
        corpus_words: 一个二维列表，例如 [['随着', '互联网', '发展'], ['网络', '科技', '进步']]
        """
        num_docs = len(corpus_words)

        # 1. 构建全局词汇表 (Vocabulary)
        vocab_set = set()
        for words in corpus_words:
            for word in words:
                vocab_set.add(word)

        self.vocab_list = list(vocab_set)
        self.vocab = {word: idx for idx, word in enumerate(self.vocab_list)}
        vocab_size = len(self.vocab)

        print(f"【白盒展示】共提取了 {vocab_size} 个不重复的特征词汇。\n")

        # 2. 计算 TF (词频矩阵)
        # 创建一个 全是0 的矩阵，行数是文档数，列数是词汇数
        tf_matrix = np.zeros((num_docs, vocab_size))

        for doc_idx, words in enumerate(corpus_words):
            doc_len = len(words)
            for word in words:
                word_idx = self.vocab[word]
                # 计算词频：该词在本文档出现的次数 / 本文档总词数
                tf_matrix[doc_idx, word_idx] += 1.0 / doc_len

        # 3. 计算 IDF (逆文档频率)
        idf_vector = np.zeros(vocab_size)
        for word, word_idx in self.vocab.items():
            # 统计包含该词的文档数量
            doc_count_containing_word = sum(1 for words in corpus_words if word in words)
            # IDF 公式：log( 总文档数 / (包含该词的文档数 + 1) )，加1是为了防止分母为0
            # 原来的代码：
            # idf_value = math.log(num_docs / (doc_count_containing_word + 1))

            # 替换为工业标准平滑公式（避免负数和0，分子分母都加1）：
            idf_value = math.log((1 + num_docs) / (1 + doc_count_containing_word)) + 1.0
            idf_vector[word_idx] = idf_value
            self.idf_weights[word] = idf_value

        # 4. 计算 TF-IDF 矩阵 (TF矩阵 乘以 IDF向量)
        tfidf_matrix = tf_matrix * idf_vector

        return tfidf_matrix


# --- 本地测试代码 ---
if __name__ == '__main__':
    # 假设这是我们刚才 preprocess.py 跑出来的分词结果
    doc1_words = ['随着', '互联网', '技术', '飞速发展', '电子', '文档', '数量', '呈', '爆炸式', '增长']
    doc2_words = ['伴随着', '网络', '科技', '极速', '进步', '数字化', '文档', '数目', '呈现出', '指数级', '上升']

    # 组合成一个语料库
    corpus = [doc1_words, doc2_words]

    # 实例化我们的白盒 TF-IDF 模型
    vectorizer = WhiteBoxTFIDF()

    # 执行数学转换
    tfidf_matrix = vectorizer.fit_transform(corpus)

    # 打印结果看看
    print("【全局词汇表的前5个词】:", vectorizer.vocab_list[:5])
    print("\n【最终生成的 TF-IDF 矩阵】 (行=文档，列=词汇特征):")
    # 设置 numpy 打印格式，保留3位小数，方便观看
    np.set_printoptions(precision=3, suppress=True)
    print(tfidf_matrix)

    print(f"\n文档1现在的数学模样（特征向量）形状: {tfidf_matrix[0].shape}")