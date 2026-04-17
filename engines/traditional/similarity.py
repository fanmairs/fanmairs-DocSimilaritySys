import numpy as np


def calculate_cosine_similarity(vector1, vector2):
    """
    白盒化计算余弦相似度： (A · B) / (||A|| * ||B||)
    """
    # 1. 计算两个向量的“点积” (对应维度相乘后求和)
    dot_product = np.dot(vector1, vector2)

    # 2. 计算两个向量的“模长” (各维度平方和的开方)
    norm_v1 = np.linalg.norm(vector1)
    norm_v2 = np.linalg.norm(vector2)

    # 3. 避免分母为0导致报错
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0

    # 4. 套用公式
    cosine_sim = dot_product / (norm_v1 * norm_v2)
    return cosine_sim


if __name__ == '__main__':
    # 为了连贯，我们把前面写的预处理和向量化模块导进来
    from text_processing.tokenizers import TextPreprocessor
    from .tfidf_backend import WhiteBoxTFIDF

    # 1. 我们的测试用例（典型的同义词洗稿）
    text1 = "随着互联网技术的飞速发展，电子文档的数量呈爆炸式增长。"
    text2 = "伴随着网络科技的极速进步，数字化文档的数目呈现出指数级上升。"

    # 2. 预处理分词
    preprocessor = TextPreprocessor(stopwords_path='dicts/stopwords.txt')
    words1 = preprocessor.clean_and_cut(text1)
    words2 = preprocessor.clean_and_cut(text2)

    # 3. TF-IDF 向量化
    vectorizer = WhiteBoxTFIDF()
    tfidf_matrix = vectorizer.fit_transform([words1, words2])

    # 4. 提取出文档1和文档2的数学向量
    vec1 = tfidf_matrix[0]
    vec2 = tfidf_matrix[1]

    # 5. 计算最终相似度！
    similarity_score = calculate_cosine_similarity(vec1, vec2)

    print("\n===============================")
    print(f"原句1: {text1}")
    print(f"原句2: {text2}")
    print("===============================")
    print(f"【传统 TF-IDF 查重结果】相似度: {similarity_score * 100 :.2f} %")

    if similarity_score < 0.3:
        print("结论：系统判定为【原创】 (但我们都知道这是洗稿抄袭！)")
