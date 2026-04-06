import jieba
import os


class TextPreprocessor:
    def __init__(self, stopwords_path=None, synonyms_path=None):
        """
        初始化预处理器，加载停用词表和同义词表
        """
        self.stopwords = set()
        if stopwords_path and os.path.exists(stopwords_path):
            with open(stopwords_path, 'r', encoding='utf-8') as f:
                # 读取停用词，去掉两端空格，并存入集合(set)中加快查询速度
                self.stopwords = set([line.strip() for line in f.readlines()])

        # 加载同义词表
        self.synonyms = {}
        if synonyms_path and os.path.exists(synonyms_path):
            with open(synonyms_path, 'r', encoding='utf-8') as f:
                for line in f:
                    words = [w.strip() for w in line.strip().split(',') if w.strip()]
                    if len(words) > 1:
                        base_word = words[0]
                        for w in words[1:]:
                            self.synonyms[w] = base_word

        # 预设一些常见的标点符号作为默认过滤
        self.punctuations = set(['，', '。', '！', '？', '、', '；', '：', '“', '”', '\n', ' ', '（', '）'])

    def clean_and_cut(self, text):
        """
        核心方法：对输入的一段文本进行清洗和分词
        """
        # 1. 使用结巴分词进行精确模式分词
        words = jieba.cut(text, cut_all=False)

        result = []
        # 2. 过滤停用词和标点符号
        for word in words:
            word = word.strip()
            if word and word not in self.stopwords and word not in self.punctuations:
                # 同义词替换
                final_word = self.synonyms.get(word, word) if hasattr(self, 'synonyms') else word
                result.append(final_word)

        return result


# --- 以下是本地测试代码，只有直接运行这个文件时才会执行 ---
if __name__ == '__main__':
    # 模拟一段涉嫌洗稿的文本
    text1 = "随着互联网技术的飞速发展，电子文档的数量呈爆炸式增长。"
    text2 = "伴随着网络科技的极速进步，数字化文档的数目呈现出指数级上升。"

    # 实例化预处理器 (这里假设你的停用词表放在 dicts/stopwords.txt)
    preprocessor = TextPreprocessor(stopwords_path='dicts/stopwords.txt')

    # 执行分词
    words1 = preprocessor.clean_and_cut(text1)
    words2 = preprocessor.clean_and_cut(text2)

    print("原文1分词结果:", words1)
    print("原文2分词结果:", words2)