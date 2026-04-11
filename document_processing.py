import os


def clean_academic_noise(text):
    import re
    cleaned = re.sub(r'\s+', ' ', text).strip()

    # 1. 基础学术元数据清洗
    cleaned = re.sub(r'(学校代码|学号|分类号|中图分类号|基金项目|DOI|收稿日期|修回日期|录用日期|作者简介|通信作者)\s*[:：]?\s*[^\n。；;]*', ' ', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'(硕士学位论文|博士学位论文|专业学位硕士学位论文|Dissertation|Thesis)', ' ', cleaned, flags=re.IGNORECASE)

    # 2. 移除前置声明、目录和纯英文摘要
    front_cut_patterns = [
        r'(学位论文原创性声明|独创性声明|版权声明|授权声明|学位论文使用授权书)[\s\S]{0,12000}(摘要|摘\s*要|关键词|关键字|引言|绪论|前言|第[一二三四五六七八九十]章)',
        r'(Abstract|英文摘要)[\s\S]{0,15000}(关键词|关键字|Key\s*Words|引言|绪论|前言|第[一二三四五六七八九十]章)'
    ]
    for p in front_cut_patterns:
        cleaned = re.sub(p, r'\2', cleaned, flags=re.IGNORECASE)

    start_markers = ['引言', '绪论', '前言', '第一章', '1 引言', '1. 引言', '一、引言']
    end_markers = ['参考文献', 'References', '致谢', 'Acknowledgements', '附录', 'Appendix', '攻读学位期间取得的成果', '攻读硕士学位期间发表的论文']
    start_positions = [cleaned.find(m) for m in start_markers if cleaned.find(m) != -1]
    if start_positions:
        start_idx = min(start_positions)
        end_positions = [cleaned.find(m, start_idx + 1) for m in end_markers if cleaned.find(m, start_idx + 1) != -1]
        end_idx = min(end_positions) if end_positions else len(cleaned)
        if end_idx - start_idx > 1000:
            cleaned = cleaned[start_idx:end_idx]

    # 3. 【核心新增】学术特定噪音清洗（图表、公式、数字、引用）

    # 3.1 移除学术引用标号 (如 [1], [1-3], [1,2])
    cleaned = re.sub(r'\[\s*\d+(?:\s*[,\-~]\s*\d+)*\s*\]', '', cleaned)

    # 3.2 移除图表标题及图表内文引用 (如 "图1-1", "表2.3", "如图1所示", "见表2")
    # 匹配独立的图表标题，通常在段落开头或独立成段
    cleaned = re.sub(r'(?:图|表|Figure|Table|Fig\.)\s*[\dA-Za-z]+[\.\-]?\d*\s+[^\n。；;]{0,30}(?=[。；;\n]|$)', ' ', cleaned, flags=re.IGNORECASE)
    # 匹配内文中的引用，如 (如图1所示), (见表2)
    cleaned = re.sub(r'[(（]?(?:如|见)?(?:图|表|Figure|Table|Fig\.)\s*[\dA-Za-z]+[\.\-]?\d*(?:所示)?[)）]?', '', cleaned, flags=re.IGNORECASE)

    # 3.3 移除独立公式及数学符号噪音
    # 匹配 LaTeX 格式的行内或块级公式
    cleaned = re.sub(r'\$\$.*?\$\$', ' ', cleaned)
    cleaned = re.sub(r'\$.*?\$', ' ', cleaned)
    # 匹配常见的独立数学等式 (如 a = b + c, E = mc^2)
    cleaned = re.sub(r'[A-Za-z0-9\s\+\-\*/\(\)\=\<\>\≈\±\µ\α\β\γ\θ\∑\∫]{10,}(?=[。；;\n]|$)', ' ', cleaned)

    # 3.4 清洗论文目录残余（包含大量点号和页码的结构）
    cleaned = re.sub(r'[\u4e00-\u9fa5A-Za-z]+[\s\.\…]+[0-9IVX]{1,3}', ' ', cleaned)

    # 3.5 清洗残余的纯数字序号章节头 (如 "1.1.1", "1.1.2", "1.2.1")
    cleaned = re.sub(r'(?:\d+\.){2,}\d+', ' ', cleaned)
    # 清洗残余的章节词 (如 "第2章", "第三章") 连续出现的情况
    cleaned = re.sub(r'(第[0-9一二三四五六七八九十百]+章\s*)+', ' ', cleaned)

    # 4. 文本段落重组与中英文比例过滤
    segments = re.split(r'(?<=[。！？；;])', cleaned)
    kept = []
    for seg in segments:
        s = seg.strip()
        if not s:
            continue
        chinese_count = len(re.findall(r'[\u4e00-\u9fa5]', s))
        latin_count = len(re.findall(r'[A-Za-z]', s))
        # 过滤掉非自然语言段落（如全是字母公式的段落、乱码段落）
        # 只有当包含一定量中文字符，并且中文字符多于英文字母时才保留
        if chinese_count >= 5 and chinese_count >= latin_count * 0.5:
            kept.append(s)

    if kept:
        cleaned = ''.join(kept)

    # 再次清理多余空格
    cleaned = re.sub(r'\s+', ' ', cleaned)

    return cleaned.strip()


def read_document(filepath, preview_mode=False):
    """
    统一的文件读取接口：支持 txt, docx, pdf
    preview_mode=True 时，保留文档的原始换行和段落结构，用于前端 UI 预览
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"找不到文件: {filepath}")

    ext = os.path.splitext(filepath)[1].lower()

    try:
        if ext == '.txt':
            # 尝试多种编码读取 txt
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    return f.read().strip()
            except UnicodeDecodeError:
                with open(filepath, 'r', encoding='gbk') as f:
                    return f.read().strip()

        elif ext in ['.doc', '.docx']:
            try:
                from docx import Document
            except ImportError:
                raise ImportError("请先安装 python-docx 库: pip install python-docx")
            doc = Document(filepath)
            return "\n".join([para.text for para in doc.paragraphs]).strip()

        elif ext == '.pdf':
            try:
                import fitz  # PyMuPDF
            except ImportError:
                raise ImportError("请先安装 PyMuPDF 库: pip install PyMuPDF")
            text = ""
            with fitz.open(filepath) as doc:
                for page in doc:
                    # 核心修复：使用 get_text("blocks") 解决双栏/多栏排版的乱序问题
                    # 它会将页面划分为多个物理文本块，并按正确的阅读顺序（先左栏后右栏）返回
                    blocks = page.get_text("blocks")
                    # 过滤掉图片块等非文本内容 (block的类型标识在最后一个元素，0表示文本)
                    text_blocks = [b[4] for b in blocks if b[-1] == 0]
                    for tb in text_blocks:
                        text += tb + "\n"

            if not preview_mode:
                # 【核心优化】：保留自然段落边界，不要粗暴地把 \n 全部替换成空格
                # 我们用一个特殊的占位符来暂时保护双换行（段落边界），然后再清洗空格
                import re
                text = re.sub(r'\n\s*\n', '<PARA_BREAK>', text) # 保护真实段落
                text = re.sub(r'\n', '', text)                  # 去除 PDF 强制换行造成的断句
                text = re.sub(r'<PARA_BREAK>', '\n\n', text)    # 恢复段落边界
                text = re.sub(r' {2,}', ' ', text)              # 将多余空格替换为单空格

                # 过滤纯英文长段落（如英文摘要）
                # text = re.sub(r'[a-zA-Z\s]{50,}', '', text)

            return text.strip()

        else:
            raise ValueError(f"不支持的文件格式: {ext}。目前仅支持 .txt, .docx, .pdf")

    except Exception as e:
        raise ValueError(f"读取文件失败 ({filepath}): {str(e)}")
