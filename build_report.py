from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_ALIGN_VERTICAL, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor


PROJECT = Path(__file__).resolve().parent
OUTPUT = PROJECT / "WaveUNet语音去噪实验报告.docx"


def set_font(run, name="宋体", size=10.5, bold=False):
    run.font.name = name
    run._element.rPr.rFonts.set(qn("w:eastAsia"), name)
    run._element.rPr.rFonts.set(qn("w:ascii"), "Times New Roman")
    run._element.rPr.rFonts.set(qn("w:hAnsi"), "Times New Roman")
    run.font.size = Pt(size)
    run.bold = bold


def shade(cell, fill):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:fill"), fill)
    tc_pr.append(shd)


def borders(table):
    tbl_pr = table._tbl.tblPr
    elem = OxmlElement("w:tblBorders")
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        tag = OxmlElement(f"w:{edge}")
        tag.set(qn("w:val"), "single")
        tag.set(qn("w:sz"), "4")
        tag.set(qn("w:color"), "D9D9D9")
        elem.append(tag)
    tbl_pr.append(elem)


def style_table(table):
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.style = "Table Grid"
    borders(table)
    for row_index, row in enumerate(table.rows):
        for cell in row.cells:
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            if row_index == 0:
                shade(cell, "1F4E78")
            elif row_index % 2 == 0:
                shade(cell, "F2F6FA")
            for paragraph in cell.paragraphs:
                paragraph.paragraph_format.space_after = Pt(2)
                paragraph.paragraph_format.space_before = Pt(2)
                paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
                for run in paragraph.runs:
                    set_font(run, size=9, bold=(row_index == 0))
                    if row_index == 0:
                        run.font.color.rgb = RGBColor(255, 255, 255)


def add_para(doc, text="", bold_prefix=None, align=None):
    p = doc.add_paragraph()
    p.paragraph_format.line_spacing = 1.5
    p.paragraph_format.space_after = Pt(6)
    p.paragraph_format.first_line_indent = Cm(0.74)
    if align is not None:
        p.alignment = align
    if bold_prefix and text.startswith(bold_prefix):
        a = p.add_run(bold_prefix)
        set_font(a, bold=True)
        b = p.add_run(text[len(bold_prefix):])
        set_font(b)
    else:
        r = p.add_run(text)
        set_font(r)
    return p


def add_bullet(doc, text):
    p = doc.add_paragraph(style="List Bullet")
    p.paragraph_format.line_spacing = 1.35
    p.paragraph_format.space_after = Pt(3)
    r = p.add_run(text)
    set_font(r)


def add_caption(doc, text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(8)
    r = p.add_run(text)
    set_font(r, size=9)


def heading(doc, text, level=1):
    p = doc.add_heading(text, level=level)
    p.paragraph_format.space_before = Pt(12 if level == 1 else 8)
    p.paragraph_format.space_after = Pt(6)
    for run in p.runs:
        set_font(run, name="黑体", size=(15 if level == 1 else 12), bold=True)
        run.font.color.rgb = RGBColor(0, 0, 0)
    return p


def add_page_number(paragraph):
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = paragraph.add_run("第 ")
    set_font(run, size=9)
    fld = OxmlElement("w:fldSimple")
    fld.set(qn("w:instr"), "PAGE")
    paragraph._p.append(fld)
    run = paragraph.add_run(" 页")
    set_font(run, size=9)


def add_figure(doc, path, caption, width=15.5):
    if path.exists():
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = Pt(4)
        p.paragraph_format.space_after = Pt(3)
        p.add_run().add_picture(str(path), width=Cm(width))
        add_caption(doc, caption)


doc = Document()
section = doc.sections[0]
section.top_margin = Cm(2.3)
section.bottom_margin = Cm(2.1)
section.left_margin = Cm(2.5)
section.right_margin = Cm(2.5)

normal = doc.styles["Normal"]
normal.font.name = "宋体"
normal._element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")
normal.font.size = Pt(10.5)

footer = section.footer.paragraphs[0]
add_page_number(footer)

# 封面
doc.add_paragraph().paragraph_format.space_after = Pt(52)
title = doc.add_paragraph(style="Title")
title.alignment = WD_ALIGN_PARAGRAPH.CENTER
title.paragraph_format.space_after = Pt(18)
r = title.add_run("基于 Wave U Net 的语音去噪算法设计与实验分析")
set_font(r, name="黑体", size=22, bold=True)

subtitle = doc.add_paragraph()
subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
subtitle.paragraph_format.space_after = Pt(50)
r = subtitle.add_run("Wave U Net BiLSTM Self Attention")
set_font(r, name="Times New Roman", size=13)

info = doc.add_table(rows=4, cols=2)
info.alignment = WD_TABLE_ALIGNMENT.CENTER
info.autofit = False
labels = ["课程名称", "学生姓名", "学号", "指导教师"]
for i, label in enumerate(labels):
    info.cell(i, 0).width = Cm(4)
    info.cell(i, 1).width = Cm(9)
    info.cell(i, 0).text = label
    info.cell(i, 1).text = ""
style_table(info)
doc.add_paragraph().paragraph_format.space_after = Pt(20)
p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = p.add_run("提交日期：________________")
set_font(r, size=11)
doc.add_page_break()

heading(doc, "摘 要", 1)
add_para(doc, "语音去噪的目标是在尽可能消除背景噪声的同时保留语音内容、可懂度和自然度。针对传统频谱掩码方法依赖相位、难以直接建模长程时序关系的问题，本实验设计并实现了一个端到端的 Wave U Net 语音去噪模型。模型以 16 kHz 带噪单声道波形为输入，通过一维卷积编码器提取多尺度局部特征，在瓶颈层引入双向长短期记忆网络和多头自注意力建模全局上下文，再利用解码器和跳跃连接恢复时域细节，最终以残差形式输出去噪波形。")
add_para(doc, "实验采用 Edinburgh Noisy Speech Database 的配对语音数据训练，并在官方 824 条完整测试语音上进行评估。结果表明，本模型的 SNR、SSNR、PESQ NB、PESQ WB、STOI 和 SI SDR 均优于带噪输入、谱减法及维纳滤波基线。模型在低输入信噪比条件下改善尤为明显，说明所设计的时域网络具有较好的噪声抑制和语音保真能力。")
add_para(doc, "关键词：语音去噪；Wave U Net；BiLSTM；自注意力；SI SDR；PESQ")

heading(doc, "1 引言", 1)
add_para(doc, "现实录音常受到环境噪声、设备噪声和混响的影响。语音去噪是语音识别、语音通信、助听设备和语音交互系统的重要前置环节。传统的谱减法和维纳滤波方法计算成本低，但通常依赖噪声统计假设，在非平稳噪声下容易产生音乐噪声或损伤语音细节。")
add_para(doc, "深度学习方法能够从带噪语音与纯净语音的配对样本中学习非线性映射。本实验选择直接作用于时域波形的 Wave U Net 结构，并在其瓶颈层加入 BiLSTM 与自注意力机制。该设计既保留 U 形网络的多尺度特征融合能力，又增强对跨时间语音上下文的建模能力。")

heading(doc, "2 相关原理与算法设计", 1)
heading(doc, "2.1 Wave U Net 编码器和解码器", 2)
add_para(doc, "Wave U Net 是将 U Net 结构从二维图像扩展到一维波形的网络。编码器使用步长为 2 的一维卷积逐层下采样，通道数依次为 32、64、128、256 和 512，从而形成由局部到全局的多尺度时域表示。解码器通过线性插值上采样、一维卷积和跳跃连接恢复时间分辨率。跳跃连接将编码器早期的细节特征传入对应解码层，可减轻下采样造成的语音瞬态信息丢失。")
heading(doc, "2.2 BiLSTM 与自注意力瓶颈层", 2)
add_para(doc, "在最深层特征上使用两层 BiLSTM。双向结构可同时利用当前时刻之前和之后的上下文，有利于区分持续性噪声和语音结构。随后使用 8 头多头自注意力计算序列内的全局依赖关系，使模型能够自适应关注与当前语音帧相关的远距离特征。注意力输出经过残差连接、层归一化和前馈网络后送入解码器。")
heading(doc, "2.3 残差预测与损失函数", 2)
add_para(doc, "模型不直接预测完整纯净波形，而是预测需要从带噪输入中修正的残差。最终输出为去噪波形等于带噪输入加预测残差。残差学习使模型在训练初期接近恒等映射，降低随机输出对语音的破坏。")
add_para(doc, "总损失定义为 L = -SI SDR + 0.1 × 多分辨率 STFT 损失。其中 SI SDR 衡量时域重建保真度；多分辨率 STFT 损失分别在 256、512 与 1024 点窗口下约束频谱幅度差异，使模型同时保留瞬态细节和相对稳定的谐波结构。")

heading(doc, "3 实验方案", 1)
heading(doc, "3.1 数据集与数据划分", 2)
add_para(doc, "训练数据来自 Edinburgh Noisy Speech Database。数据集提供带噪语音和对应纯净参考语音。项目将训练文件按说话人划分为训练集和验证集，以避免同一说话人泄漏到验证阶段。所有音频统一重采样为 16 kHz 单声道。训练时将训练和验证语音缓存为 2 秒、50% 重叠的波形片段；正式测试始终在 824 条原始完整 WAV 上进行。")
heading(doc, "3.2 训练设置", 2)
settings = doc.add_table(rows=1, cols=2)
settings.rows[0].cells[0].text = "项目"
settings.rows[0].cells[1].text = "设置"
for a, b in [
    ("采样率", "16 kHz"), ("训练片段", "2 秒，50% 重叠"),
    ("编码器通道", "32，64，128，256，512"), ("BiLSTM", "2 层，双向各 256 维"),
    ("自注意力", "8 头，dropout 为 0.1"), ("优化器", "AdamW，学习率 1e-3，权重衰减 1e-4"),
    ("学习率调度", "3 轮预热加余弦退火，最小学习率 1e-5"),
    ("最大训练轮数", "100"), ("早停策略", "验证集 SI SDR 连续 15 轮不提升"),
]:
    cells = settings.add_row().cells
    cells[0].text, cells[1].text = a, b
style_table(settings)
add_caption(doc, "表 1 主要训练配置")
heading(doc, "3.3 完整语音推理与后处理", 2)
add_para(doc, "任意长度测试语音在推理内部按 2 秒窗口送入模型，采用 50% 重叠相加方式重建完整波形。这一窗口化仅用于限制显存占用，最终结果和所有指标均以完整语音为单位。推理后使用 80 至 7500 Hz 带通滤波，并将输出峰值与输入峰值匹配。传统基线方法也采用相同后处理，以保证比较公平。")
heading(doc, "3.4 评价指标", 2)
add_bullet(doc, "SNR：输出信号与误差能量的全局比值，单位为 dB。")
add_bullet(doc, "SSNR：对有效 20 ms 帧求平均的分段信噪比，更能反映局部去噪情况。")
add_bullet(doc, "PESQ NB 与 PESQ WB：分别在窄带和宽带条件下评价感知语音质量。")
add_bullet(doc, "STOI：短时客观可懂度，范围为 0 至 1，数值越大越好。")
add_bullet(doc, "SI SDR：尺度不变信号失真比，衡量波形重建质量，单位为 dB。")

heading(doc, "4 实验结果与分析", 1)
add_para(doc, "本实验将模型与原始带噪语音、谱减法和维纳滤波进行比较。所有结果均来自 Edinburgh 官方测试集的 824 条完整语音。表 2 给出全体测试语音的平均指标。")
result = doc.add_table(rows=1, cols=7)
headers = ["方法", "SNR out", "SSNR out", "PESQ NB", "PESQ WB", "STOI", "SI SDR"]
for i, name in enumerate(headers):
    result.rows[0].cells[i].text = name
for row in [
    ("带噪输入", "8.45", "1.52", "2.945", "1.967", "0.921", "8.45"),
    ("谱减法", "15.49", "6.48", "3.105", "2.308", "0.921", "16.00"),
    ("维纳滤波", "15.60", "6.68", "3.131", "2.328", "0.920", "16.11"),
    ("Wave U Net BiLSTM Attention", "16.94", "8.49", "3.381", "2.542", "0.937", "17.70"),
]:
    cells = result.add_row().cells
    for i, value in enumerate(row):
        cells[i].text = value
style_table(result)
add_caption(doc, "表 2 824 条完整测试语音的平均客观指标")
add_para(doc, "与维纳滤波相比，所提出模型的 SNR 提升 1.34 dB，SSNR 提升 1.81 dB，PESQ WB 提升 0.214，STOI 提升 0.017，SI SDR 提升 1.59 dB。这说明模型不仅降低了整体误差，也改善了局部帧的噪声抑制、感知质量和可懂度。谱减法与维纳滤波的 STOI 接近带噪输入，而深度模型能够带来可懂度增益。")
add_figure(doc, PROJECT / "result" / "38ba233466f457f33deeb32cafd799f3.png", "图 1 Wave U Net 模型的完整语音分组评估结果", 15.5)
add_figure(doc, PROJECT / "result" / "a2372aacf60e0628915c69c3b507433b.png", "图 2 传统基线的完整语音评估结果", 15.5)
add_para(doc, "从按输入 SNR 分组的结果看，模型在低 SNR 组的去噪提升最突出。低 SNR 组中，输出 SNR 为 13.77 dB、SSNR 为 6.25 dB、SI SDR 为 15.53 dB。高 SNR 组的改进幅度相对较小，原因是原始带噪语音已经具有较高质量，指标存在一定上限。")
heading(doc, "4.1 结果讨论", 2)
add_para(doc, "Wave U Net 的跳跃连接有助于恢复高频细节，BiLSTM 提供双向语音上下文，自注意力加强远距离依赖建模，三者共同提升了模型在复杂噪声下的性能。尽管模型优于传统基线，结果仍受到训练数据分布、后处理策略和超参数设置影响。因此，与其他公开论文结果比较时，应保证数据划分、采样率、指标实现和后处理设置一致。")
heading(doc, "4.2 局限性与改进方向", 2)
add_bullet(doc, "当前模型在 Edinburgh 数据上训练和测试，跨数据集泛化能力需要通过 VCTK 加噪、NOIZEUS 等独立测试集进一步验证。")
add_bullet(doc, "训练损失主要优化 SI SDR 和频谱误差，未来可加入感知相关损失或多任务学习，以进一步提高 PESQ 与 STOI。")
add_bullet(doc, "目前采用单通道模型，后续可扩展到多麦克风场景并结合空间信息。")
add_bullet(doc, "推理后处理能够改善听感和输出响度，但报告中应始终说明其存在，以避免与无后处理结果混淆。")

heading(doc, "5 结论", 1)
add_para(doc, "本实验完成了基于 Wave U Net、BiLSTM 和自注意力机制的端到端语音去噪算法设计、训练和完整语音评估。模型在 Edinburgh 824 条完整测试语音上获得 16.94 dB 的输出 SNR、8.49 dB 的 SSNR、2.542 的 PESQ WB、0.937 的 STOI 和 17.70 dB 的 SI SDR，整体优于谱减法和维纳滤波。实验表明，多尺度时域建模与全局上下文建模的结合能够有效改善语音去噪性能。")

heading(doc, "参考文献", 1)
refs = [
    "[1] Macartney J, Weyde T. Improved Speech Enhancement with Wave U Net. arXiv preprint arXiv:1811.11307, 2018.",
    "[2] Valentini Botinhao L, Wang X, Takaki S, Yamagishi J. A study of data augmentation approaches for speech enhancement. Interspeech, 2016.",
    "[3] Loizou P C. Speech Enhancement Theory and Practice. 2nd ed. CRC Press, 2013.",
    "[4] ITU T Recommendation P.862. Perceptual evaluation of speech quality PESQ. International Telecommunication Union, 2001.",
    "[5] Taal C H, Hendriks R C, Heusdens R, Jensen J. An algorithm for intelligibility prediction of time frequency weighted noisy speech. IEEE Transactions on Audio Speech and Language Processing, 2011, 19(7): 2125 2136.",
    "[6] Le Roux J, Wisdom S, Erdogan H, Hershey J R. SDR half baked or well done. ICASSP, 2019.",
    "[7] Vaswani A, Shazeer N, Parmar N, et al. Attention Is All You Need. NeurIPS, 2017.",
    "[8] Schuster M, Paliwal K K. Bidirectional recurrent neural networks. IEEE Transactions on Signal Processing, 1997, 45(11): 2673 2681.",
]
for ref in refs:
    p = doc.add_paragraph()
    p.paragraph_format.first_line_indent = Cm(0)
    p.paragraph_format.hanging_indent = Cm(0.7)
    p.paragraph_format.line_spacing = 1.25
    p.paragraph_format.space_after = Pt(4)
    r = p.add_run(ref)
    set_font(r, size=9.5)

doc.core_properties.title = "基于 Wave U Net 的语音去噪算法设计与实验分析"
doc.core_properties.author = ""
doc.save(OUTPUT)
print(OUTPUT)
