from __future__ import annotations

from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
from xml.sax.saxutils import escape
from datetime import datetime, timezone

OUT = Path('/media/volume/dataset/xma8/work/icaif_ecc_news_attention/docs/project_stage_report_20260323_cn.docx')

SECTIONS = [
    ('title', 'ICAIF 项目阶段性报告'),
    ('subtitle', '从 noisy timing 到可信的 after-hours ECC 增量信号'),
    ('meta', 'FT5005 / ICAIF 仓库阶段总结｜2026年3月23日'),
    ('heading', '摘要'),
    ('body', '这份阶段性报告的目标，不是把故事讲得很大，而是把当前真正站得住的东西讲清楚。项目已经从早期“多模态都加一点看看”的发散状态，逐步收束成一条更干净的主线：在 noisy timing 的财报电话会议场景里，我们把任务改写为 shock_minus_pre 的 after-hours 波动冲击预测；在强 market prior 之上，clean after_hours 子样本中的 A4 observability/alignment 与紧凑 Q&A semantics 提供了目前最可信的增量信号。跨 ticker 的 transfer 结果则提醒我们，这些局部增量信息不是任何时候都该相信，更稳妥的做法是 reliability-aware abstention。最近一轮探索还发现，最难的 analyst question 本身带有一个小而真实的局部信号，但它更像一个局部探针，而不是已经成熟的主方法。整体上，这个项目现在更像是在迷雾里开车：我们还没有完美 GPS，但已经找到了几条真正可靠的路标。'),
    ('heading', '1. 研究背景：我们到底在解决什么问题？'),
    ('body', '财报电话会议（earnings conference call, ECC）是一场高密度的公开问答：管理层先给主叙事，分析师再围绕风险、需求、指引和执行细节不断追问。市场确实会对这类信息做出反应，但真正麻烦的地方在于时间并不干净。共享数据里的 A2 往往只告诉我们计划中的开会时间，而 A4 虽然提供了句子级时间戳，却是有噪声的、不完整的。换句话说，我们不是在光线充足的实验室里建模，而更像是在隔着一层雾去听一场对话，再猜市场什么时候真正听到了什么。'),
    ('body', '这也是为什么项目后来逐步放弃了“把所有模态都堆上去，也许会更强”的思路。因为如果任务定义本身不够诚实、基线不够强，模型再复杂，也很容易只是记住哪个公司一贯更波动，而不是学到会议本身的新信息。于是我们做了两个关键收缩：第一，把目标从原始波动水平改写成 shock_minus_pre，也就是会后相对于会前基线多出来的那部分冲击；第二，把主要战场收缩到 off-hours，尤其是更干净的 after_hours 子样本。'),
    ('body', '直觉上，这就像先把地形、天气和车况这些“本来就存在的东西”扣掉，再去看这次驾驶动作本身到底带来了什么额外偏移。对我们来说，那些“本来就存在的东西”主要就是市场先验、公司固有波动风格，以及会前已经被消化的信息。'),
    ('heading', '2. 当前故事线：核心 hypothesis 与要回答的问题'),
    ('body', '到目前为止，项目主线已经可以收束成四个问题。第一，什么样的目标才更诚实？我们的假设是 shock_minus_pre 比直接预测原始 post-call volatility level 更适合作为主任务，因为它更接近“会议带来的增量信息”。第二，真正可信的 ECC 增量信号到底在哪里？目前证据更支持 A4 这种带噪但有价值的 observability/alignment 线索，加上一套紧凑的 Q&A semantic block。第三，跨公司 transfer 时，文本信号该不该被无条件相信？现在的答案偏向“不要”，也就是只有在可靠性足够高时才放行局部语义修正。第四，局部最难的问题里，会不会藏着比整段对话更尖锐的信号？这是最近的 exploratory 支线。'),
    ('heading', '3. 具体实验方法：我们现在是怎么做的？'),
    ('body', '当前主实验仍然基于 DJ30 试点样本。仓库里的 baseline-ready panel 目前覆盖 553 个事件，并且完成了 A1（问答对）、A2（计划时间）、A4（带噪句子时间戳）、音频文件和高频市场数据的 join。对于主线 fixed-split 结果，我们重点使用 clean after_hours 子样本；对应的严格 split 大小为 train 89 / val 23 / test 60。对 transfer 分析，我们进一步使用跨时间窗口的 held-out 评估，并保留 matched sample 作为更严格的比较环境。'),
    ('body', '方法上，我们尽量保持“先把问题讲清楚，再把模型加复杂”的节奏，因此主干模型仍然是相对克制的 residual ridge 家族。最重要的步骤包括：先用 pre-call market prior 建强基线，再加入 controls 和 A4 observability/alignment，然后测试低维 Q&A 语义表示是否还能带来增量。transfer 侧则利用 agreement / disagreement 的结构，决定什么时候信任局部语义修正，什么时候让 market baseline 接管。'),
    ('body', '如果打个不太严肃但很贴切的比方，这个框架像两层装置：第一层是地基，决定房子先别塌；第二层是传感器，告诉我们屋里哪里真的在震。A4 更像是雾里的路标，Q&A 语义更像是对话里真正让空气变紧的那一刻。只有当地基稳，传感器读数才值得相信。'),
    ('heading', '4. 初步结果与分析'),
    ('subheading', '4.1 主线 fixed-split：可信的 ECC 增量是窄而干净的'),
    ('body', '先看更广义的 corrected off-hours benchmark。结果显示，prior-only 基线在 clean off-hours 上只有 R^2 ≈ 0.198，而结构化 residual 主线可以达到 R^2 ≈ 0.913。这说明任务改写是值得的：模型并不只是原地记忆，而是在解释剩余冲击。'),
    ('body', '如果往论文最安全的主线收缩，真正最值得强调的是 clean after_hours 的 ladder。当前关键结果包括：pre-call market only 的 test R^2 ≈ 0.9174；pre-call market + controls ≈ 0.9194；pre-call market + A4 + compact Q&A semantics ≈ 0.9271；pre-call market + controls + A4 + compact Q&A semantics ≈ 0.9347。'),
    ('body', '这个结果最重要的地方不只是分数更高了一点，而是它告诉我们：当前最可信的增量价值并不在重型 sequence 建模，也不在泛泛的多模态堆叠，而是在带噪但有用的时间可观测性（A4）和紧凑问答语义的结合。简单说，真正有用的不是把整场会议都榨成一个大而全的向量，而是先确认“我们大致知道市场听到了哪一段”，再去看那一段对话到底在讲什么。'),
    ('subheading', '4.2 transfer：关键不是更花哨，而是更知道什么时候闭嘴'),
    ('body', '跨 ticker 的 transfer 结果给了一个很重要的提醒：局部语义信号不是任何时候都该被信任。我们后来逐步收束出的最稳方法，不是更复杂的 router 家族，而是一个非常克制的策略：reliability-aware abstention。也就是说，当 retained views 一致时，我们允许局部修正介入；当它们彼此打架时，就让 pre_call_market_only 这条更稳的主干接管。'),
    ('body', '在 pooled temporal transfer benchmark 上，这条路线目前达到 R^2 ≈ 0.99792，略好于 retained semantic/audio backbone（约 0.99788）和 validation-selected expert（约 0.99788）。这件事听起来很保守，但其实很科研：不是所有增量信息都值得强行用上，什么时候不该出手，本身就是方法的一部分。'),
    ('subheading', '4.3 hardest-question 支线：单个最难问题可能比整段对话更尖'),
    ('body', '最近最有意思的一条 exploratory 发现，是 hardest-question 这条线。我们不再把整段问答一起建模，而是只盯住局部最难的 analyst question。当前最强的 local exploratory route 是 non-structural hardest-question LSA(4) bi-gram，其 latest held-out window 表现约为 R^2 ≈ 0.99865，略高于 hard abstention 的 R^2 ≈ 0.99864，对应 paired p(MSE) ≈ 0.044。'),
    ('body', '这个提升还不大，所以不能把它夸成主方法胜利，但它已经足够说明：整场对话里最尖锐的那一个问题，有时比整段对话更有信号密度。更有意思的是，这条线后来越来越像一个 question-centric、non-structural 的局部 framing 信号：把 structural / strategic probe 词汇 mask 掉以后，最强结果几乎一模一样；加 hardest answer 或 top-1 Q&A pair 视角，反而变差；用小型 supervised text subspace（如 PLS）去“学得更聪明”，也没有超过当前的无监督紧凑表示。'),
    ('body', '这些结果合在一起，给出的图景很生动：真正有用的东西，不太像“会议里提到了哪个大主题”，而更像“分析师是怎么提那个问题的”。很多正向 pocket 都更像是在要求管理层帮忙把模型搭清楚、把机制说透、把数字讲明白，而不是泛泛聊战略大词。'),
    ('heading', '5. 现阶段 contribution：我们已经可以比较稳地说什么？'),
    ('body', '如果把当前贡献压缩一下，我认为至少可以较稳地说出三层结论。第一层是问题重写和评估纪律：ECC 研究不能只追逐更复杂的多模态模型，而必须先把任务定义和基线设稳。把目标改写为 shock_minus_pre，并在强 market prior 之上做 residual 评估，是项目的一条基础贡献。'),
    ('body', '第二层是可信的增量信号是狭窄而结构化的。当前最可靠的 fixed-split 主线是 clean after_hours 中的 A4 加 compact Q&A semantics。它的价值在于“窄而真”，而不是“全而大”。第三层是 transfer 需要可靠性意识。跨公司 transfer 不是把一个语义模块硬迁过去就结束了，当前最稳的方法点是 reliability-aware abstention：当局部修正的可靠性不足时，允许模型后退一步。'),
    ('body', '此外，hardest-question 这条线虽然还处于 exploratory 阶段，但已经贡献了一个很有意思的研究判断：局部 analyst question framing 可能是比全局问答融合更尖锐的 transfer signal。'),
    ('heading', '6. 结论与 limitations'),
    ('body', '从阶段性进度来看，项目已经明显脱离了“什么都试一点”的早期状态，开始形成一个可以自洽的研究故事：主线固定在 noisy timing 下的 after-hours volatility shock prediction；最可信的 ECC 增量来自 A4 与紧凑 Q&A 语义；transfer 侧最稳的方法不是更复杂，而是更知道何时 abstain；hardest-question 发现则提供了一个新的局部研究窗口。'),
    ('body', '但局限也非常清楚。第一，当前仍然是 DJ30 pilot 规模，很多结果虽然方向明确，但样本还不够大。第二，transfer 侧最强路线虽然 coherent，却还称不上压倒性胜利，很多提升幅度依然很小。第三，hardest-question 这条新线虽然很有意思，但它现在更像一个被确认存在的局部机制，还不是已经成熟到可以直接上升为主方法的模块。第四，音频、重型 sequence、以及更花哨的 routing 家族，目前都没有形成比主线更强、更干净的贡献，不适合抢故事中心。'),
    ('body', '总的来说，这个项目现阶段最像是在一条山路上逐渐找到真正的路标。我们还没有到山顶，也还不能说每一步都已经万无一失；但至少已经知道，哪些灯光是远处的村庄，哪些只是雾里的反光。这个判断本身，就比盲目继续堆方法更重要。'),
]


def run_props(size=22, bold=False, italic=False, center=False):
    rpr = [
        '<w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" w:eastAsia="SimSun"/>' ,
        f'<w:sz w:val="{size}"/>',
        f'<w:szCs w:val="{size}"/>'
    ]
    if bold:
        rpr.append('<w:b/>')
        rpr.append('<w:bCs/>')
    if italic:
        rpr.append('<w:i/>')
        rpr.append('<w:iCs/>')
    return ''.join(rpr)


def paragraph(text: str, *, size=22, bold=False, italic=False, center=False, space_after=120) -> str:
    t = escape(text)
    ppr = [f'<w:spacing w:after="{space_after}"/>']
    if center:
        ppr.append('<w:jc w:val="center"/>')
    return (
        '<w:p>'
        f'<w:pPr>{"".join(ppr)}</w:pPr>'
        '<w:r>'
        f'<w:rPr>{run_props(size=size, bold=bold, italic=italic, center=center)}</w:rPr>'
        f'<w:t xml:space="preserve">{t}</w:t>'
        '</w:r>'
        '</w:p>'
    )

parts = []
for kind, text in SECTIONS:
    if kind == 'title':
        parts.append(paragraph(text, size=34, bold=True, center=True, space_after=80))
    elif kind == 'subtitle':
        parts.append(paragraph(text, size=26, bold=False, center=True, space_after=80))
    elif kind == 'meta':
        parts.append(paragraph(text, size=18, italic=True, center=True, space_after=220))
    elif kind == 'heading':
        parts.append(paragraph(text, size=26, bold=True, space_after=100))
    elif kind == 'subheading':
        parts.append(paragraph(text, size=22, bold=True, space_after=80))
    else:
        parts.append(paragraph(text, size=22, space_after=120))

body_xml = ''.join(parts) + (
    '<w:sectPr>'
    '<w:pgSz w:w="11906" w:h="16838"/>'
    '<w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="1440" w:header="720" w:footer="720" w:gutter="0"/>'
    '</w:sectPr>'
)

document_xml = f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:wpc="http://schemas.microsoft.com/office/word/2010/wordprocessingCanvas"
 xmlns:mc="http://schemas.openxmlformats.org/markup-compatibility/2006"
 xmlns:o="urn:schemas-microsoft-com:office:office"
 xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
 xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math"
 xmlns:v="urn:schemas-microsoft-com:vml"
 xmlns:wp14="http://schemas.microsoft.com/office/word/2010/wordprocessingDrawing"
 xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"
 xmlns:w10="urn:schemas-microsoft-com:office:word"
 xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"
 xmlns:w14="http://schemas.microsoft.com/office/word/2010/wordml"
 xmlns:wpg="http://schemas.microsoft.com/office/word/2010/wordprocessingGroup"
 xmlns:wpi="http://schemas.microsoft.com/office/word/2010/wordprocessingInk"
 xmlns:wne="http://schemas.microsoft.com/office/word/2006/wordml"
 xmlns:wps="http://schemas.microsoft.com/office/word/2010/wordprocessingShape"
 mc:Ignorable="w14 wp14">
  <w:body>{body_xml}</w:body>
</w:document>'''

content_types = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
  <Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
  <Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>
</Types>'''

rels = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>
  <Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>
</Relationships>'''

now = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
core = f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties"
 xmlns:dc="http://purl.org/dc/elements/1.1/"
 xmlns:dcterms="http://purl.org/dc/terms/"
 xmlns:dcmitype="http://purl.org/dc/dcmitype/"
 xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <dc:title>ICAIF 项目阶段性报告</dc:title>
  <dc:subject>Chinese Word stage report</dc:subject>
  <dc:creator>OpenAI Codex</dc:creator>
  <cp:lastModifiedBy>OpenAI Codex</cp:lastModifiedBy>
  <dcterms:created xsi:type="dcterms:W3CDTF">{now}</dcterms:created>
  <dcterms:modified xsi:type="dcterms:W3CDTF">{now}</dcterms:modified>
</cp:coreProperties>'''

app = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties"
 xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">
  <Application>Microsoft Office Word</Application>
  <DocSecurity>0</DocSecurity>
  <ScaleCrop>false</ScaleCrop>
  <SharedDoc>false</SharedDoc>
  <HyperlinksChanged>false</HyperlinksChanged>
  <AppVersion>16.0000</AppVersion>
</Properties>'''

OUT.parent.mkdir(parents=True, exist_ok=True)
with ZipFile(OUT, 'w', ZIP_DEFLATED) as zf:
    zf.writestr('[Content_Types].xml', content_types)
    zf.writestr('_rels/.rels', rels)
    zf.writestr('docProps/core.xml', core)
    zf.writestr('docProps/app.xml', app)
    zf.writestr('word/document.xml', document_xml)

print(f'Wrote {OUT}')
