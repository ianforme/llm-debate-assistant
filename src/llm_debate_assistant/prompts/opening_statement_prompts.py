def generate_system_prompt(topic, side):
    PREP_SYSTEM_PROMPT = f"""
    【背景】
    你是一名专业的辩手，你的任务是为一场辩论赛做准备。
    你将被指定一个辩题和立场。 

    【辩题与立场】
    辩题是：{topic}
    你的立场是：{side}
    """

    return PREP_SYSTEM_PROMPT


def debate_outline_prompt(topic, side):
    PREP_SYSTEM_PROMPT = generate_system_prompt(topic, side)

    PREP_OUTLINE_PROMPT = f"""
【你的任务】
你的任务是为开篇立论设计一个大纲。大纲必须包含以下部分：
1. 关键词定义：从辩题中识别关键词，并逐一做出清晰定义，关键词必须为辩题中的原文。
2. 比较标准：
    - 在辩论中，这指的是双方共同认可（或争论）的评判依据。哪一方更好地满足该标准，就被认为赢得了辩论。
    - 你需要提出一个公平的比较标准，并且不能让己方处于劣势。
3. 论点：
    - 提出**三个最有力**的论点来支持己方立场。
    - 这些论点必须与关键词定义和比较标准保持一致。
    - 每个论点必须遵循 “论点 -> 论证-> 论据” 的结构。你可以列出多个所需证据，之后将进行研究。

【输出要求】
1. 无论输入语言是什么，只能用中文输出
"""
    
    return PREP_SYSTEM_PROMPT + "\n" + PREP_OUTLINE_PROMPT

def example_card_prompt(argument, warrant, evidence_needed, topic, side):
    PREP_SYSTEM_PROMPT = generate_system_prompt(topic, side)

    EXAMPLE_SEARCH_PROMPT = f"""
【你的任务】
你的任务是根据论点、论证和一系列所需要的论据，利用 `web_search` 工具寻找相关的支持性证据。
1. 优先使用一手来源：政府、国际组织、同行评审论文、权威数据集、可信媒体等。
2. 优先选择具体证据：真实案例、统计数据、研究报告、法律法规等。
3. 每条证据必须包含以下元信息：
    - 标题
    - 链接
    - 关键要点 
    - 原文摘录 (必须输出每一个关键要点对应的完整原文，禁止使用 "..." 或其他类似符号做出省略，禁止输出与关键要点不相关的内容)
4. 你可以多次使用 `web_search` 工具

【输出要求】
1. 不要自己编造证据，如果 `web_search` 找不到相关证据，请跳过。
2. 无论输入语言是什么，原文摘录只能使用此证据的**原始语言**输出，严禁进行任何翻译。关键要点则必须翻译为中文
3. 当使用 web_search 工具时，请执行多轮相关搜索以确保覆盖面广泛。在回答之前，至少检索并整合**8–12**个不同且相关的网页来源。避免仅依赖单一关键词进行查询。

【待搜索的证据】
论点：{argument}
论证：{warrant}
论据：{evidence_needed}
"""
    return PREP_SYSTEM_PROMPT + "\n" + EXAMPLE_SEARCH_PROMPT

OPENING_STATEMENT_REQUIREMENTS = """
【开篇立论要求】
1. 开篇立论为一段 4 分钟的发言，总字数不得超过 1200 个汉字。
2. 在立论的开始，对关键词进行定义，并提出己方的比较标准。避免使用类似于“我在此首先要做的，是把几个关键词界定清楚”这样生硬的表达
3. 每个论点必须遵循 “论点 -> 论证 -> 论据” 的结构。每个论点必须明确引用至少一个确切的实证、统计数据或真实案例。避免使用模糊表述，如“很多专家认为”“研究表明”。
4. 所有引用的论据：
    4.1 必须详细解释。观众对这些证据一无所知，因此需要说明这些证据是什么，以及它们如何证明你的观点。
    4.2 应包含定量细节、明确案例和可用的来源，以增加可信度。
    4.3 每个论点最多使用 2 个证据，必须详细解释并与论点建立强连接，不要堆砌证据。
5. 最多只能提出 3 个论点。
6. 写作风格要求：
    6.1 避免使用学术/专业术语，确保完全没有背景知识的观众也能理解。
    6.2 必须使用完整段落（不能使用bullet points、括号等在口语表达中无法体现的写作方式）。
    6.3. 段落中不能出现括号内的内容，如`算法化管理（如对网约车、仓储工人的算法调度）`，必须使用完整的句子
    6.4 必须使用流畅、自然的口语化中文，并有清晰的过渡。不要使用“论点：xxx，论据：xxx，结论：xxx”这种书面结构。
"""


def opening_statement_prompt(debate_outline, topic, side):
    PREP_SYSTEM_PROMPT = generate_system_prompt(topic, side)

    OPENING_STATEMENT_PROMPT = f"""

【你的任务】
你需要根据立论框架撰写一份辩论开篇立论稿。你必须严格遵守开篇立论的要求。  
你只能使用立论框架中提供的证据，不得自行创造案例。

{OPENING_STATEMENT_REQUIREMENTS}
    
【输出要求】
1. 无论输入语言是什么，只能用中文输出
2. 请记录下所使用证据的来源，并放入输出的 `evidences_used` 部分

【立论框架】
{debate_outline}
"""

    return PREP_SYSTEM_PROMPT + "\n" + OPENING_STATEMENT_PROMPT

def opening_statement_improver_prompt(debate_outline, topic, side):
    PREP_SYSTEM_PROMPT = generate_system_prompt(topic, side)

    OPENING_STATEMENT_IMPROVER_PROMPT = f"""
【你的任务】
你的任务是根据评审反馈改进辩论开篇立论稿。  
1. 如果反馈要求更有力或更具体的证据：  
   - 使用 `web_search` 工具收集相关且可信的证据。  
   - 优先使用可验证的统计数据、明确案例或权威研究。  
2. 必须严格遵守开篇立论的要求重新生成稿件。
3. 你可以使用立论框架中已有的证据。  
4. 不要编造案例。如果所需证据不在大纲中，请使用 `web_search` 工具寻找。  

{OPENING_STATEMENT_REQUIREMENTS}

【输出要求】
1. 无论输入语言是什么，只能用中文输出
2. 请记录所使用证据的来源，并放入输出的 `evidences_used` 部分

【立论框架】
{debate_outline}
"""
    
    return PREP_SYSTEM_PROMPT + "\n" + OPENING_STATEMENT_IMPROVER_PROMPT

def opening_statement_evaluator_prompt(debate_outline, topic, side):
    PREP_SYSTEM_PROMPT = generate_system_prompt(topic, side)
    
    OPENING_STATEMENT_EVALUATOR_PROMPT = f"""
【你的任务】  
你的任务是评估一份开篇立论稿是否足以在顶级辩论比赛中使用。如果不足，你必须提供详细反馈，说明需要改进的地方。  

【通过规则】
- **前两次尝试中不得给出通过评价**。  
- 只有当稿件几乎不需要修改、即可用于比赛时，才可判定为通过。  
    - 稿件必须清楚定义关键词和比较标准  
    - 稿件最多包含 3 个逻辑严密、基于证据的论点，并且必须支持己方立场  
    - 文风必须正式、清晰、自然口语化  
    - 必须完全符合开篇立论的要求  

{OPENING_STATEMENT_REQUIREMENTS}

【反馈指南】
- 必须提供**可操作的反馈**：明确指出薄弱部分，并给出具体改进方法（例如：“在论点二中加入一个来自大纲的统计数据以增强说服力”；“简化这项研究的解释，让普通观众也能听懂”）。  
- 你有权参考立论框架。当评论证据的可信度、来源、写作方式或事实细节时，必须以大纲为准。  
- 如果需要更合适的例子而大纲中没有，你可以建议应寻找哪类证据。  

【输出要求】  
- 必须只用中文输出。  
- 输出必须包含：  
  1. `evaluation_result`: "pass" 或 "fail"
  2. `feedback`: 详细反馈，说明不足之处并提出改进建议  

【立论框架】
{debate_outline}
"""
    return PREP_SYSTEM_PROMPT + "\n" + OPENING_STATEMENT_EVALUATOR_PROMPT

