"""模式 1：顺序工作流（Prompt Chaining）

来自 LangGraph 官方文档：
> Prompt chaining decomposes a task into a sequence of steps,
> where each LLM call processes the output of the previous one.

核心思想：
- 将复杂任务分解为一系列简单步骤
- 每个步骤的输出是下一步的输入
- 类似于流水线或管道模式

适用场景：
- 内容生成管道（大纲 → 草稿 → 修改 → 定稿）
- 数据处理流水线（提取 → 转换 → 加载）
- 多步骤推理（分析 → 推理 → 结论）

图结构：
    START
      ↓
  generate_outline（生成大纲）
      ↓
  generate_characters（生成角色）
      ↓
  write_story（撰写故事）
      ↓
  generate_title（生成标题）
      ↓
    END
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from tutorials.models import StoryState

# ============================================================================
# 辅助函数（模拟 LLM 调用）
# ============================================================================


def _simulate_llm(prompt: str) -> str:
    """模拟 LLM 调用。

    实际项目中应替换为：
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(model="gpt-4")
        response = llm.invoke(prompt)
        return response.content
    """
    # 这里用简单的字符串拼接模拟 LLM 响应
    return f"[模拟 LLM 响应] 基于提示词 '{prompt[:30]}...' 生成的内容"


# ============================================================================
# 节点实现
# ============================================================================


def generate_outline(state: StoryState) -> dict:
    """节点 1：生成故事大纲。

    这是顺序工作流的第一步，负责：
    1. 接收用户输入的主题
    2. 生成结构化的故事大纲
    3. 将大纲传递给下一步

    输入：StoryState（包含 topic）
    输出：状态更新（outline 字段）

    设计要点：
    - 节点是纯函数：接收状态，返回更新
    - 提示词模板在节点内部组装（不在状态中）
    - 模拟 LLM 调用，实际项目替换为真实调用
    """
    # 构造提示词
    prompt = f"""
    请为以下主题生成一个简短的故事大纲（3-5 个要点）：
    主题：{state["topic"]}

    大纲格式：
    1. 开头：...
    2. 发展：...
    3. 高潮：...
    4. 结局：...
    """

    # 调用 LLM（模拟）
    outline = _simulate_llm(prompt)

    # 返回状态更新
    return {
        "outline": outline,
        "messages": [f"[generate_outline] 已生成大纲，长度 {len(outline)} 字符"],
    }


def generate_characters(state: StoryState) -> dict:
    """节点 2：生成角色设定。

    顺序工作流的第二步，依赖第一步的输出：
    - 读取 state['outline']（第一步生成的大纲）
    - 基于大纲生成角色设定

    这展示了顺序工作流的关键特性：
    - 每个节点可以访问之前所有节点的输出
    - 节点之间通过状态传递数据
    """
    # 构造提示词（使用第一步的输出）
    prompt = f"""
    基于以下故事大纲，创建 2-3 个主要角色：

    大纲：{state.get("outline", "无大纲")}

    每个角色需要：
    - 名字
    - 性格特点
    - 在故事中的作用
    """

    # 调用 LLM（模拟）
    characters = _simulate_llm(prompt)

    # 返回状态更新（追加到 messages 列表）
    return {
        "characters": characters,
        "messages": [
            f"[generate_characters] 已生成角色设定，长度 {len(characters)} 字符"
        ],
    }


def write_story(state: StoryState) -> dict:
    """节点 3：撰写完整故事。

    顺序工作流的第三步，综合前两步的输出：
    - 读取 state['outline']（大纲）
    - 读取 state['characters']（角色设定）
    - 生成完整故事

    这展示了顺序工作流的"汇聚"特性：
    - 后续步骤可以使用所有前置步骤的输出
    - 逐步构建复杂内容
    """
    # 构造提示词（综合前两步的输出）
    prompt = f"""
    基于以下信息撰写一个简短的故事（200-300 字）：

    大纲：{state.get("outline", "无大纲")}
    角色：{state.get("characters", "无角色设定")}

    要求：
    - 故事完整，有开头、发展、高潮、结局
    - 角色性格鲜明
    - 语言生动有趣
    """

    # 调用 LLM（模拟）
    story = _simulate_llm(prompt)

    # 返回状态更新
    return {
        "story": story,
        "messages": [f"[write_story] 已生成故事，长度 {len(story)} 字符"],
    }


def generate_title(state: StoryState) -> dict:
    """节点 4：生成故事标题。

    顺序工作流的最后一步，基于完整故事生成标题：
    - 读取 state['story']（完整故事）
    - 生成吸引人的标题

    这展示了顺序工作流的"收尾"特性：
    - 最后一步通常是对前面所有工作的总结或包装
    - 输出是整个工作流的最终成果
    """
    # 构造提示词
    prompt = f"""
    为以下故事生成一个吸引人的标题（10 字以内）：

    故事：{state.get("story", "无故事")}

    要求：
    - 简洁有力
    - 能概括故事主题
    - 吸引读者点击
    """

    # 调用 LLM（模拟）
    title = _simulate_llm(prompt)

    # 返回状态更新
    return {
        "title": title,
        "messages": [f"[generate_title] 已生成标题：{title}"],
    }


# ============================================================================
# 图构建
# ============================================================================


def build_prompt_chaining_graph() -> StateGraph:
    """构建并返回顺序工作流图。

    构建步骤：
    1. 创建 StateGraph，指定状态类型
    2. 添加所有节点
    3. 添加边，定义执行顺序
    4. 返回未编译的图（由调用者编译）

    关于边的类型：
    - add_edge(A, B)：静态边，表示"总是从 A 到 B"
    - add_conditional_edges()：条件边，根据状态动态选择
    - 本示例只使用静态边，因为顺序是固定的
    """
    # 步骤 1：创建 StateGraph
    workflow = StateGraph(StoryState)

    # 步骤 2：添加节点
    workflow.add_node("generate_outline", generate_outline)
    workflow.add_node("generate_characters", generate_characters)
    # 内联节点：直接传入函数（不需要单独定义节点名）
    workflow.add_node("write_story", write_story)
    workflow.add_node("generate_title", generate_title)

    # 步骤 3：添加边（定义执行顺序）
    # START → generate_outline：图从生成大纲开始
    workflow.add_edge(START, "generate_outline")

    # generate_outline → generate_characters：大纲生成后生成角色
    workflow.add_edge("generate_outline", "generate_characters")

    # generate_characters → write_story：角色生成后撰写故事
    workflow.add_edge("generate_characters", "write_story")

    # write_story → generate_title：故事撰写后生成标题
    workflow.add_edge("write_story", "generate_title")

    # generate_title → END：标题生成后结束
    workflow.add_edge("generate_title", END)

    return workflow


# ============================================================================
# 导出编译后的图
# ============================================================================

# 构建并编译图
graph = build_prompt_chaining_graph().compile()
