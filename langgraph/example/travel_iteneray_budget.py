from typing import TypedDict, List, Dict, Any, Annotated
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

# 定义状态
class State(TypedDict):
    # 用户输入
    user_preferences: Dict[str, Any]
    # 系统生成
    recommended_destinations: List[str]
    selected_destination: str
    initial_budget: float
    adjusted_budget: float
    itinerary: Dict[str, Any]
    # 控制流
    budget_approved: bool
    itinerary_approved: bool
    # 消息历史
    messages: Annotated[List[Dict], add_messages]

# 节点函数
def input_user_preferences(state: State) -> State:
    """1. 输入用户偏好"""
    print("=== 步骤1: 输入用户偏好 ===")

    # 模拟用户输入 (实际应用中从UI获取)
    preferences = {
        "travel_type": "休闲度假",  # 休闲度假/冒险/文化体验
        "travel_dates": "2024-10-01 至 2024-10-07",
        "travelers": 2,
        "interests": ["海滩", "美食", "购物"],
        "budget_limit": 10000,  # 预算上限
        "preferred_activities": ["spa", "水上活动", "城市观光"]
    }

    state["user_preferences"] = preferences
    print(f"已记录用户偏好: {preferences}")
    return state

def recommend_destinations(state: State) -> State:
    """2. 推荐目的地"""
    print("\n=== 步骤2: 推荐目的地 ===")

    preferences = state["user_preferences"]

    # 基于偏好的简单推荐逻辑
    if "海滩" in preferences["interests"] and "休闲度假" in preferences["travel_type"]:
        destinations = ["巴厘岛", "马尔代夫", "普吉岛"]
    elif "文化体验" in preferences["interests"]:
        destinations = ["京都", "罗马", "开罗"]
    else:
        destinations = ["东京", "新加坡", "悉尼"]

    state["recommended_destinations"] = destinations
    print(f"推荐的目的地: {destinations}")

    # 简单选择第一个推荐 (实际中让用户选择)
    state["selected_destination"] = destinations[0]
    print(f"自动选择目的地: {destinations[0]}")

    return state

def check_budget(state: State) -> State:
    """检查预算是否超支"""
    print("\n=== 检查预算 ===")

    preferences = state["user_preferences"]
    initial_budget = state["initial_budget"]

    if initial_budget <= preferences["budget_limit"]:
        state["budget_approved"] = True
        print("✅ 预算在范围内，无需调整")
    else:
        state["budget_approved"] = False
        print(f"⚠️ 预算超支！需要调整")
        print(f" 当前预算: ¥{initial_budget}")
        print(f" 预算上限: ¥{preferences['budget_limit']}")

    return state

def generate_itinerary(state: State) -> State:
    """4. 生成行程"""
    print("\n=== 步骤4: 生成行程 ===")

    destination = state["selected_destination"]
    budget = state["adjusted_budget"]

    # 生成简单的行程
    itinerary = {
        "destination": destination,
        "duration": "7天6晚",
        "budget": budget,
        "daily_plan": [
            {"day": 1, "activities": ["抵达", "酒店入住", "市区观光"]},
            {"day": 2, "activities": ["海滩活动", "SPA体验"]},
            {"day": 3, "activities": ["水上活动", "海鲜晚餐"]},
            {"day": 4, "activities": ["购物", "文化体验"]},
            {"day": 5, "activities": ["一日游", "当地美食"]},
            {"day": 6, "activities": ["自由活动", "告别晚餐"]},
            {"day": 7, "activities": ["早餐", "前往机场"]}
        ],
        "hotel": f"{destination}豪华度假村",
        "transportation": "经济舱机票 + 当地交通"
    }

    state["itinerary"] = itinerary
    print(f"已生成{destination}的行程")
    print(f"总预算: ¥{budget}")

    return state

def calculate_initial_budget(state: State) -> State:
    """3. 计算初始预算"""
    print("\n=== 步骤3: 计算初始预算 ===")

    destination = state["selected_destination"]
    preferences = state["user_preferences"]
    travelers = preferences["travelers"]

    # 基于目的地的基础费用
    destination_costs = {
        "巴厘岛": 4000,
        "马尔代夫": 8000,
        "普吉岛": 3500,
        "京都": 5000,
        "罗马": 6000,
        "开罗": 3000,
        "东京": 5500,
        "新加坡": 4500,
        "悉尼": 7000,
    }

    base_cost = destination_costs.get(destination, 5000)
    total_budget = base_cost * travelers

    state["initial_budget"] = total_budget
    state["adjusted_budget"] = total_budget
    print(f"目的地: {destination}")
    print(f"基础费用: ¥{base_cost}/人 × {travelers}人 = ¥{total_budget}")

    return state

def adjust_plan(state: State) -> State:
    """调整计划以降低预算"""
    print("\n=== 调整计划 ===")

    current_budget = state.get("adjusted_budget", state["initial_budget"])

    # 每次削减10%
    adjusted = current_budget * 0.9
    state["adjusted_budget"] = adjusted
    print(f"调整预算: ¥{current_budget} → ¥{adjusted}")

    return state

def recalculate_budget(state: State) -> State:
    """重新计算预算并检查是否在范围内"""
    print("\n=== 重新计算预算 ===")

    preferences = state["user_preferences"]
    adjusted_budget = state["adjusted_budget"]

    if adjusted_budget <= preferences["budget_limit"]:
        state["budget_approved"] = True
        print(f"✅ 调整后预算 ¥{adjusted_budget} 在范围内")
    else:
        state["budget_approved"] = False
        print(f"⚠️ 调整后预算 ¥{adjusted_budget} 仍超支，继续调整")

    return state

def user_modification(state: State) -> State:
    """用户修改行程"""
    print("\n=== 用户修改 ===")
    print("根据用户反馈调整行程...")

    # 模拟用户修改：微调预算
    state["adjusted_budget"] = state["adjusted_budget"] * 0.95
    print(f"调整后预算: ¥{state['adjusted_budget']}")

    return state

def human_review(state: State) -> State:
    """人工审核（模拟）"""
    print("\n=== 人工审核 ===")
    print("行程已提交审核...")

    # 模拟审核结果（实际中由人工操作）
    # 80%概率通过审核
    import random
    approved = random.random() > 0.2

    if approved:
        print("✅ 审核通过")
        state["itinerary_approved"] = True
    else:
        print("⚠️ 审核未通过，需要修改")
        state["itinerary_approved"] = False

    return state


def wait_user_confirmation(state: State) -> State:  # 1 usage
    """等待用户确认（模拟）"""
    print("\n=== 等待用户确认 ===")
    print("行程已发送给用户...")

    # 模拟用户确认（实际中等待用户输入）
    # 70%概率用户确认
    import random
    confirmed = random.random() > 0.3

    if confirmed:
        print("✅ 用户确认行程")
        state["itinerary_approved"] = True
    else:
        print("⚠️ 用户要求修改")
        state["itinerary_approved"] = False

    return state

def final_output(state: State) -> State:  # 1 usage
    """6. 最终输出"""
    print("\n" + "=" * 50)
    print("最终行程规划完成！")
    print("=" * 50)

    itinerary = state["itinerary"]
    preferences = state["user_preferences"]

    print(f"\n目的地: {itinerary['destination']}")
    print(f"旅行类型: {preferences['travel_type']}")
    print(f"出行人数: {preferences['travelers']}")
    print(f"旅行日期: {preferences['travel_dates']}")
    print(f"总预算: ¥{state['adjusted_budget']}")

    print(f"\n行程安排:")
    for day in itinerary["daily_plan"]:
        print(f"  第{day['day']}天: {', '.join(day['activities'])}")

    print(f"\n住宿: {itinerary['hotel']}")
    print(f"交通: {itinerary['transportation']}")

    return state
    
# 创建图
workflow = StateGraph(State)

# 添加节点
workflow.add_node("input_preferences", input_user_preferences)
workflow.add_node("recommend_destinations", recommend_destinations)
workflow.add_node("calculate_budget", calculate_initial_budget)
workflow.add_node("check_budget", check_budget)
workflow.add_node("adjust_plan", adjust_plan)
workflow.add_node("recalculate_budget", recalculate_budget)
workflow.add_node("generate_itinerary", generate_itinerary)
workflow.add_node("human_review", human_review)
workflow.add_node("wait_confirmation", wait_user_confirmation)
workflow.add_node("user_modification", user_modification)
workflow.add_node("final_output", final_output)

# 设置入口点
workflow.set_entry_point("input_preferences")

# 添加边 (主要流程)
workflow.add_edge(start_key="input_preferences", end_key="recommend_destinations")
workflow.add_edge(start_key="recommend_destinations", end_key="calculate_budget")
workflow.add_edge(start_key="calculate_budget", end_key="check_budget")

# 条件边: 预算检查
def budget_router(state: State):
    if state.get("budget_approved", False):
        return "generate_itinerary"
    else:
        return "adjust_plan"

workflow.add_conditional_edges(
    source="check_budget",
    path=budget_router,
    path_map={
        "adjust_plan": "adjust_plan",
        "generate_itinerary": "generate_itinerary"
    }
)

workflow.add_edge(start_key="adjust_plan", end_key="recalculate_budget")

# 重新计算后再次检查预算
workflow.add_conditional_edges(
    source="recalculate_budget",
    path=budget_router,
    path_map={
        "adjust_plan": "adjust_plan",  # 如果还超支，继续调整
        "generate_itinerary": "generate_itinerary"
    }
)

# 行程生成后的流程
workflow.add_edge(start_key="generate_itinerary", end_key="human_review")
workflow.add_edge(start_key="human_review", end_key="wait_confirmation")

# 条件边：用户确认
def confirmation_router(state: State):
    if state.get("itinerary_approved", False):
        return "final_output"
    else:
        return "user_modification"

workflow.add_conditional_edges(
    source="wait_confirmation",
    path=confirmation_router,
    path_map={
        "final_output": "final_output",
        "user_modification": "user_modification"
    }
)

# 用户修改后重新生成行程
workflow.add_edge(start_key="user_modification", end_key="generate_itinerary")

# 最终输出到结束
workflow.add_edge(start_key="final_output", end_key=END)

# 编译图
app = workflow.compile()

# 可视化图（需要graphviz）
try:
    from IPython.display import Image, display

    display(Image(app.get_graph().draw_mermaid_png()))
except Exception as e:
    print("无法显示图形，但应用已创建")

# 运行应用
if __name__ == "__main__":
    print("开始旅游助手流程...\n")

    # 初始化状态
    initial_state = State(
        user_preferences={},
        recommended_destinations=[],
        selected_destination="",
        initial_budget=0.0,
        adjusted_budget=0.0,
        itinerary={},
        budget_approved=False,
        itinerary_approved=False,
        messages=[]
    )

    # 运行工作流
    final_state = app.invoke(initial_state)

    print("\n" + "=" * 50)
    print("流程执行完成！")
    print("=" * 50)
    
    with open("graph.png", "wb") as f:
        f.write(app.get_graph().draw_mermaid_png())