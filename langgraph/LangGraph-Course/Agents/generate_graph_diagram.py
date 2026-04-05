# 导入绑图库
import matplotlib.pyplot as plt
# 导入图形补丁模块
import matplotlib.patches as mpatches
# 导入圆角矩形和箭头补丁
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
# 导入numpy数学计算库
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 创建画布和坐标轴
fig, ax = plt.subplots(1, 1, figsize=(16, 12))
# 设置坐标轴范围
ax.set_xlim(0, 16)
ax.set_ylim(0, 12)
# 隐藏坐标轴
ax.axis('off')

# 颜色方案
color_llm = '#4A90E2'
color_retriever = '#7ED321'
color_condition = '#F5A623'
color_end = '#BD10E0'
color_start = '#50E3C2'
arrow_color = '#4A4A4A'

# 节点位置
start_pos = (8, 11)
llm_pos = (8, 8.5)
condition_pos = (8, 6)
retriever_pos = (4, 3.5)
end_pos = (12, 3.5)

# 绘制 START 节点
start_box = FancyBboxPatch((start_pos[0]-1, start_pos[1]-0.4), 2, 0.8,
                            boxstyle="round,pad=0.3",
                            facecolor=color_start, edgecolor='white', linewidth=2)
ax.add_patch(start_box)
ax.text(start_pos[0], start_pos[1], 'START', ha='center', va='center',
        fontsize=14, fontweight='bold', color='white')

# 绘制 llm 节点
llm_box = FancyBboxPatch((llm_pos[0]-1.5, llm_pos[1]-0.6), 3, 1.2,
                          boxstyle="round,pad=0.3",
                          facecolor=color_llm, edgecolor='white', linewidth=2)
ax.add_patch(llm_box)
ax.text(llm_pos[0], llm_pos[1]+0.15, 'llm 节点', ha='center', va='center',
        fontsize=13, fontweight='bold', color='white')
ax.text(llm_pos[0], llm_pos[1]-0.25, '调用 LLM (gpt-4o)', ha='center', va='center',
        fontsize=10, color='white')

# 绘制 condition 节点
condition_box = FancyBboxPatch((condition_pos[0]-1.8, condition_pos[1]-0.5), 3.6, 1,
                                boxstyle="round,pad=0.3",
                                facecolor=color_condition, edgecolor='white', linewidth=2)
ax.add_patch(condition_box)
ax.text(condition_pos[0], condition_pos[1]+0.15, 'should_continue', ha='center', va='center',
        fontsize=12, fontweight='bold', color='white')
ax.text(condition_pos[0], condition_pos[1]-0.2, '检查工具调用', ha='center', va='center',
        fontsize=10, color='white')

# 绘制 retriever_agent 节点
retriever_box = FancyBboxPatch((retriever_pos[0]-1.8, retriever_pos[1]-0.6), 3.6, 1.2,
                                boxstyle="round,pad=0.3",
                                facecolor=color_retriever, edgecolor='white', linewidth=2)
ax.add_patch(retriever_box)
ax.text(retriever_pos[0], retriever_pos[1]+0.2, 'retriever_agent 节点', ha='center', va='center',
        fontsize=12, fontweight='bold', color='white')
ax.text(retriever_pos[0], retriever_pos[1]-0.1, '执行 retriever_tool', ha='center', va='center',
        fontsize=10, color='white')
ax.text(retriever_pos[0], retriever_pos[1]-0.35, '查询 ChromaDB', ha='center', va='center',
        fontsize=10, color='white')

# 绘制 END 节点
end_box = FancyBboxPatch((end_pos[0]-1, end_pos[1]-0.4), 2, 0.8,
                          boxstyle="round,pad=0.3",
                          facecolor=color_end, edgecolor='white', linewidth=2)
ax.add_patch(end_box)
ax.text(end_pos[0], end_pos[1], 'END', ha='center', va='center',
        fontsize=14, fontweight='bold', color='white')

# 绘制箭头
# START -> llm （从起点到LLM节点的箭头）
ax.annotate('', xy=(llm_pos[0], llm_pos[1]+0.6), xytext=(start_pos[0], start_pos[1]-0.4),
            arrowprops=dict(arrowstyle='->', color=arrow_color, lw=2.5))

# llm -> condition （从LLM到条件判断的箭头）
ax.annotate('', xy=(condition_pos[0], condition_pos[1]+0.5), xytext=(llm_pos[0], llm_pos[1]-0.6),
            arrowprops=dict(arrowstyle='->', color=arrow_color, lw=2.5))

# condition -> retriever_agent (True) （条件为真时转到检索代理）
ax.annotate('', xy=(retriever_pos[0]+0.5, retriever_pos[1]+0.6),
            xytext=(condition_pos[0]-1.5, condition_pos[1]-0.5),
            arrowprops=dict(arrowstyle='->', color=arrow_color, lw=2.5))
ax.text(5.2, 5.2, 'True (有工具调用)', ha='center', va='center',
        fontsize=10, fontweight='bold', color=color_retriever,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=color_retriever, alpha=0.9))

# condition -> END (False) （条件为假时结束）
ax.annotate('', xy=(end_pos[0]-0.5, end_pos[1]+0.4),
            xytext=(condition_pos[0]+1.5, condition_pos[1]-0.5),
            arrowprops=dict(arrowstyle='->', color=arrow_color, lw=2.5))
ax.text(10.8, 5.2, 'False (无工具调用)', ha='center', va='center',
        fontsize=10, fontweight='bold', color=color_end,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=color_end, alpha=0.9))

# retriever_agent -> llm (循环) （检索完成后返回LLM继续处理）
ax.annotate('', xy=(llm_pos[0]-1.2, llm_pos[1]-0.3),
            xytext=(retriever_pos[0]-1.8, retriever_pos[1]),
            arrowprops=dict(arrowstyle='->', color=arrow_color, lw=2.5,
                          connectionstyle="arc3,rad=-0.4"))
ax.text(2.5, 6.5, '返回 LLM (循环)', ha='center', va='center',
        fontsize=10, fontweight='bold', color=color_llm,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=color_llm, alpha=0.9))

# 添加标题
ax.text(8, 11.5, 'RAG Agent Graph 流程图', ha='center', va='center',
        fontsize=18, fontweight='bold', color='#2C3E50')

# 添加图例
legend_elements = [
    mpatches.Patch(facecolor=color_start, label='START - 开始'),
    mpatches.Patch(facecolor=color_llm, label='llm - LLM 调用节点'),
    mpatches.Patch(facecolor=color_condition, label='should_continue - 条件检查'),
    mpatches.Patch(facecolor=color_retriever, label='retriever_agent - 工具执行节点'),
    mpatches.Patch(facecolor=color_end, label='END - 结束')
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=10,
          framealpha=0.9, edgecolor='gray')

# 添加说明文本
ax.text(0.5, 1.5, '执行流程:', fontsize=12, fontweight='bold', color='#2C3E50')
ax.text(0.5, 1.0, '1. 用户输入问题 → 2. LLM 判断是否需要工具 → 3. 如需工具则执行检索 → 4. 返回 LLM 生成答案 → 5. 结束',
        fontsize=9, color='#4A4A4A')

# 自动调整布局
plt.tight_layout()
# 保存图片，设置分辨率为300dpi
plt.savefig('RAG_Agent_Graph.png', dpi=300, bbox_inches='tight', facecolor='white')
print("流程图已保存为 RAG_Agent_Graph.png")
# 关闭图形释放内存
plt.close()
