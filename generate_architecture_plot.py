import matplotlib.pyplot as plt
import matplotlib.patches as patches

fig, ax = plt.subplots(figsize=(12, 6))

# Define node properties
box_props = dict(boxstyle="round,pad=0.5", facecolor="#e1f5fe", edgecolor="#0277bd", linewidth=1.5)
text_props = dict(fontsize=12, ha="center", va="center", fontweight="bold", color="#01579b")
arrow_props = dict(facecolor="#424242", edgecolor="#424242", arrowstyle="->,head_width=0.5,head_length=0.6", lw=2)

# Define node positions
nodes = {
    "Query": (0.1, 0.5),
    "Retrieval\n(Phase 2: Hetero)": (0.35, 0.5),
    "Reranking\n(Phase 1: PRF)": (0.6, 0.5),
    "Evidence Org\n(Phase 3)": (0.85, 0.7),
    "Generation\n(LLM)": (0.85, 0.4),
    "Verification\n(Phase 4: CoVe)": (0.85, 0.1),
    "Answer / Reject": (1.1, 0.4)
}

# Draw nodes
for name, pos in nodes.items():
    ax.text(pos[0], pos[1], name, bbox=box_props, **text_props)

# Draw arrows
ax.annotate("", xy=(0.23, 0.5), xytext=(0.15, 0.5), arrowprops=arrow_props)
ax.annotate("", xy=(0.49, 0.5), xytext=(0.46, 0.5), arrowprops=arrow_props)

# Rerank to Evidence Org
ax.annotate("", xy=(0.76, 0.65), xytext=(0.69, 0.55), arrowprops=arrow_props)

# Evidence Org to Gen
ax.annotate("", xy=(0.85, 0.46), xytext=(0.85, 0.64), arrowprops=arrow_props)

# Gen to Verify
ax.annotate("", xy=(0.85, 0.16), xytext=(0.85, 0.34), arrowprops=arrow_props)

# Output
ax.annotate("", xy=(1.01, 0.4), xytext=(0.94, 0.4), arrowprops=arrow_props)

# PRF Loop
ax.annotate("Confidence < τ\n(Expand & Re-retrieve)", 
            xy=(0.35, 0.6), xytext=(0.6, 0.6), 
            arrowprops=dict(facecolor="#d32f2f", edgecolor="#d32f2f", arrowstyle="->,head_width=0.4", lw=1.5, connectionstyle="arc3,rad=-0.4"),
            fontsize=10, ha="center", va="bottom", color="#c62828", fontweight="bold")

# Reject Loop
ax.annotate("Conflict Detected\n(Trigger No-Answer)", 
            xy=(1.05, 0.3), xytext=(0.94, 0.1), 
            arrowprops=dict(facecolor="#d32f2f", edgecolor="#d32f2f", arrowstyle="->,head_width=0.4", lw=1.5, connectionstyle="arc3,rad=-0.2"),
            fontsize=10, ha="left", va="top", color="#c62828", fontweight="bold")

ax.set_xlim(0, 1.2)
ax.set_ylim(0, 0.9)
ax.axis("off")

plt.tight_layout()
plt.savefig("paper/zjuthesis/figures/rag_architecture_flow.png", dpi=300, bbox_inches='tight')
print("✅ Architecture flow diagram generated successfully.")
