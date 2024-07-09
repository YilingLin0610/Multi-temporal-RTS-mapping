"""
Draw the picture showing the TADA performance
Author: Yiling Lin
"""
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

mpl.rcParams['font.family'] = ['helvetica']
mpl.rcParams['font.size'] = 40

# Data
data_2022=np.array([
[0.696],[0.762],[0.811],
[0.641],[0.724],[0.767],
[0.171],[0.290],[0.313]
])

bar_width = 0.6
spacing = 0.02  # spacing between bars
group_spacing = 0.3  # spacing between every bar groups

# 定义颜色
colors=["#B0AEBC","#9EB3C4","#5D82A4","#375874","#DAB9AC","#883F52"]
colors=["#B0AEBC"]
# 绘制图形
fig, ax = plt.subplots(figsize=(8, 6))
labels=["TADA"]
for i in range(len(colors)):
    # Calculate the x-position of each bar
    pos = np.arange(len(data_2022)) + i * (bar_width + spacing)
    ax.bar(pos, data_2022[:, i], bar_width, color=colors[i],label=labels[i])

ax.set_xticks(np.arange(len(data_2022)) + ((len(colors) - 1) / 2) * (bar_width + spacing))
ax.set_xticklabels(['2020', '2021', '2022','2020', '2021', '2022','2020', '2021', '2022'],fontsize=50)

#Draw the baseline
y_lim=[0.581,0.553,0.624,0.448,0.436,0.494,0.153,0.118 ,0.127 ]
for i in range(len(y_lim)):
    if(i==1):
        ax.axhline(y=y_lim[i], linestyle='-', color='#883F52', xmin=0.048 + i * 0.1055, xmax=0.110 + i * 0.1055,
                   linewidth=3,label="Baseline (direct transfer)")
    else:
        ax.axhline(y=y_lim[i], linestyle='-', color='#883F52',xmin=0.048+i*0.1055, xmax=0.110+i*0.1055,linewidth=3)
ax.axhspan(ymin=0.0, ymax=1, facecolor='#EFB0AB', alpha=0.4,xmin=0.02, xmax=0.35, zorder=-1)
ax.axhspan(ymin=0.0, ymax=1, facecolor='#BDD7EE', alpha=0.4,xmin=0.35, xmax=0.68, zorder=-1)
ax.axhspan(ymin=0.0, ymax=1, facecolor='#D9D9D9', alpha=0.4,xmin=0.68, xmax=1, zorder=-1)


ax.legend(frameon=True, framealpha=1, edgecolor='none', fontsize=15, loc='upper right', ncol=2)


ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_linewidth(4)
ax.spines['left'].set_linewidth(4)


ax.tick_params(axis='both', which='major', labelsize=12, width=4, length=8)
ax.legend(frameon=True, framealpha=1, edgecolor='none', fontsize=13, loc='upper right', ncol=2)

ax.set_ylim([0,1])
ax.set_ylabel("F1",fontsize=16)
ax.set_xlabel("Year",fontsize=16)

plt.subplots_adjust(top=0.8, bottom=0.2, left=0.1, right=0.9, wspace=group_spacing)
ax.text(1.2, 0.8, "IoU threshold = 0", ha='center', va='center', fontsize=14, weight='bold')
ax.text(4.5, 0.8, "IoU threshold = 0.4", ha='center', va='center', fontsize=14, weight='bold')
ax.text(7.5, 0.8, "IoU threshold = 0.8", ha='center', va='center', fontsize=14, weight='bold')
# Show and save the figure
plt.show()
fig.savefig(r'C:\Users\凌凌七\Desktop\学习文件夹\研究生零年级\毕业设计\RSE\images\\'+"TADA"+".svg", bbox_inches='tight')