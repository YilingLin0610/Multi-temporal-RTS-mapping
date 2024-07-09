#coding=gbk
"""
Draw the picture showing the fine-tuning performance
Author: Yiling Lin
"""

import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
mpl.rcParams['font.family'] = ['Helvetica']
mpl.rcParams['font.size'] = 40

#data
data_2022=np.array([
[0.638 ,0.644,0.674],[0.730,0.773 ,0.791 ],[0.723 ,0.739  ,0.715  ],
[0.477 ,0.527 ,0.566],[0.647 ,0.699 ,0.734 ],[0.585  ,0.632 ,0.613 ],
[0.201 ,0.257 ,0.340],[0.359,0.426,0.468],[0.264,0.346,0.380]

])




bar_width = 0.25
spacing = 0.02  # spacing between bars
group_spacing = 0.3  # spacing between every bar groups

#
colors=["#B0AEBC","#9EB3C4","#5D82A4","#375874","#DAB9AC","#883F52"]
colors=["#B0AEBC","#9EB3C4","#5D82A4"]


fig, ax = plt.subplots(figsize=(8, 6))
labels=["Fine-tuning : 5%","10%","20%"]
for i in range(len(colors)):
    # Calculate the x-position of each bar
    pos = np.arange(len(data_2022)) + i * (bar_width + spacing)
    ax.bar(pos, data_2022[:, i], bar_width, color=colors[i],label=labels[i])
y_lim=[0.581,0.553,0.624,0.448,0.436,0.494,0.153,0.118 ,0.127 ]
x_min=(np.arange(len(data_2022))-3*bar_width/2)/9
x_max=(np.arange(len(data_2022))+3*bar_width/2)/9

for i in range(len(y_lim)):
    if(i==1):
        ax.axhline(y=y_lim[i], linestyle='-', color='#883F52',xmin=0.048+i*0.103, xmax=0.126+i*0.103,linewidth=3,label="Baseline (direct transfer)")
    else:
        ax.axhline(y=y_lim[i], linestyle='-', color='#883F52', xmin=0.048 + i * 0.103, xmax=0.126 + i * 0.103,
                   linewidth=3)

ax.axhspan(ymin=0.0, ymax=1, facecolor='#EFB0AB', alpha=0.4,xmin=0.01, xmax=0.346, zorder=-1)
ax.axhspan(ymin=0.0, ymax=1, facecolor='#BDD7EE', alpha=0.4,xmin=0.346, xmax=0.66, zorder=-1)
ax.axhspan(ymin=0.0, ymax=1, facecolor='#D9D9D9', alpha=0.4,xmin=0.66, xmax=1, zorder=-1)


ax.set_xticks(np.arange(len(data_2022)) + ((len(colors) - 1) / 2) * (bar_width + spacing))
ax.set_xticklabels(['2020', '2021', '2022','2020', '2021', '2022','2020', '2021', '2022'],fontsize=50)



ax.legend(frameon=True, framealpha=1, edgecolor='none', fontsize=13, ncol=4)


ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_linewidth(4)
ax.spines['left'].set_linewidth(4)
ax.text(1.2, 0.8, "IoU threshold = 0", ha='center', va='center', fontsize=14, weight='bold')
ax.text(4.5, 0.8, "IoU threshold = 0.4", ha='center', va='center', fontsize=14, weight='bold')
ax.text(7.5, 0.8, "IoU threshold = 0.8", ha='center', va='center', fontsize=14, weight='bold')


ax.tick_params(axis='both', which='major', labelsize=12, width=4, length=8)
ax.set_ylabel("F1",fontsize=16)
ax.set_xlabel("Year",fontsize=16)


ax.set_ylim([0,1])


plt.subplots_adjust(top=0.8, bottom=0.2, left=0.1, right=0.9, wspace=group_spacing)

# Show and save the figure
plt.show()
fig.savefig(r'C:\Users\凌凌七\Desktop\学习文件夹\研究生零年级\毕业设计\RSE\images\\'+"compare_fine_tuning2"+".svg", bbox_inches='tight')