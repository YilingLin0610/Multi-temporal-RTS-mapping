# coding=gbk
"""
Draw the picture showing the distribution of RTS's aspect
Author: Yiling Lin
"""
from windrose import WindroseAxes, WindAxes, plot_windrose,wrscatter
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.cm as cm
from distirtution_new import calculate_image_pixels



aspect_landscape=calculate_image_pixels(r"F:\graduate-project\data\rasters_COP30\aspect.tif")



df = pd.read_excel(r'F:\graduate-project\statistics\Insar\Areas\2022_presoon.xls')
aspect=df["aspect"]
font1 = {'family': 'Helvetica',
             'weight': 'normal',
             'size': 25,
             }

font2 = {'family': 'Helvetica',
             'weight': 'normal',
             'size': 20,
             }

ax=WindroseAxes.from_ax()


ax.bar(aspect, df["number"], normed=True, color=(242 / 255, 207 / 255, 211 / 255, 1),alpha=1)
ax.set_xticklabels(['E','NE','N','NW','W','SW','S','SE'],fontname='Helvetica',fontsize=35)



y=np.zeros(len(aspect_landscape))+7
data = {'y':y, 'landscape':aspect_landscape}
df2= pd.DataFrame(data)
plot_windrose(df2, kind='contour', var_name='y',
                  direction_name='landscape',
                  normed=True,cmap=cm.Greys,ax=ax,alpha=0.5,lw=3,linestyle='--',)
ax.set_yticks([4,8,12,16,20])
ax.set_yticklabels([4,8,12,16,20],fontname='Times New Roman',fontsize=35,color="#78625A")
ax.set_title("Aspect",y=-0.13,font="Helvetica",fontsize=35)
ax.set_ylim([0,20])
ax.set_legend=None  
plt.show()
#plt.savefig(r'C:\Users\凌凌七\Desktop\学习文件夹\研究生零年级\毕业设计\RSE\images\Aspect.svg', bbox_inches='tight')

