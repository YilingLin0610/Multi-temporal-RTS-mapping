"""
Draw the pie picture showing soil type distribution
Author: Yiling Lin
"""
#coding=gbk

from distirtution_new import calculate_image_pixels
import matplotlib.pyplot as plt
from collections import Counter
plt.rcParams['font.sans-serif'] = ['Helvetica']
plt.rcParams["font.size"] = 20
# Read the soil texture TIFF file
soil_texture_landscape = calculate_image_pixels( "F:\graduate-project\statistics\TADA_statistics\datasets\soil_texture\soil_texture.tif")

counts = Counter(soil_texture_landscape )
total_elements = len(soil_texture_landscape )

percentages = {num: count / total_elements * 100 for num, count in counts.items()}

elements =["Sandy loam soil","Loam soil","Silt loam soil","Loamy sand soil"]

weight2 = [0.253968254,0.53968254,0.206349206,0]
weight1 = [0.2423708598376837,59.75200153542444/100,5.859974775169993/100,10.150937705637201/100]
cs =["#CEBFD1","#F3CED3","#C0B1A8","#7F7F7F"]
outer_cs = cs
inner_cs = cs

fig = plt.figure(figsize=(12, 8),

                )

# draw the outer ring
wedges1, texts1, autotexts1 = plt.pie(x=weight1,
                                      autopct='%3.1f%%',
                                      radius=1,
                                      pctdistance=0.85,
                                      labels=elements,
                                      #startangle=90,
                                      counterclock=False,
                                      colors=cs,
                                      wedgeprops={'edgecolor': 'white',
                                                  'linewidth': 2,
                                                  'linestyle': '-'
                                                  },
                                     )

# draw the inner ring
wedges2, texts2, autotexts2 = plt.pie(x=weight2,
                                      autopct='%3.1f%%',
                                      radius=0.7,
                                      pctdistance=0.75,
                                      labels=elements,

                                      counterclock=False,
                                      colors=inner_cs,
                                      wedgeprops={'edgecolor': 'white',
                                                  'linewidth': 2,
                                                  'linestyle': '-'
                                                  },

                                     )


plt.title("Soil type",y=0,font="Helvetica",fontsize=25)
plt.show()
#plt.savefig(r'C:\Users\凌凌七\Desktop\学习文件夹\研究生零年级\毕业设计\RSE\images\soil texture.svg', bbox_inches='tight')