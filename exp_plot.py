"""
this code combines different GRNs from experimental data
"""

import matplotlib.pyplot as plt 
import matplotlib.image as img 

f0 = img.imread('Experimental data/Exp GRN/GRN_data0_time0.png') 
f1 = img.imread('Experimental data/Exp GRN/GRN_data0_time1.png') 
f2 = img.imread('Experimental data/Exp GRN/GRN_data0_time2.png') 
f3 = img.imread('Experimental data/Exp GRN/GRN_data0_time3.png') 

fz = 8
plt.figure(1)
ax = plt.subplot(4, 1, 1)
ax.set_title("from t=0h to t=12h", fontsize=fz)
plt.imshow(f0)
plt.axis('off')
ax = plt.subplot(4, 1, 2)
ax.set_title("from t=12h to t=24h", fontsize=fz)
plt.imshow(f1)
plt.axis('off')
ax = plt.subplot(4, 1, 3)
ax.set_title("from t=24h to t=48h", fontsize=fz)
plt.imshow(f2)
plt.axis('off')
ax = plt.subplot(4, 1, 4)
ax.set_title("from t=48h to t=72h", fontsize=fz)
plt.imshow(f3)
plt.axis('off')
plt.subplots_adjust(wspace=0, hspace=0)
plt.savefig('grn0.png', dpi=600, bbox_inches='tight')
plt.show()



f0 = img.imread('Experimental data/Exp GRN/GRN_data1_time0.png') 
f1 = img.imread('Experimental data/Exp GRN/GRN_data1_time1.png') 
f2 = img.imread('Experimental data/Exp GRN/GRN_data1_time2.png') 
f3 = img.imread('Experimental data/Exp GRN/GRN_data1_time3.png') 

fz = 12
plt.figure(1, figsize=(8,8))
ax = plt.subplot(2, 2, 1)
ax.set_title("from t=0d to t=2d", fontsize=fz)
plt.imshow(f0)
plt.axis('off')
ax = plt.subplot(2, 2, 2)
ax.set_title("from t=2d to t=5d", fontsize=fz)
plt.imshow(f1)
plt.axis('off')
ax = plt.subplot(2, 2, 3)
ax.set_title("from t=5d to t=20d", fontsize=fz)
plt.imshow(f2)
plt.axis('off')
ax = plt.subplot(2, 2, 4)
ax.set_title("from t=20d to t=22d", fontsize=fz)
plt.imshow(f3)
plt.axis('off')
plt.subplots_adjust(wspace=0.1, hspace=0.1)
plt.savefig('grn1.png', dpi=600, bbox_inches='tight')
plt.show()



f0 = img.imread('Experimental data/Exp GRN/GRN_data2_time0.png') 
f1 = img.imread('Experimental data/Exp GRN/GRN_data2_time1.png') 
f2 = img.imread('Experimental data/Exp GRN/GRN_data2_time2.png') 
f3 = img.imread('Experimental data/Exp GRN/GRN_data2_time3.png') 
f4 = img.imread('Experimental data/Exp GRN/GRN_data2_time4.png') 

fz = 12
plt.figure(1, figsize=(8,8))
ax = plt.subplot(3, 2, 1)
ax.set_title("from t=0h to t=12h", fontsize=fz)
plt.imshow(f0)
plt.axis('off')
ax = plt.subplot(3, 2, 2)
ax.set_title("from t=12h to t=24h", fontsize=fz)
plt.imshow(f1)
plt.axis('off')
ax = plt.subplot(3, 2, 3)
ax.set_title("from t=24h to t=36h", fontsize=fz)
plt.imshow(f2)
plt.axis('off')
ax = plt.subplot(3, 2, 4)
ax.set_title("from t=36h to t=72h", fontsize=fz)
plt.imshow(f3)
plt.axis('off')
ax = plt.subplot(3, 2, 5)
ax.set_title("from t=72h to t=96h", fontsize=fz)
plt.imshow(f3)
plt.axis('off')
plt.subplots_adjust(wspace=0.1, hspace=0)
plt.savefig('grn2.png', dpi=600, bbox_inches='tight')
plt.show()

