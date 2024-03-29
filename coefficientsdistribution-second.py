from mpl_toolkits.mplot3d import Axes3D
from AdSFunctions import*
from matplotlib import cm
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
import numpy as np
import random
import scipy

######
##PLOTTING DISTRIBUTION OF INTERACTION COEFFICIENTS, 2ND RESONANCE MANIFOLD##
######

N = 10

ThirdResMan = np.loadtxt('ThirdGammas9October2018-N=%.0f.txt'%N).view(float)
GamMax = max(abs(ThirdResMan))

C = len(ThreeList(N))
M3 = TTM(N)

J = 5
K = 9

data = np.zeros([J*K,3])

PlacementArray = np.zeros([J,K])

for m in range(C):
    
    n,s,j = ListOfModes[M3[m,0]]
    n1,s1,j1 = ListOfModes[M3[m,1]]
    n2,s2,j2 = ListOfModes[M3[m,2]]
    n3,s3,j3 = ListOfModes[M3[m,3]]
    n4,s4,j4 = ListOfModes[M3[m,4]]
    n5,s5,j5 = ListOfModes[M3[m,5]]
    
    xs = abs(j-j1)
    ys = int(round(np.sqrt((j-j2)**2 + (j-j3)**2 + (j-j4)**2 + (j-j5)**2),0))
    
    PlacementArray[xs,ys] += 1

XValues, YValues, ZValues = [], [], []

for j in range(J):
    for k in range(K):
        XValues.append(j)
        YValues.append(k)
        ZValues.append(PlacementArray[j,k])

data[:,0], data[:,1], data[:,2] = XValues, YValues, ZValues

# regular grid covering the domain of the data
X,Y = np.meshgrid(np.arange(0, J, 0.5), np.arange(0, K, 0.5))
XX = X.flatten()
YY = Y.flatten()

order = 2    # 1: linear, 2: quadratic
# best-fit quadratic curve
A = np.c_[np.ones(data.shape[0]), data[:,:2], np.prod(data[:,:2], axis=1), data[:,:2]**2]
C,_,_,_ = scipy.linalg.lstsq(A, data[:,2])
    
# evaluate it on a grid
Z = np.dot(np.c_[np.ones(XX.shape), XX, YY, XX*YY, XX**2, YY**2], C).reshape(X.shape)

# plot points and fitted surface
fig, ax = plt.subplots()
#ax = fig.add_subplot(111, projection='3d')
#cax = ax.scatter(XValues, YValues, ZValues)# rstride=1, cstride=1, alpha=0.5, cmap=cm.coolwarm)
#ax.scatter(data[:,0], data[:,1], data[:,2], c='r')#, s=None)
im = ax.imshow(PlacementArray, cmap='coolwarm', interpolation='nearest', origin = 'lower', norm=LogNorm())

ax.set_xticks(np.arange(0, K, 1))
ax.set_yticks(np.arange(0, J, 1))

ax.set_ylabel(r'$|j-j_{1}|$',fontsize=10)
ax.set_xlabel(r'$\sqrt{\sum_{k\neq 1} (j-j_{k})^{2}}$',fontsize=10)

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = fig.colorbar(im, cax=cax)

plt.tight_layout()
#plt.show()
plt.savefig('SecondCoefficientsDistribution.jpg')

