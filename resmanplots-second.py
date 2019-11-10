from mpl_toolkits.mplot3d import Axes3D
from AdSFunctions import*
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
import random
import scipy

####
##PLOTTING 2ND RESONANCE MANIFOLD AVERAGE VALUE FOR A GIVEN (X,Y) COORDINATE##
####

N = 10

ThirdResMan = np.loadtxt('ThirdGammas9October2018-N=%.0f.txt'%N).view(float)
GamMax = max(abs(ThirdResMan))

print 'ThirdResMan average is', sum(ThirdResMan) / float(len(ThirdResMan))

C = len(ThreeList(N))
M3 = TTM(N)

data = np.zeros([C,3])

for m in range(C):
    
    n,s,j = ListOfModes[M3[m,0]]
    n1,s1,j1 = ListOfModes[M3[m,1]]
    n2,s2,j2 = ListOfModes[M3[m,2]]
    n3,s3,j3 = ListOfModes[M3[m,3]]
    n4,s4,j4 = ListOfModes[M3[m,4]]
    n5,s5,j5 = ListOfModes[M3[m,5]]
    
    Gam = ThirdResMan[m]
    
    xs = abs(j-j1)
    data[m,0] = xs
    
    ys = np.sqrt((j-j2)**2 + (j-j3)**2 + (j-j4)**2 + (j-j5)**2)
    data[m,1] = ys
    
    zs = Gam
    data[m,2] = zs

P = 10
PlacementArray = np.zeros([5,9])

for m in range(C):
    
    n,s,j = ListOfModes[M3[m,0]]
    n1,s1,j1 = ListOfModes[M3[m,1]]
    n2,s2,j2 = ListOfModes[M3[m,2]]
    n3,s3,j3 = ListOfModes[M3[m,3]]
    
    xs = abs(j-j1)
    ys = int(round(np.sqrt((j-j2)**2 + (j-j3)**2 + (j-j4)**2 + (j-j5)**2),0))

    PlacementArray[xs,ys] += 1

print 'PlacementArray is', PlacementArray
PANormalized = PlacementArray/C

XMAX, YMAX = 4.0, 8.0

# regular grid covering the domain of the data, original aranges were (0, 3.25, 0.5), (0, 6.0, 0.5)
X,Y = np.meshgrid(np.arange(0, XMAX, 0.5), np.arange(0, YMAX, 0.5))
XX = X.flatten()
YY = Y.flatten()

order = 2    # 1: linear, 2: quadratic
if order == 1:
    # best-fit linear plane
    A = np.c_[data[:,0], data[:,1], np.ones(data.shape[0])]
    C,_,_,_ = scipy.linalg.lstsq(A, data[:,2])    # coefficients
    
    # evaluate it on grid
    Z = C[0]*X + C[1]*Y + C[2]

# or expressed using matrix/vector product
#Z = np.dot(np.c_[XX, YY, np.ones(XX.shape)], C).reshape(X.shape)

elif order == 2:
    # best-fit quadratic curve
    A = np.c_[np.ones(data.shape[0]), data[:,:2], np.prod(data[:,:2], axis=1), data[:,:2]**2]
    C,_,_,_ = scipy.linalg.lstsq(A, data[:,2])
    
    # evaluate it on a grid
    Z = np.dot(np.c_[np.ones(XX.shape), XX, YY, XX*YY, XX**2, YY**2], C).reshape(X.shape)

# plot points and fitted surface
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
cax = ax.plot_surface(X, Y, Z, alpha=0.5, rstride=1, cstride=1, cmap=cm.coolwarm)# facecolors=cm.nipy_spectral(PANormalized))
#ax.scatter(data[:,0], data[:,1], data[:,2], c='k', alpha=0.05)#, s=None)

plt.xticks(np.arange(0, XMAX, 1))
plt.yticks(np.arange(0, YMAX, 1))

ax.set_xlabel(r'$|j-j_{1}|$',fontsize=10)
ax.set_ylabel(r'$\sqrt{\sum_{k\neq 1} (j-j_{k})^{2}}$',fontsize=10)
ax.set_zlabel(r'$\langle\Gamma\rangle$',fontsize=10)

#ax.set_xlim(0,3)
#ax.set_ylim(0,6)
#ax.set_zlim(0,0.08)

#m = cm.ScalarMappable(cmap=cm.nipy_spectral)
#m.set_array([])
#plt.colorbar(m)

fig.colorbar(cax)

plt.title('Second Nonlinear Term Resonance Manifold')
plt.tight_layout()
plt.show()
#plt.savefig('SecondResonanceManifold-9October2018.jpg')
