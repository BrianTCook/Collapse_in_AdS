from mpl_toolkits.mplot3d import Axes3D
from AdSFunctions import*
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
import random
import scipy

####
##PLOTTING 1ST RESONANCE MANIFOLD AVERAGE VALUE FOR A GIVEN (X,Y) COORDINATE##
####

N = 12
SGD = np.load('SecondGammaDictionary-N=%s.npy'%N) #np.load('SecondGammaDictionary-N=%s.npy'%N)
vals = SGD.item().values()

C = len(ThreeList(N))

data = np.zeros([C,3])

P = 8
dataCo = np.zeros([P*P,3])

m = 0
for indices in SGD.item().keys():
	
	gam = SGD.item().get(indices)
	n, n1, n2, n3 = indices

	n, s, j = ListOfModes[n]
	n1, s1, j1 = ListOfModes[n1]
	n2, s2, j2 = ListOfModes[n2]
	n3, s3, j3 = ListOfModes[n3]
    
	xs = abs(j-j1)
	data[m,0] = xs

	ys = np.sqrt((j-j2)**2 + (j-j3)**2)
	data[m,1] = ys

	zs = gam
	data[m,2] = zs
    
	xxs = abs(j-j1)
	yys = int(round(np.sqrt((j-j2)**2 + (j-j3)**2),0))

	m += 1

XMAX, YMAX = 4.5, 7.5

# regular grid covering the domain of the data, before combining plots was (0,3.5,0.5) and (0,4,0.5)
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
cax = ax.plot_surface(X, Y, Z, alpha=0.5, rstride=1, cstride=1, cmap=cm.coolwarm)#facecolors=cm.nipy_spectral(PANormalized))
#ax.scatter(data[:,0], data[:,1], data[:,2], c='k', alpha = 0.1)#, s=None)

plt.xticks(np.arange(0, XMAX, 1))
plt.yticks(np.arange(0, YMAX, 1))

ax.set_xlabel(r'$|j-j_{1}|$',fontsize=10)
ax.set_ylabel(r'$\sqrt{\sum_{k\neq 1} (j-j_{k})^{2}}$',fontsize=10)
ax.set_zlabel(r'$\langle \Gamma \rangle$',fontsize=10)

#ax.set_xlim(0,3)
#ax.set_ylim(0,3*np.sqrt(2))
#ax.set_zlim(0,0.2)

#m = cm.ScalarMappable(cmap=cm.nipy_spectral)
#m.set_array([])
#plt.colorbar(m)
fig.colorbar(cax)

#cbar = fig.colorbar(cax)#, ticks=[0, 0.05, 0.1])

plt.title('First Nonlinear Term Resonance Manifold, N=%s'%N)
plt.tight_layout()
plt.savefig('SecondResonanceManifold-N=%s.jpg'%N)

fig = plt.figure()
ax = fig.add_subplot(111)
plt.title('Resonance Manifold Histogram, N=%s'%N)
plt.hist(vals, bins = 100)
ax.set_xlabel(r'$\Gamma$', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
plt.tight_layout()
plt.savefig('SecondResonanceManifoldHistogram-N=%s.jpg'%N)
