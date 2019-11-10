from AdSFunctions import*
from matplotlib.colors import LogNorm
import numpy as np
import matplotlib.pyplot as plt

valsList = [[3.000,'m','3',1500],[4.243,'b','3\sqrt{2}',750],[6.000,'g','6',400],[8.485,'r','6\sqrt{2}',200]]

N = 14

plt.figure()

for eps, color, eps_analytic, TotalTime in valsList:
	BA = np.loadtxt('BArrayData-eps=%.03f-N=%i.txt'%(eps,N)).view(complex)

	T = TotalTime*80

	ranges = [range(N),range(T)]

	Pi = np.zeros([T],dtype=complex)

	for l,m in itertools.product(*ranges):
		n,s,j = ListOfModes[l]
		Pi[m] = (1/2.)*np.sqrt(3+2*j)*eAnalytic(0,j)*BA[l,m]

	Pi_Squared = Pi*np.conj(Pi)

	tLS  = np.linspace(0,TotalTime,T)

	X1 = []
	Y1 = []

	for t in range(4,T):
		if t%800 == 0:
	    		X1.append(tLS[t])
	    		Y1.append(Pi_Squared[t])

	plt.semilogy(X1,Y1,color,label=r'$\epsilon = %s$'%eps_analytic)

plt.xlabel(r'$t$',fontsize=16)
plt.ylabel(r'$\Pi^{2}(t,0)$',fontsize=16)
plt.legend(loc='best')
plt.tight_layout()
plt.savefig('PiS-8February2019.jpg')

plt.figure()

for eps, color, eps_analytic, TotalTime in valsList:
	BA = np.loadtxt('BArrayData-eps=%.03f-N=%i.txt'%(eps,N)).view(complex)
	T = TotalTime*80

	ranges = [range(N),range(T)]

	Pi = np.zeros([T],dtype=complex)

	for l,m in itertools.product(*ranges):
		n,s,j = ListOfModes[l]
		Pi[m] = (1/2.)*np.sqrt(3+2*j)*eAnalytic(0,j)*BA[l,m]

	Pi_Squared = Pi*np.conj(Pi)

	tLS  = np.linspace(0,TotalTime,T)

	X1 = []
	Y1 = []

	for t in range(4,T):
		if t%800 == 0:
	    		X1.append(tLS[t])
	    		Y1.append(eps**-2 * Pi_Squared[t])

	plt.semilogy(X1,Y1,color,label=r'$\epsilon = %s$'%eps_analytic)

plt.xlabel(r'$t$',fontsize=16)
plt.ylabel(r'$\epsilon^{-2} \Pi^{2}(t,0)$',fontsize=16)
plt.legend(loc='best')
plt.tight_layout()
plt.savefig('PiS-8February2019-adj.jpg')
