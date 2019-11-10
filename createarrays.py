from AdSFunctions import *
import numpy as np

N = 12

valsList = [[3,1500],[3*np.sqrt(2),750],[6,400],[6*np.sqrt(2),200]]

for eps, TotalTime in valsList:

	T = TotalTime*80

	FOBA = FOBArray(T,eps,TotalTime,N)
	np.savetxt('FOBArrayData-eps=%.03f-N=%i.txt'%(eps,N), FOBA.view(float))

	ASTwo = AvoidSumTwo(T,eps,TotalTime,N)
	np.savetxt('ASTwoData-eps=%.03f-N=%i.txt'%(eps,N), ASTwo.view(float))

	ASThree = AvoidSumThree(T,eps,TotalTime,N)
	np.savetxt('ASThreeData-eps=%.03f-N=%i.txt'%(eps,N), ASThree.view(float))

	BdA = BdotArray(T,eps,TotalTime,N)
	np.savetxt('BdotArrayData-eps=%.03f-N=%i.txt'%(eps,N), BdA.view(float))

	BA = BArray(T,eps,TotalTime,N)
	np.savetxt('BArrayData-eps=%.03f-N=%i.txt'%(eps,N), BA.view(float))
