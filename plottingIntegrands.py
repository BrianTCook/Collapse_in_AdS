from AdSFunctions import*
import matplotlib.pyplot as plt
import numpy as np
import random

X = np.linspace(0.01,np.pi/2-0.01,500)

sList = [-1,1]
jList = [0,1,2,3,4,5,6]

Y = []

s,s1,s2,s3,s4,s5,j,j1,j2,j3,j4,j5 = random.choice(sList), random.choice(sList), random.choice(sList), random.choice(sList), random.choice(sList), random.choice(sList), random.choice(jList), random.choice(jList), random.choice(jList), random.choice(jList), random.choice(jList), random.choice(jList)

Ys = []

while len(Ys) < 10:
	Y = []
	s,s1,s2,s3,s4,s5,j,j1,j2,j3,j4,j5 = random.choice(sList), random.choice(sList), random.choice(sList), random.choice(sList), random.choice(sList), random.choice(sList), random.choice(jList), random.choice(jList), random.choice(jList), random.choice(jList), random.choice(jList), random.choice(jList)
	for x in X:
		Y.append(IntegrandGauss3(x,s,s1,s2,s3,s4,s5,j,j1,j2,j3,j4,j5))
	Ys.append(Y)

plt.figure()
for Y in Ys:
	plt.plot(X,Y)
	plt.axhline(y=0, color='k',linestyle='--')

plt.show()
	
