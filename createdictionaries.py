from AdSFunctions import*
import numpy as np

N = 12

'''
SecGamDic = SGD(N)
np.save('SecondGammaDictionary-N=%i.npy'%N, SecGamDic)
'''

ThiGamDic = TGD(N)
np.save('ThirdGammaDictionary-N=%i.npy'%N, ThiGamDic)
