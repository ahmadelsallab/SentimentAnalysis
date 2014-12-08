'''
Created on Dec 8, 2014

@author: asallab
'''
f = [{0:2, 1:-1}, {0:1, 1:-3}, {0:5, 1:-1}, {0:1, 1:-1}]
l = [4, 2, 1, 3]
from liblinearutil import train
cParam = 16# Best cross validation accuracy
nFoldsParam = 10
classifierModel = train(l, f, '-c ' + str(cParam))
train(l, f, '-c ' + str(cParam) + ' -v ' + str(nFoldsParam))