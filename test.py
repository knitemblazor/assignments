
import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.utils import shuffle

class classifier:
	
	def __init__(self,path,split_ratio):
		self.path=path
		self.split_ratio=split_ratio
	def redeemer(self):
		df=pd.read_csv(self.path)
		df=shuffle(df)
		df=np.array(df)
		data_pts,features=df.shape
		div=int(self.split_ratio*data_pts)
		gnb = GaussianNB()
		y_pred = gnb.fit(df[:div,1:56], df[:div,57]).predict(df[div:,1:56])
		print("Number of mislabeled points :",(df[div:,57] != y_pred).sum())
		return (df[div:,57] == y_pred).sum()/(data_pts-div)

d=classifier("spambase.data",.7)
print("accuracy:",d.redeemer())
