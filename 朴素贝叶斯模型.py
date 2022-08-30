#1 读取数据

import pandas as pd
df = pd.read_excel('肿瘤数据.xlsx')
df.head()

#2 划分特征变量和目标变量

X = df.drop(columns='肿瘤性质') 
y = df['肿瘤性质']   

#3 模型搭建
#划分训练集和测试集
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

#朴素贝叶斯模型
from sklearn.naive_bayes import GaussianNB
nb_clf = GaussianNB()  # 高斯朴素贝叶斯模型
nb_clf.fit(X_train,y_train)

y_pred=nb_clf.predict(X_test)

a=pd.DataFrame()
a['预测值']=list(y_pred)
a['实际值']=list(y_test)
print(a)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_pred,y_test))

