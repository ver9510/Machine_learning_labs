
# coding: utf-8

# In[8]:


import pandas
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier


# In[3]:


#загрузка датасета
def load_dataset(filename):
	csv_dataset = pandas.read_csv(filename, header=0).values
	dataset=csv_dataset
	return dataset


# In[4]:


dataset = load_dataset("train.csv")


# In[6]:


print(dataset)


# In[5]:


class_train = dataset[:,0]
data_train = dataset[:,1:]
print(class_train)
print(data_train[0])


# In[6]:


test_dataset = load_dataset("test.csv")
#data_test=test_dataset[:,1:]
#class_test = test_dataset[:,0]


# In[9]:


forest = DecisionTreeClassifier(random_state=100)
forest = forest.fit( data_train, class_train )


# In[10]:


result = forest.predict(test_dataset)


# In[11]:

print("results")
print(result)


print("result with i")
result1=[]
i=1
for res in result:	
	print(i,",",res)
	i=i+1
	
