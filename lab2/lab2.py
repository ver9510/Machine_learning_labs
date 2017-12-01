import pandas
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

#загрузка датасета
def load_dataset(filename):
	csv_dataset = pandas.read_csv(filename, header=None).values
	dataset=csv_dataset
	return dataset
	
#разбиение на обучающую и тестовую выборки
def split_dataset(dataset,test_size):
	letter_attr = dataset[:,1:] # список атрибутов (признаков) для каждой буквы
	letter_class = dataset[:,0] # классы букв
	data_train, data_test, class_train, class_test = train_test_split(letter_attr, letter_class, test_size=test_size)
	return data_train, class_train, data_test, class_test

#обучение дерева и расчет точности
def train_and_score_tree(dataset,forest,size):
	data_train, class_train, data_test, class_test = split_dataset(dataset, size)
	forest = forest.fit( data_train, class_train )
	return forest.score(data_test, class_test)

def main():
	dataset=load_dataset("letter-recognition.csv")
	random_forest = RandomForestClassifier(n_estimators=100)
	decision_tree = DecisionTreeClassifier(random_state=100)
	# Получение средней точности классификации на тестовых данных
	print("Size of datasets \t Random forest \t Decision tree ")
	print( "60% train, 40% test:\t", train_and_score_tree(dataset,random_forest, 0.4),"\t", train_and_score_tree(dataset,decision_tree,0.4))
	print( "70% train, 30% test:\t", train_and_score_tree(dataset,random_forest, 0.3),"\t", train_and_score_tree(dataset,decision_tree,0.3))
	print( "80% train, 20% test:\t", train_and_score_tree(dataset,random_forest, 0.2),"\t", train_and_score_tree(dataset,decision_tree,0.2))
	print( "90% train, 10% test:\t", train_and_score_tree(dataset,random_forest, 0.1),"\t", train_and_score_tree(dataset,decision_tree,0.1))
	
main()
