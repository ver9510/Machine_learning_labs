from __future__ import division
import pandas
import numpy
import operator
from sklearn.model_selection import train_test_split
from math import sqrt
from collections import Counter
from sklearn.neighbors import KNeighborsClassifier

def load_dataset(test_size):
	csv_dataset = pandas.read_csv("letter-recognition.csv", header=None).values
	dataset=csv_dataset
	letter_attr = dataset[:,1:] # список атрибутов (признаков) для каждой буквы
	letter_class = dataset[:,0] # классы букв
	data_train, data_test, class_train, class_test = train_test_split(letter_attr, letter_class, test_size=test_size)
	return data_train, class_train, data_test, class_test
	
# евклидово расстояние от объекта №1 до объекта №2
def euclidean_distance(instance1, instance2):
	squares = [(i - j) ** 2 for i, j in zip(instance1, instance2)]
	return sqrt(sum(squares))

# рассчет расстояний до всех объектов в датасете
def calc_neighbours_distance(instance, data_train, class_train, k):
	distances = []
	for i in data_train:
		distances.append(euclidean_distance(instance, i))
	distances = tuple(zip(distances, class_train))
	# cортировка расстояний по возрастанию
	# k ближайших соседей
	return sorted(distances, key=operator.itemgetter(0))[:k]

# определение самого распространенного класса среди соседей
def get_most_common(neigbours):
	return Counter(neigbours).most_common()[0][0][1]

# классификация тестовой выборки
def get_predictions(data_train, class_train, data_test, k):
	predictions = []
	for j in data_test:
		neighbours = calc_neighbours_distance(j, data_train, class_train, k)
		most_common = get_most_common(neighbours)
		predictions.append(most_common)
	return predictions

# измерение точности
def calc_accuracy(data_train, class_train, data_test, class_test, k):
	predictions = get_predictions(data_train, class_train, data_test, k)
	mean = [i == j for i, j in zip(class_test, predictions)]
	return sum(mean) / len(mean)

#Сравнение работы реализованного алгоритма с библиотечным:
def main():
	data_train, class_train, data_test, class_test = load_dataset(4000)
	accuracy = calc_accuracy(data_train, class_train, data_test, class_test, 15)
	print('myKNeighboursClass ', 'Accuracy: ', accuracy)
	clf = KNeighborsClassifier(n_neighbors=15)
	clf.fit(data_train, class_train)
	print('sklKNeigboursClass ', 'Accuracy: ', clf.score(data_test, class_test))

main()


