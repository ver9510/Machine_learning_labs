import math
import pandas
import numpy
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

# разделение датасета на тестовую и обучающую выборку
def load_dataset(test_size):
	dataset = pandas.read_csv("letter-recognition.csv", header=None).values
	letter_attr = dataset[:,1:] # список атрибутов (признаков) для каждой буквы
	letter_class = dataset[:,0] # классы букв
	data_train, data_test, class_train, class_test = train_test_split(letter_attr, letter_class, test_size=test_size,random_state=55)
	return data_train, class_train, data_test, class_test
	
# Разделяет обучающую выборку по классам таким образом, чтобы можно было получить все элементы,
# принадлежащие определенному классу.
def separate_by_class(data_train, class_train):
	classes_dict = {}
	for i in range(len(data_train)):
		classes_dict.setdefault(class_train[i], []).append(data_train[i])
	return classes_dict

# инструменты для обобщения данных
def mean(numbers): # Среднее значение
	return sum(numbers) / float(len(numbers))

def stand_dev(numbers): # вычисление дисперсии
	var = sum([pow(x - mean(numbers), 2) for x in numbers]) / float(len(numbers) - 1)
	return math.sqrt(var)

def summarize(data_train): # обобщение данных
# Среднее значение и среднеквадратичное отклонение для каждого атрибута
	summaries = [(mean(att_numbers), stand_dev(att_numbers)) for att_numbers in	zip(*data_train)]
	return summaries
	
# Обучение классификатора
def summarize_by_class(data_train, class_train):
	# Разделяет обучающую выборку по классам таким образом, чтобы можно было получить все элементы,
	# принадлежащие определенному классу.
	classes_dict = separate_by_class(data_train, class_train)
	summaries = {}
	for class_name, instances in classes_dict.items():
	# Среднее значение и среднеквадратичное отклонение атрибутов для каждого класса	входных данных
		summaries[class_name] = summarize(instances)
	return summaries
	
# вычисление апостериорной вероятности принадлежности объекта к определенному классу
def calc_probability(x, mean, stdev):
	if stdev == 0:
		stdev += 0.000001 # добавляем эпсилон, если дисперсия равна 0
	exponent = math.exp(-(math.pow(x - mean, 2) / (2 * math.pow(stdev, 2))))
	return (1 / (math.sqrt(2 * math.pi) * stdev)) * exponent
	
# вычисление вероятности принадлежности объекта к каждому из классов
def calc_class_probabilities(summaries, instance_attr):
	probabilities = {}
	for class_name, class_summaries in summaries.items():
		probabilities[class_name] = 1.0
		for i in range(len(class_summaries)):
			mean, stdev = class_summaries[i]
			x = float(instance_attr[i])
			probabilities[class_name] *= calc_probability(x, mean, stdev)
	return probabilities
	
# классификация одного объекта
def classificate_one(summaries, instance_attr):
	# вычисление вероятности принадлежности объекта к каждому из классов
	probabilities = calc_class_probabilities(summaries, instance_attr)
	best_class = None
	max_probability = -1
	for class_name, probability in probabilities.items():
		if best_class is None or probability > max_probability:
			max_probability = probability
			best_class = class_name
	return best_class

# классификация тестовой выборки
def classificate(summaries, data_test):
	predictions = []
	for i in range(len(data_test)):
		result = classificate_one(summaries, data_test[i])
		predictions.append(result)
	return predictions
	
# сравнение результатов классификации с реальными, вычисление точности классификации
def calc_accuracy(summaries, data_test, class_test):
	correct_answer = 0
	# классификация тестовой выборки
	predictions = classificate(summaries, data_test)
	for i in range(len(data_test)):
		if class_test[i] == predictions[i]:
			correct_answer += 1
	return correct_answer / float(len(data_test))
	
def main():
	data_train, class_train, data_test, class_test = load_dataset(4000)
	summary = summarize_by_class(data_train, class_train)
	
	accuracy = calc_accuracy(summary, data_test, class_test)
	print('myNBClass ', 'Accuracy: ', accuracy)
	clf = GaussianNB()
	clf.fit(data_train, class_train)
	print('sklNBClass ', 'Accuracy: ', clf.score(data_test, class_test))
	
	
main()
