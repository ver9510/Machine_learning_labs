import pandas
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

#загрузка датасета
def load_dataset(filename):
	dataset = pandas.read_json(filename)
	return dataset
	
#разбиение на обучающую и тестовую выборки
def split_dataset(dataset_df,test_size):
	ingredients = dataset_df['ingredients'].values # список ингредиентов (признаков) для каждой кухни
	print(ingredients)
	array_of_ingredients = make_array_of_strings(ingredients)
	class_cuisine = dataset_df['cuisine'].values # кухни мира
	#all_ingred_map = create_ingredients_map(ingredients)
	#ingredients_num = encode_ingredients(ingredients, all_ingred_map)
	data_train, data_test, class_train, class_test = train_test_split(array_of_ingredients, class_cuisine, test_size=test_size)
	return data_train, class_train, data_test, class_test
	
def make_array_of_strings(receipts):
	array_of_ingredients=[]
	for ingredients in receipts:
		str_ingr=""
		str_ingr = str_ingr.join(ingredients)
		array_of_ingredients.append(str_ingr)
	print(array_of_ingredients)
			
	
def main():
	dataset = load_dataset("train.json/train.json")
	random_forest = RandomForestClassifier(n_estimators=100)
	data_train, class_train, data_test, class_test = split_dataset(dataset,0.25)
	vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5)
	X_train = vectorizer.fit_transform(data_train)
	X_test = vectorizer.transform(data_test)
	random_forest = random_forest.fit( X_train, class_train )
	pred = clf.predict(X_test)
	score = metrics.accuracy_score(y_test, pred)
	print("accuracy:   %0.3f" % score)
	
main()
