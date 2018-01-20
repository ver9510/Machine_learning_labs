import pandas
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier


# загрузка датасета
def load_dataset(filename):
    dataset = pandas.read_json(filename)
    return dataset


# разбиение на обучающую и тестовую выборки
def split_dataset(dataset_df, test_size):
    ingredients = dataset_df['ingredients'].values  # список ингредиентов (признаков) для каждой кухни
    # print(ingredients)
    array_of_ingredients = make_array_of_strings(ingredients)
    class_cuisine = dataset_df['cuisine'].values  # кухни мира
    # all_ingred_map = create_ingredients_map(ingredients)
    # ingredients_num = encode_ingredients(ingredients, all_ingred_map)
    # print(type(array_of_ingredients))
    data_train, data_test, class_train, class_test = train_test_split(array_of_ingredients, class_cuisine,
                                                                      test_size=test_size)
    return data_train, class_train, data_test, class_test


def make_array_of_strings(receipts):
    array_of_ingredients = []
    for ingredients in receipts:
        str_ingr = " "
        items = []
        for item in ingredients:
            item = "_".join(item.split(" "))
            items.append(item)
        str_ingr = str_ingr.join(items)
        array_of_ingredients.append(str_ingr)
    # print(array_of_ingredients)
    # print(type(array_of_ingredients))
    return array_of_ingredients

def pred_test_data(test_data, vectorizer, classifier):
    ingredients = test_data["ingredients"].values
    test_data = make_array_of_strings(ingredients)
    X_test_data = vectorizer.transform(test_data)
    pred_cuisines = classifier.predict(X_test_data)
    return pred_cuisines

def print_submission(ids, predictions):
    print("id,cuisine")
    i=0
    for id in ids:
        print(id, predictions[i])
        i=i+1

def main():
    dataset = load_dataset("train.json/train.json")
    random_forest = RandomForestClassifier(n_estimators=100)
    data_train, class_train, data_test, class_test = split_dataset(dataset, 0.25)
    vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5)
    X_train = vectorizer.fit_transform(data_train)
    X_test = vectorizer.transform(data_test)

    # random_forest = random_forest.fit(X_train, class_train)
    # pred_forest = random_forest.predict(X_test)
    # score = metrics.accuracy_score(class_test, pred_forest)
    # print("RandomForest accuracy:   %0.3f" % score)
    # RandomForest accuracy: 0.710

    kn_classifier = KNeighborsClassifier(n_neighbors=10)
    kn_classifier = kn_classifier.fit(X_train,class_train)
    pred_kn = kn_classifier.predict(X_test)
    score = metrics.accuracy_score(class_test, pred_kn)
    print("KNeighbors accuracy:   %0.3f" % score)

    # test_data = load_dataset("test.json/test.json")
    # pred_cuisines = pred_test_data(test_data,vectorizer,random_forest)
    # print(pred_cuisines)
    #
    # print_submission(test_data["id"].values, pred_cuisines)



main()
