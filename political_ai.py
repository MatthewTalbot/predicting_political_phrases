import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder
from nltk.stem.snowball import SnowballStemmer
import time
import pickle

from sklearn.model_selection import KFold
class politicalAI:
  def __init__(self, split) -> None:
      self.split = split
      self.training_raw = pd.read_csv("shortned_twitter_logs.csv")
      self.training = self.training_raw.sample(frac = 1, random_state=1)
      self.x_train = self.snowball_stemmer(np.array(self.training["Tweet"]))
      self.y_train = np.array(self.training["Political Party"])
      self.pipe = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer())])
      self.X = self.pipe.fit_transform(self.x_train.flatten())
      self.le = LabelEncoder()
      self.Y = self.le.fit_transform(self.y_train.ravel())
      self.sgd = SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42, max_iter=1000, tol=None)
      self.nb = MultinomialNB()
      self.tree = DecisionTreeClassifier()
      self.file_nb = "nb_model_" + str(self.split) + ".sav"
      self.file_sgd = "sgd_model_" + str(self.split) + ".sav"
      self.file_tree = "tree_model_" + str(self.split) + ".sav"
      
  
  def k_fold(self):
    nb_score = []
    sgd_score = []
    tree_score = []
    kf = KFold(n_splits = self.split)
    for train_index, validate_index in kf.split(self.X):
      X_train, Y_train = self.X[train_index], self.Y[train_index]
      X_validate, Y_validate = self.X[validate_index], self.Y[validate_index]
      
      self.nb.fit(X_train, Y_train)
      self.sgd.fit(X_train, Y_train)
      self.tree.fit(X_train, Y_train)
      
      
      nb_score.append(self.nb.score(X_validate, Y_validate))
      sgd_score.append(self.sgd.score(X_validate, Y_validate))
      tree_score.append(self.tree.score(X_validate, Y_validate))
    

    saving_scores = pd.DataFrame(np.transpose(np.array([nb_score, sgd_score, tree_score])), columns=["NB Score", "SGD Score", "Tree Score"])
    file_fitness = "fitness_scores_" + str(self.split) + ".csv"
    saving_scores.to_csv(file_fitness)
    pickle.dump(self.nb, open(self.file_nb, "wb"))
    pickle.dump(self.sgd, open(self.file_sgd, "wb"))
    pickle.dump(self.tree, open(self.file_tree, "wb"))

  def make_prediction(self):
    valid_input = ""
    while True:
      user_input = input("Did you train your own model?(y/n)")
      if user_input == "y":
        user_input = input("what was the model number? ")
        valid_input = user_input
      user_input = input("Enter the model Number you would like to use(50, 100, 1000, 2000, 4000, 8000): ")
      if user_input == "50" or user_input == "100" or user_input == "1000" or user_input == "2000" or user_input == "4000" or user_input == "8000":
        valid_input = user_input
        break
      else:
        print("Invalid File")
    
    chosen_file_nb = "nb_model_" + valid_input + ".sav"
    chosen_file_sgd = "sgd_model_" + valid_input + ".sav"
    chosen_file_tree = "tree_model_" + valid_input + ".sav"

    load_nb_model = pickle.load(open(chosen_file_nb, "rb"))
    load_sgd_model = pickle.load(open(chosen_file_sgd, "rb"))
    load_tree_model = pickle.load(open(chosen_file_tree, "rb"))

    print("enter q to quit")
    while True:
      value = input("Enter a political phrase: ")
      if value == "q":
        break
    
      phrase = self.pipe.transform(self.snowball_stemmer(value))
      nb_predict = load_nb_model.predict(phrase)
      sgd_predict = load_sgd_model.predict(phrase)
      tree_predict = load_tree_model.predict(phrase)

      print("NB model predicted: " + self.le.inverse_transform(nb_predict)[0])
      print("SGD model predicted: " + self.le.inverse_transform(sgd_predict)[0])
      print("Tree model predicted: " + self.le.inverse_transform(tree_predict)[0])

  def snowball_stemmer(self, word_sentence):
    ss = SnowballStemmer("english")
    snowball_sentence = []

    for sentence in word_sentence:
      snowball_sentence.append(" ".join([ss.stem(word) for word in sentence.split(" ")]))

    return np.array(snowball_sentence)

  
def main():
  split = 8000
  print("Split: " + str(split))
  ai = politicalAI(split)
  '''
  start_time = time.time()
  ai.k_fold()
  end_time = time.time() - start_time
  print("Cross Validation of " + str(split) + " took: " + str(end_time) + "seconds")
  '''
  ai.make_prediction()
  

main()
