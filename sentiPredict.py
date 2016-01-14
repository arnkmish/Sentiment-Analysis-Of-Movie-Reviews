'''

	This sentiment predictor python file does some very basic feature extraction from the review collection and then applies and compares various Supervised Learning algorithms' performances. The feature extraction process first gets rid of all the punctuation marks from the reviews, then it gets rid of all the non alphabetic characters from the reviews and finally performs stemming operation on each word in the review strings. After all these preprocessing, first the Bag Of Words model transformation and then the TF-IDF Model transformation is done to all the reviews, so that the features are in numerical form, which are finally used to train various supervised learning classifiers. And finally it prints out the respective accuracies of the trained classfiers. The training and testing test contains the 90% and 10% of the entire review data respectively.
 
'''

from pymongo import MongoClient
import nltk
import sklearn
import numpy as np
import string
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.linear_model import SGDClassifier, LinearRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier


if __name__ == "__main__":
	client = MongoClient()
	dBase = client.movieReviews

	shuffledMovRevs = dBase.shuffledMovRevs

	listOfReviews = []
	listOfSentiments = []
	badString = []
	stemmer = SnowballStemmer("english")
	print "Feature extraction started ..."
	for item in shuffledMovRevs.find():
		try:
			example_sentence = str(item['text'])
			final_sentence = example_sentence.translate(None, string.punctuation)
			wordList = word_tokenize(final_sentence)
			stemmedWordList = []
			s = ' '.join([i for i in wordList if i.isalpha()])
			wordList = word_tokenize(s)
			for words in wordList:
				stemmedWordList.append(stemmer.stem(words))
			final_sentence = ' '.join(stemmedWordList)
			listOfReviews.append(final_sentence) 
			listOfSentiments.append(item['sentiment'])
		except:
			badString.append(count)
	
	
	Y = np.asarray(listOfSentiments)
	
	count_vect = CountVectorizer(ngram_range = (1,3), max_features = 3250)
	X_train_counts = count_vect.fit_transform(listOfReviews)

	
	tfidf_transformer = TfidfTransformer()
	X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
	X = X_train_tfidf.toarray()

	
	print "GaussianNB training started..."
	clf4 = GaussianNB().fit(X[:int(float(len(listOfReviews)) * 0.9)], Y[:int(float(len(listOfSentiments)) * 0.9)])
	print "Prediction accuracy score by GaussianNB: ", clf4.score(X[int(float(len(listOfReviews)) * 0.9):], Y[int(float(len(listOfSentiments)) * 0.9):])

	print "MultinomialNB training started..."
	clf4 = MultinomialNB().fit(X[:int(float(len(listOfReviews)) * 0.9)], Y[:int(float(len(listOfSentiments)) * 0.9)])
	print "Prediction accuracy score by MultinomialNB: ", clf4.score(X[int(float(len(listOfReviews)) * 0.9):], Y[int(float(len(listOfSentiments)) * 0.9):])
	
	print "BernoulliNB training started..."
	clf4 = BernoulliNB().fit(X[:int(float(len(listOfReviews)) * 0.9)], Y[:int(float(len(listOfSentiments)) * 0.9)])
	print "Prediction accuracy score by BernoulliNB: ", clf4.score(X[int(float(len(listOfReviews)) * 0.9):], Y[int(float(len(listOfSentiments)) * 0.9):])
	'''
	print "SVC training started..."
	clf4 = SVC().fit(X[:int(float(len(listOfReviews)) * 0.9)], Y[:int(float(len(listOfSentiments)) * 0.9)])
	print "Prediction accuracy score by SVC: ", clf4.score(X[int(float(len(listOfReviews)) * 0.9):], Y[int(float(len(listOfSentiments)) * 0.9):])
	'''
	print "LinearSVC training started..."
	clf4 = LinearSVC().fit(X[:int(float(len(listOfReviews)) * 0.9)], Y[:int(float(len(listOfSentiments)) * 0.9)])
	print "Prediction accuracy score by LinearSVC: ", clf4.score(X[int(float(len(listOfReviews)) * 0.9):], Y[int(float(len(listOfSentiments)) * 0.9):])
	
	print "DecisionTreeClassifier training started..."
	clf5 = DecisionTreeClassifier(min_samples_split = 40).fit(X[:int(float(len(listOfReviews)) * 0.9)], Y[:int(float(len(listOfSentiments)) * 0.9)])
	print "Prediction accuracy score by DecisionTreeClassifier: ", clf5.score(X[int(float(len(listOfReviews)) * 0.9):], Y[int(float(len(listOfSentiments)) * 0.9):])

	print "RandomForestClassifier training started..."
	clf6 = RandomForestClassifier(min_samples_split = 40).fit(X[:int(float(len(listOfReviews)) * 0.9)], Y[:int(float(len(listOfSentiments)) * 0.9)])
	print "Prediction accuracy score by RandomForestClassifier: ", clf6.score(X[int(float(len(listOfReviews)) * 0.9):], Y[int(float(len(listOfSentiments)) * 0.9):])

	print "AdaBoostClassifier training started..."
	clf7 = AdaBoostClassifier().fit(X[:int(float(len(listOfReviews)) * 0.9)], Y[:int(float(len(listOfSentiments)) * 0.9)])
	print "Prediction accuracy score by AdaBoostClassifier: ", clf7.score(X[int(float(len(listOfReviews)) * 0.9):], Y[int(float(len(listOfSentiments)) * 0.9):])
	