from pymongo import MongoClient
import nltk
import sklearn
import glob
import os

if __name__ == "__main__":
	client = MongoClient()
	dBase = client.movieReviews
	MovRevs = dBase.MovRevs
	MovRevs.remove({})
	count = 0
	negDir = "C:/Education/FutureDatasets/Labelled_Movie_Reviews/neg/"
	posDir = "C:/Education/FutureDatasets/Labelled_Movie_Reviews/pos/"
	os.chdir(negDir)
	listOfFiles = glob.glob("./*.txt")

	for f in listOfFiles:
		count += 1
		with open(f, 'r') as currentFile:
			MovRevs.insert({'revID': count, 'text': currentFile.read().replace('\n',' ').replace('\r', ' '), 'sentiment': 0})

	os.chdir(posDir)
	listOfFiles = glob.glob("./*.txt")

	for f in listOfFiles:
		count += 1
		with open(f, 'r') as currentFile:
			MovRevs.insert({'revID': count, 'text': currentFile.read().replace('\n',' ').replace('\r', ' '), 'sentiment': 1})
	
	flag = 0

	for item in MovRevs.find():
		if flag > 2:
			break
		print item
		flag += 1
	print MovRevs.count()
	print len(MovRevs.distinct('revID'))