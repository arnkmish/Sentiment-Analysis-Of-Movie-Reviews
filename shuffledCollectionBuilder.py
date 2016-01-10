from pymongo import MongoClient
import random

if __name__ == "__main__":
	client = MongoClient()
	dBase = client.movieReviews

	MovRevs = dBase.MovRevs

	shuffledMovRevs = dBase.shuffledMovRevs

	shuffledMovRevs.remove({})

	print MovRevs.count()
	countList = [i for i in range(1,2001)]

	print MovRevs.find_one()

	
	while len(countList) != 0:
		countID = random.choice(countList)
		for item in MovRevs.find({'revID': countID}):
			shuffledMovRevs.insert(item)
		countList.remove(countID)

	print shuffledMovRevs.count()
	print len(shuffledMovRevs.distinct("revID"))