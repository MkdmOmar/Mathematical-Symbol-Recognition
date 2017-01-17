import csv
import sys

top25 = open("top25.txt","r")

for ID in top25:
	with open(ID + 'final.csv', 'wb') as csvfile:
	    mywriter = csv.writer(csvfile, delimiter=',')
	    dicts = open(ID[:-1] + '.text', 'r')
	    i = 0
	    for line in dicts:
	     	if (i == 0): #write out header
	     		mywriter.writerow([line])
	     		i = 1
	     	else:
	     		matrix = eval(line)
	     		for stroke in matrix:
	     			for dictionary in stroke:
	     				a = dictionary["x"]
	     				b = dictionary["y"]
	     				c = dictionary["time"]
	     				mywriter.writerow([a,b,c])
	     			mywriter.writerow(["-"])	
	     		mywriter.writerow([";"])


	   