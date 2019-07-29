''' Name: as5.py
Creator: Ahwan Mishra (ahwan100@gmail.com)
Date: September 2, 2018
Description: In this file text-summarization algorithms are implemented.
'''

''' N.B.  This program gives different ouputs when we use different functions. For word spiltng when we use 
nltk.word_tokenize everywhere it gives different answer than split(" "). I have used nltk.word_tokenize in 
most of the places because it gives more meaningful outputs, but not in some places where it is called many times 
because nltk.word_tokenize is comparatively slower and makes it run slow. So I have only used it wherever it is 
not called many times.
'''

import nltk
import numpy
import networkx
import string
import math

from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.metrics.pairwise import cosine_similarity

lemmatizer = WordNetLemmatizer()


import codecs

#Part1
#Reading the file
with codecs.open('input.txt', encoding='utf-8') as myfile:
	data=myfile.read().replace('\n',' ')

#Segmenting the document into sentences
a=nltk.sent_tokenize(data)
b=a[:]

#Preprocess each sentences
for i in range(len(a)):
	a[i]=a[i].replace(',','')
for i in range(len(a)):
	a[i]=a[i].lower()

for i in range(len(a)):
	a[i]=a[i].translate(string.punctuation)

stop_words = set(stopwords.words('english'))

for i in range(len(a)):
	temp=""
	for w in nltk.word_tokenize(a[i]): 
		if w not in stop_words:
			temp=temp+w+" "
	a[i]=temp

for i in range(len(a)):
	temp=""
	for w in nltk.word_tokenize(a[i]):
		w=lemmatizer.lemmatize(w,'v')
		temp=temp+w+" "
	a[i]=temp

#Part2:  1
def TF(w,s):
	count=0
	for w1 in s:
		if(w1==w):
			count=count+1
	return count


def IDF(w,a):
	t=len(a)
	count=0
	for i in range(t):
		if w in a[i]:  
		
			count=count+1
	if(count==0):
		return 0.0
	ans=math.log10(t/count)
	return ans
	
	
def TF_IDF(w,s,a):
	return TF(w,s)*IDF(w,a)


#Part2:  2
str1=""
for i in range(len(a)):
	str1=str1+a[i]+" "


#Removing duplicate words
str2=str1
str1=""
for w in nltk.word_tokenize(str2):
	if w not in str1:
		str1=str1+w+" "




words_tokenized=nltk.word_tokenize(str1)
matrix = [ [ None for j in range(len(a))] for i in range(len(words_tokenized))] 



#Filling the table with TF_IDF values
for i in range(len(words_tokenized)):
	for j in range(len(a)):
		matrix[i][j]=TF_IDF(words_tokenized[i],a[j],a)

		
		
#Part 3: Implementing matrix-based summarization algorithm			
u, s, vh = numpy.linalg.svd(matrix, full_matrices=True)
result_list=[0 for i in range(15)]
c=-1
for i in range(5):
	top_3_idx = numpy.argsort(vh[i])[-3:]
	result_list[c]=top_3_idx[0]
	c+=1
	result_list[c]=top_3_idx[1]
	c+=1
	result_list[c]=top_3_idx[2]
	c+=1
result_list.sort()
new_result_list=[]

#Removing duplicates if exists
for i in result_list:
	if i not in new_result_list:
		new_result_list.append(i)
print ("\n\n== Output of Matrix based summarization ==\n\n")


for i in range(len(new_result_list)):
	print (b[new_result_list[i]])

print ("\n\n\n")


#Part 4: Implementing graph-based summarization algorithm
G=networkx.Graph()
for i in range(len(a)):
	G.add_node(i)

matrix2 = [ [ None for j in range(len(words_tokenized))] for i in range(len(a))] 

for i in range(len(words_tokenized)):
	for j in range(len(a)):
		matrix2[j][i]=matrix[i][j]

for i in range(len(a)):
	for j in range(len(a)):
		 G.add_edge(i,j,weight=cosine_similarity(numpy.asmatrix(matrix2[i]),numpy.asmatrix(matrix2[j])))
		 
p_rank=networkx.pagerank(G)
p_rank_arr=[None for i in range(len(p_rank))]
for i in range(len(p_rank)):
	p_rank_arr[i]=float(p_rank[i])
	
top_15_idx = numpy.argsort(p_rank_arr)[-15:]
new_result_list=[]

#Removing duplicates if exists
for i in top_15_idx:
	if i not in new_result_list:
		new_result_list.append(i)
new_result_list.sort()

print ("== Output of Graph based summarization ==")
print ("\n\n\n")
for i in range(len(new_result_list)):
	print (b[new_result_list[i]])

print ("\n\n")
