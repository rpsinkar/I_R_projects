import numpy as np
import sys
from scipy import spatial  #for calculating cosine similarity

word_vect={}        #for storing wordvector
no_to_query={}      #for mapping from query number to query
query_to_number={}  #for mapping from query to query number
query_to_queryVector={} #for storing query vector of each query
expanded_query={}       #for storing expanded query for each query

#this function is used for finding the query vector
#it is alled in the main function
def find_query_vector():
    print 'finding query..'
    for query in query_to_number.keys():

        query_words=query.split()
        query_vector=np.zeros(shape=(1,300))
        for word in query_words:
            if word in word_vect:
                query_vector=query_vector+word_vect[word]
        query_to_queryVector[query]=query_vector


#This function is used for finding the expanded query
#It is caled in teh main function
def expand_query():
    for query in query_to_number.keys():
        print 'query......'
        word_list={}
        for key,val in word_vect.iteritems():
            consine= 1 - spatial.distance.cosine(query_to_queryVector[query].tolist(), val.tolist())
            if len(word_list)<5:
                word_list[key]=consine
            else:
                for k,v in word_list.iteritems():
                    if consine>v:
                        word_list.pop(k)
                        word_list[key]=consine
                        break
        expanded_query[query_to_number[query]]=query+" ".join(word_list)


#This is the main function for calculating teh query vector and reading values from file
#It computes the expanded query and writes them onto a file
if __name__=="__main__":
    count=0

    #loading the data from the word_vector file
    print 'loading data.......'
    with open('/home/ashwin/Downloads/glove.840B.300d.txt','r') as file:
        for line in file:
            line=line.strip()
            tok=line.split(' ',1)
            list1=tok[1].split()
            x=np.array(list1)
            word_vect[tok[0]]=x.astype(np.float)
            count+=1
            sys.stdout.write(str(count)+'\r')
            sys.stdout.flush()
            if count==100000:
                break
    print 'loading done.......'

    #maintaing the word to number thing
    with open('files/query.txt', 'r') as queryfile:
        for line in queryfile:
            line = line.strip()
            tok = line.split(' ', 1)
            print tok
            no_to_query[tok[0]] = tok[1]
            query_to_number[tok[1]] = tok[0]

    find_query_vector()
    expand_query()

    #writing the expanded query onto the file
    with open('expanded_query.txt','w') as file2:
        for key in sorted(expanded_query.keys()):
            file2.write(str(key)+'\t'+expanded_query[key]+'\n')

    #writing the query vectors into the file
    with open('queryvector.txt','w') as file3:
        for key in sorted(query_to_queryVector.keys()):

            file3.write(str(query_to_number[key])+'\t'+str(query_to_queryVector[key])+'\n')
