'''
please set path accordingly to rouge and documents

you should  set threshold manually in the code 
'''

import os
from nltk.corpus import stopwords as sw
from nltk import RegexpTokenizer
import string
from math import sqrt, log10,log
import networkx as nx
from pyrouge import Rouge155
import shutil
import re




if os.path.isdir('./Textrank_summaries'):
    shutil.rmtree('./Textrank_summaries')

os.makedirs('./Textrank_summaries')

#----Update these paths accordingly
rougePath = '/home/roshan/IR/RELEASE-1.5.5'
path = '/home/roshan/IR/Assignement2_IR/'


# tokenizer
tk = RegexpTokenizer(r'\w+')
stopwords = sw.words('english')


idf = dict()
sentences = dict()
tf_idf = dict()
wordsOfSentences = dict()
text_Ranks = list()

# acessing all topics folders list
topics = [d for d in os.listdir(path) if d.startswith('Topic')]
topics.sort()


# Calculating document frequency
def calculate_df(word):
    n = 0
    for s in sentences:
        if sentences[s].find(word) >= 0:
            n+=1
    return n


# Calcuate tf-idf vectors for each sentences
def tfidf_Calculation(line, index):
    uwords = set()
    for word in tk.tokenize(line):
        if word not in stopwords and word not in string.punctuation:
            uwords.add(word)
            if word not in idf:
                idf[word] = log10(len(sentences) / calculate_df(word))

    tf_idf[index] = [(1 + log10(sentences[index].count(w))) * idf[w] for w in uwords]
    wordsOfSentences[index] = uwords


# returns the similarity b/w two sentences 
def cosine_similarity(i, j):

    words1 = wordsOfSentences[i]
    words2 = wordsOfSentences[j]

    commomWords = words1 & words2


    n = 0
    for w in commomWords:
        n+= ( (1 + log10(sentences[i].count(w))) * (1 + log10(sentences[j].count(w))) * (idf[w]**2))

    if n == 0:
        return 0

    p1 = 0

    for v in tf_idf[i]:
        p1 += v ** 2

    p2 = 0
    for v in tf_idf[j]:
        p2 += v ** 2

    p = sqrt(p1 * p2)

    sim_score = n/ p

    return sim_score


# Alpha = 0.85 and Max Iterations are 100 and Tolerance is 0.01 
def Textrank(graph):
    count = 0
    nodes = graph.number_of_nodes()
    textranks = dict()
    prevtextranks = dict()
    # initializing textranks to 1
    for i in range(0, nodes):
        textranks[i] = 1
        prevtextranks[i] = 1

    flag = True
    iter = 0
    while (flag):
        iter += 1
        #print "iteration", iter
        flag2 = False
        for i in range(nodes - 1, -1, -1):
            key = i
            finalSum = 0.0
            for v in graph.in_edges(key):

                temp = textranks[v[0]]
                sum_weights = 0
                flag2 = True
                for e in graph.out_edges(v[0]):
                    sum_weights = sum_weights + graph.get_edge_data(e[0], e[1])['weight']

                if sum_weights != 0:
                    finalSum = finalSum + graph.get_edge_data(v[0], key)['weight'] * (float(temp) / sum_weights)

            prevtextranks[key] = textranks[key]
            textranks[key] = 0.15 + 0.85 * finalSum

        count = count + 1

        if flag2:
            for i in range(0, nodes):
                if (abs(prevtextranks[i] - textranks[i]) > 0.01):
                    flag2 = False
                    break
        if flag2:
            flag = False

        elif (count > 100):
            flag = False

    return (sorted(textranks.items(), key=lambda x: -x[1]))


def Generating_textrank_summary(topic):
    summary = list()
    noOfWords = 0
    for i in range(len(text_Ranks)):
        sentIndex = text_Ranks[i][0]
        summary.append(sentIndex)
        noOfWords += len(tk.tokenize(sentences[sentIndex]))

        if noOfWords > 250:
            break

    fp = open('./Textrank_summaries/' +topic + '_.txt', 'w')

    for s in summary:
        fp.write(sentences[s] + '\n')
    fp.close()



# Calculate Rouge Score for given topic t
def Rouge_score(t):

    print 'Rouge Score for topic {}'.format(t),'\n'
    r = Rouge155(rouge_dir=rougePath)
    r.system_dir = '/home/roshan/IR/Assignement2_IR/Textrank_summaries/'
    r.model_dir = '/home/roshan/IR/Assignement2_IR/GroundTruth/'
    tn = re.findall(r'\d+',t)[0]
    r.system_filename_pattern = 'Topic('+ tn +')_.txt'
    r.model_filename_pattern = t+'.1'


    output = r.convert_and_evaluate()
    print output,'\n\n'



threshold = 0.3


# construction of Graph using networkx
def contruct_graph(sentences, threshold):

    graph = nx.DiGraph()

    graph.add_nodes_from(xrange(len(sentences)))

    for i in range(0, len(sentences) - 1):
        for j in range(i + 1, len(sentences)):
            wt = cosine_similarity(i, j)
            #print i,j, wt

            if wt >= threshold:
                graph.add_edge(i,j, weight = wt)
                graph.add_edge(j,i,weight = wt)

    global text_Ranks
    text_Ranks = Textrank(graph)


#Main Function

if __name__ == '__main__':
    for t in topics:
        # acessing docs in topic
        docs = os.listdir(path + t)
        docPaths = list()
        index = 0

        for d in docs:
            tempPath = path + t + '/' + d
            docPaths.append(tempPath)

        for d in docPaths:
            with open(d) as f:
                for item in f.read().split('</P>'):
                    if "<P>" in item:
                        # 3 is len of tag <P>
                        line = item[item.find('<P>') + 3:]
                        line = line.replace('\n', ' ').strip()

                        if len(line.translate(None, string.punctuation)) == 0:
                            continue

                        if len(line.split(' ')) < 2:
                            continue

                        sentences[index] = line
                        tfidf_Calculation(line,index)
                        index += 1



        # create graph
        contruct_graph(sentences , threshold)

        #generate Summary
        Generating_textrank_summary(t)

        Rouge_score(t)

        tf_idf.clear()
        idf.clear()
        sentences.clear()
        wordsOfSentences.clear()
        text_Ranks = list()
