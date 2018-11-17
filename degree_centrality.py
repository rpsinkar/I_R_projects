import os
import re
from nltk.corpus import stopwords as sw
from nltk import RegexpTokenizer
import string
from math import sqrt,log
import networkx as nx
from pyrouge import Rouge155
import shutil



idf = dict() # for storing the idf of words
sentences = dict() # for storing sentences in topic
tf_idf = dict() # for storing tf-idf vector for each senctence
wordsOfSentence = dict() # for storing the Set of words in each sentence 


if os.path.isdir('./degreeCen_Summaries'):
   shutil.rmtree('./degreeCen_Summaries')

os.makedirs('./degreeCen_Summaries')

rougePath = '/home/roshan/IR/RELEASE-1.5.5'

# tokenizer
tk = RegexpTokenizer(r'\w+')
stopwords = sw.words('english')
path = '/home/roshan/IR/Assignement2_IR/'




# acessing  topics folders list
topics = [d for d in os.listdir(path) if d.startswith('Topic')]
topics.sort()

# calculating  document frequency
def calculate_df(word):
    n = 0
    for s in sentences:
        if sentences[s].find(word) >= 0:
            n+=1
    return n


# calculate tf_idf vector for each sentece
def tfidf_Calculation(line, index):
    uwords = set()
    for word in tk.tokenize(line):
        if word not in stopwords and word not in string.punctuation:
            uwords.add(word)
            if word not in idf:
                idf[word] = log(len(sentences)/calculate_df(word))

    tf_idf[index] = [ (1+log(sentences[index].count(w))) * idf[w] for w in uwords]
    wordsOfSentence[index] = uwords




def cosine_similarity(i, j):

    words1 = wordsOfSentence[i]
    words2 = wordsOfSentence[j]

    commomWords = words1 & words2


    n = 0
    for w in commomWords:
        n+= ( (1 + log(sentences[i].count(w))) * (1 + log(sentences[j].count(w))) * (idf[w]**2))

    if n == 0:
        return 0

    p1 = 0

    for v in tf_idf[i]:
        p1 += v ** 2

    p2 = 0
    for v in tf_idf[j]:
        p2 += v ** 2

    p = sqrt(p1 * p2)

    similarity = n/ p

    return similarity

def Generating_deg_centr_summaries(graph, topic):

    summary = []

    fp = open('./degreeCen_Summaries/' + topic + '_.txt', 'w')

    No_of_words = 0

    highdegree_node = 0
    highest_deg= 0

    for i in graph.nodes:
        if graph.degree(i) > highest_deg:
            highdegree_node = i
            highest_deg= graph.degree(i)

    while(No_of_words <250):

        if graph.number_of_edges()==0:
            break

        summary.append(highdegree_node)
        No_of_words += len(tk.tokenize(sentences[highdegree_node]))

        for e in list(graph.neighbors(highdegree_node)):
            graph.remove_node(e)

        graph.remove_node(highdegree_node)

        highest_deg= 0

        for i in graph.nodes:
            if graph.degree(i) > highest_deg:
                highdegree_node = i
                highest_deg= graph.degree(i)


    fp.write('\n'.join(sentences[i] for i in summary))

    fp.close()







threshold = 0.3






def Rouge_score(t):

    print 'Getting Rouge Score {}'.format(t),'\n'
    r = Rouge155(rouge_dir=rougePath)
    r.system_dir = '/home/roshan/IR/Assignement2_IR/degreeCen_Summaries/'
    r.model_dir = '/home/roshan/IR/Assignement2_IR/GroundTruth/'
    tn = re.findall(r'\d+',t)[0]
    r.system_filename_pattern = 'Topic('+ tn +')_.txt'
    r.model_filename_pattern = t+'.1'


    output = r.convert_and_evaluate()
    print output,'\n\n'





def construct_graph(sentences, threshold):

    graph = nx.Graph()

    graph.add_nodes_from(xrange(len(sentences)))

    print graph.number_of_nodes()

    for i in range(0, len(sentences) - 1):
        for j in range(i + 1, len(sentences)):
            wt = cosine_similarity(i, j)

            if wt >= threshold:
                graph.add_edge(i,j)

    print graph.number_of_edges()

    return graph


if __name__ == '__main__':
    for t in topics:
        # acessing docs in topic
        docs = os.listdir(path + t)
        doc_paths = list()
        index = 0

        for d in docs:
            tempPath = path + t + '/' + d
            doc_paths.append(tempPath)

        for d in doc_paths:
            with open(d) as f:
                for item in f.read().split('</P>'):
                    if "<P>" in item:
                        
                        line = item[item.find('<P>') + 3:]
                        line = line.replace('\n', ' ').strip()

                        if len(line.translate(None, string.punctuation)) == 0:
                            continue

                        if len(line.split(' ')) < 2:
                            continue

                        sentences[index] = line
                        tfidf_Calculation(line,index)
                        index += 1

        #  Graph construction
        graph = construct_graph(sentences , threshold)

        Generating_deg_centr_summaries(graph, t)

        tf_idf.clear()
        idf.clear()
        sentences.clear()
        wordsOfSentence.clear()

        Rouge_score(t)