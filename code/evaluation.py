import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
class LinkPrediction():
    def __init__(self,test_file):
        self.links = [[], [], []]
        self.test_file=test_file
        sufs = ['_0', '_50', '_100']
        for i, suf in enumerate(sufs):
            with open(self.test_file + suf) as infile:
                for line in infile.readlines():
                    s, t, label = [int(item) for item in line.strip().split()]
                    self.links[i].append([s, t, label])

    def evaluate(self, embedding_matrix):
        test_y = [[], [], []]
        pred_y = [[], [], []]
        pred_label = [[], [], []]
        for i in range(len(self.links)):
            for s, t, label in self.links[i]:
                test_y[i].append(label)
                pred_y[i].append(embedding_matrix[0][s].dot(embedding_matrix[1][t]))
                if pred_y[i][-1] >= 0:
                    pred_label[i].append(1)
                else:
                    pred_label[i].append(0)

        auc = [0, 0, 0]
        for i in range(len(test_y)):
            auc[i] = roc_auc_score(test_y[i], pred_y[i])
        return auc


def node_classfication(embedding_matrix):
    cat_matrix=np.concatenate([embedding_matrix[0],embedding_matrix[1]],axis=1)
    data=[]
    label=[]
    train = []
    train_label = []
    test = []
    test_label = []
    with open('./node.txt') as file:
        for lines in file.readlines():
            line = lines.split()
            data.append(cat_matrix[int(line[0])])
            label.append(int(line[1]))

    train, test, train_label, test_label = train_test_split(data, label, test_size=0.3)


    train = np.array(train)
    train_label = np.array(train_label)
    test = np.array(test)
    test_label = np.array(test_label)

    mlr = OneVsRestClassifier(LogisticRegression(max_iter=10000), n_jobs=-1)
    mlr.fit(train, train_label)
    predict = mlr.predict(test)
    micro = f1_score(test_label, predict, average='micro')
    macro = f1_score(test_label, predict, average='macro')
    return micro, macro

def gr(embedding):
    node = []
    with open('./gr_node.txt')as file:
        for lines in file.readlines():
            line = lines.split()
            node.append(int(line[0]))
    embedding = np.array(embedding)
    k = [1, 2, 5, 10, 50, 100, 200]
    k_max = max(k)
    rank = []
    for j in node:
        id = []

        for i in range(embedding[0].shape[0]):
            id.append(i)
        pre = embedding[0][j].dot(embedding[1].T)
        pre = list(pre)
        dic1 = dict(zip(id, pre))

        dic1 = sorted(dic1, key=dic1.get, reverse=True)

        rank.append(dic1[0:k_max])
    graph = [{}, {}]
    with open('./edge.txt') as infile:
        for line in infile.readlines():
            source_node, target_node = line.strip().split(' ')
            source_node = int(source_node)
            target_node = int(target_node)

            if source_node not in graph[0]:
                graph[0][source_node] = []
            if target_node not in graph[1]:
                graph[1][target_node] = []

            graph[0][source_node].append(target_node)
            graph[1][target_node].append(source_node)
    dic2=dict(zip(node,rank))
    total_percision=[]
    for topk in k:
        percision=[]
        for j in dic2.keys():
            count=0
            for i in dic2.get(j)[0:topk]:
                if i in graph[0].get(j):
                    count+=1
            percision.append(count/topk)
        total_percision.append(np.mean(percision))

    return total_percision