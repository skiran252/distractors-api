import psutil
import time
import gensim.downloader
import os
from nltk.stem import LancasterStemmer

process = psutil.Process(os.getpid())
print("FASTTEXT LOAD START:",psutil.virtual_memory().used/(1024*1024),"MB")
start = time.time()
glove_vectors = gensim.downloader.load('fasttext-wiki-news-subwords-300')
print("FASTTEXT LOAD END:",time.time()-start,"MEM:",psutil.virtual_memory().used/(1024*1024),"MB")

def Lematisation_Check_with_probs(answer,options,Rl):
    lancaster=LancasterStemmer()
    l=0
    rl=[]
    lemAns=lancaster.stem(answer)
    for i in options:
      print(i,lemAns)
      if lemAns!=lancaster.stem(i[0]) and l<Rl:
          rl.append(lancaster.stem(i))
          l+=1
    return rl

def get_distractors(answer: str,rem:int = 0):
    try:
        fTextOp=glove_vectors.most_similar(answer.split())
        return fTextOp
    except:
        return []

get_distractors("rust",3)