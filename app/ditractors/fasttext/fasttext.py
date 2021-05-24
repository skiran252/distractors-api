from nltk.stem import LancasterStemmer
import gensim.downloader
import psutil
import time



print("FASTTEXT LOAD START:",psutil.virtual_memory().used/(1024*1024),"MB")
start = time.time()
glove_vectors = gensim.downloader.load('fasttext-wiki-news-subwords-300')
print("FASTTEXT LOAD END:",time.time()-start,"MEM:",psutil.virtual_memory().used/(1024*1024),"MB")


def Extract(lst): 
    return [item[0] for item in lst]
def Lematisation_Check(answer,options,Rl):
    lancaster=LancasterStemmer()
    l=0
    # options=Extract(options)    
    #options=RemovePunc(options)
    rl=[]
    lemAns=lancaster.stem(answer)
    for i in options:
      if lemAns!=lancaster.stem(i[0]) and l<Rl:
          rl.append(i)
          l+=1 
    return rl
def Lematisation_Initial_Check(options,answer):
    rl=[]
    lancaster=LancasterStemmer()
    lemAns=lancaster.stem(answer)
    for i in options:
      if lemAns!=lancaster.stem(i) and similar(answer,i)<0.6:
              rl.append(i)
    return rl