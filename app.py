import logging
import os
import string
import time
from collections import OrderedDict
from difflib import SequenceMatcher
import json
import gensim.downloader
import psutil
import uvicorn
from fastapi import FastAPI, HTTPException
from nltk.stem import LancasterStemmer
from sense2vec import Sense2Vec
from starlette.middleware.cors import CORSMiddleware

process = psutil.Process(os.getpid())
print("FASTTEXT LOAD START:",psutil.virtual_memory().used/(1024*1024),"MB")
start = time.time()
glove_vectors = gensim.downloader.load('fasttext-wiki-news-subwords-300')
print("FASTTEXT LOAD END:",time.time()-start,"MEM:",psutil.virtual_memory().used/(1024*1024),"MB")
def Extract(lst): 
    return [item[0] for item in lst]
def Lematisation_Check(answer,options,Rl):
    lancaster=LancasterStemmer()
    l=0
    options=Extract(options)
    # options=RemovePunc(options)
    rl=[]
    lemAns=lancaster.stem(answer)
    for i in options:
      if lemAns!=lancaster.stem(i) and l<Rl:
          rl.append(lancaster.stem(i))
          l+=1 
    return rl

def Lematisation_Check_with_probs(answer,options,Rl):
    lancaster=LancasterStemmer()
    l=0
    rl=[]
    lemAns=lancaster.stem(answer)
    for i in options:
      print(lancaster.stem(i[0]))
      if lemAns!=lancaster.stem(i[0]) and l<Rl:
          rl.append(lancaster.stem(i))
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


print("LOADING SENSE2VEC MODEL","MEMORY USAGE BEFORE LOADING:",psutil.virtual_memory().used/(1024*1024),"MB")
start = time.time()
s2v = Sense2Vec().from_disk('/home/skira/distractors-api/s2v_reddit_2019_lg')
print("LOADED SENSE2VEC TOOK:", time.time()-start,"MEMORY USAGE:",psutil.virtual_memory().used/(1024*1024),"MB")

def edits(word):
    "All edits that are one edit away from `word`."
    letters = 'abcdefghijklmnopqrstuvwxyz '+string.punctuation
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    deletes = [L + R[1:] for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
    replaces = [L + c + R[1:] for L, R in splits if R for c in letters]
    inserts = [L + c + R for L, R in splits for c in letters]
    return set(deletes + transposes + replaces + inserts)

def sense2vec_get_words(word,s2v):
    output = []
    word_preprocessed =  word.translate(word.maketrans("","", string.punctuation))
    word_preprocessed = word_preprocessed.lower()
    word_edits = edits(word_preprocessed)
    word = word.replace(" ", "_")
    sense = s2v.get_best_sense(word)
    print(sense,' <------------------ sense')
    most_similar = s2v.most_similar(sense, n=50)
    compare_list = [word_preprocessed]
    for each_word in most_similar:
        append_word = each_word[0].split("|")[0].replace("_", " ")
        append_word = append_word.strip()
        append_word_processed = append_word.lower()
        append_word_processed = append_word_processed.translate(append_word_processed.maketrans("","", string.punctuation))
        if append_word_processed not in compare_list and word_preprocessed not in append_word_processed and append_word_processed not in word_edits:
            output.append(append_word.title())
            compare_list.append(append_word_processed)
    out = list(OrderedDict.fromkeys(output))
    return out

def get_options(answer):
    distractors = []
    try:
        distractors = sense2vec_get_words(answer, s2v)
        if len(distractors) > 0:
            print(" Sense2vec_distractors successful for word : ", answer)
            return distractors
    except:
        print(" Sense2vec_distractors failed for word : ", answer)
    return distractors

def get_application() -> FastAPI:
	application = FastAPI(title="Haystack-API", debug=True, version="0.1")

	application.add_middleware(
		CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
	)
	return application
app = get_application()

@app.post('/models/fasttext')
def get_distractors(answer: str,rem:int = 0,):
    try:
        fTextOp=glove_vectors.most_similar(answer.split())
        return Lematisation_Check(answer,fTextOp,rem)
    except:
        return []

@app.post('/models/sense2vec')
def get_sense_vectors(answer: str):
    options_generated = get_options(answer)
    maxi = 0
    result = []
    for opt in options_generated:
        if SequenceMatcher(None, answer.lower(), opt.lower()).ratio() < 0.4 :
            result.append(opt)
            maxi+=1
            if maxi ==3:
                break
        
    return result
@app.post('/models/sense2vec-test')
def get_sense_vectors(answer: str,n: int = 80):
    n = min(80,n)
    outputs = []
    word = answer
    word_preprocessed =  word.translate(word.maketrans("","", string.punctuation))
    word_preprocessed = word_preprocessed.lower()
    word_edits = edits(word_preprocessed)
    word = word.replace(" ", "_")
    sense = s2v.get_best_sense(word)
    print(sense,' <------------------ sense')
    most_similar = s2v.most_similar(sense, n=n)
    for out in most_similar:
        out = list(out)
        out[1] = str(out[1])
        outputs.append(out)
    return outputs,sense
@app.post('/models/fasttext-test')
def get_distractors(answer: str,rem:int = 0):
    topn = max(10,rem)
    try:
        fTextOp=glove_vectors.most_similar(answer.split(),topn=topn)
        for out in fTextOp:
            out = list(out)
            out[1] = str(out[1])
        return fTextOp
    except:
        return []
if __name__ == "__main__":
	uvicorn.run(app, host="0.0.0.0", port=8000)