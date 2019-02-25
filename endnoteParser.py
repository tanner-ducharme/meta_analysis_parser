from collections import Counter
import pickle
import re
import string


import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.util import ngrams
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import xml.etree.ElementTree as ET



def stemmer(text):
    ps = PorterStemmer()
    stemmed_text = []
    for word in text:
        stemmed_text.append(ps.stem(word))
    return stemmed_text


def lemmatizer(text):
    lz = WordNetLemmatizer()
    lemma_text = []
    for word in text:
        
        lemma_text.append(lz.lemmatize(word))
    return lemma_text

def cleanText(text):    
    stopWords = set(stopwords.words('english'))
    words = text.split()    
    words = [word.lower() for word in words]  

    # remove prevent '-' from being removed when removing punctuation    
    puncList = list(string.punctuation)
    puncList.remove('-')
    punc = ""
    for p in puncList:
        punc += p
    
    table  = str.maketrans('', '', punc)
    words = [word.translate(table) for word in words]    
        
    cleanedWords = []
    for word in words:
        if word in stopWords:
            pass
        else:
            cleanedWords.append(word)
            
            
#   cleanedWords = lemmatizer(cleanedWords)
    cleanedWords = stemmer(cleanedWords)
    
    return cleanedWords



# NOTE: should we stem or lemmatize keywords?

def cleanKeyWords(keywords):
    
    cleanedWords = []
    for keyword in keywords:
        
        # splits cases like  '*Actins/me [Metabolism]' into just 'Actins'
        keyword = keyword.split('/')
        keyword  = keyword[0] 

        cleanedWords.append(keyword)
        
    # lowercase, remove punc
    cleanedWords = [word.lower() for word in cleanedWords]
    
    puncList = list(string.punctuation)
    puncList.remove('-')
    punc = ""
    for p in puncList:
        punc += p
    
    table  = str.maketrans('', '', punc)
    cleanedWords = [word.translate(table) for word in cleanedWords]
    
    
    # removes keywords that contain numbers for simplicity
    final_set = []
    for word in cleanedWords:
        
        contains_digit = False
        for char in word:
                        
            if char.isdigit():
                contains_digit = True
        if not contains_digit:
            final_set.append(word)
            
    
    # removes duplicates
    
    final_set = list(set(final_set))                     
    return final_set
        
def getENLdata(root, relevant):
    data = []   

    for record in root.iter('record'): # iterate through all records
        entry = []
        
        try: # capture title if present
            titles = record.find('titles')
            title = titles.find('title')
            title_text = title.find('style').text
            entry.append(title_text)
            entry.append(cleanText(title_text))

        except AttributeError:
            entry.append('')
            entry.append('')



        
        try: # capture keywords if present
            keywords = record.find('keywords')
            word_list = []
            for word in keywords:
                word_text = word.find('style').text
                word_list.append(word_text)    
            entry.append(word_list)
            entry.append(cleanKeyWords(word_list))            

        except TypeError:
            entry.append([])       
            entry.append([])
            
        
        try: # capture abstract if present            
            abstract = record.find('abstract')
            abstract_text = abstract.find('style').text
            entry.append(abstract_text)
            entry.append(cleanText(abstract_text))
        
        except AttributeError:
            entry.append([])      
            entry.append([])
            

        entry.append(relevant)        
        data.append(entry)
        
    return data


def nGrammer(corpus, n):
    # NOTE: automatically includes uni-grams for now
    ''' corpus: string
        n: list of ngram options
            e.g. n = [2,3] will return a list consisting of the uni, bi
            and trigrams found in corpus
    '''
    grams = []
    for ng in n:
        n_gram = ngrams(corpus.split(), ng)
        for gram in n_gram:
            gram = gram[0] + " " + gram[1]
            grams.append(gram)
    return grams
        
def titleCorpus(df):
    titles_list = df['title'].values.tolist() # gets list of all title lists
    titles = [' '.join(title) for title in titles_list] # converts each title from list of words to string
    title_corpus = []
    for title in titles:
        grams = nGrammer(title, [2])
        title_corpus.append(title.split() + grams)
    return title_corpus, titles
        


def abstractCorpus(df):
    abstracts_list = df['abstract'].values.tolist()
    abstracts = [' '.join(title) for title in abstracts_list]
    abstract_corpus = []
    for abstract in abstracts:
        grams = nGrammer(abstract, [2])
        abstract_corpus.append(abstract.split() + grams)
    return abstract_corpus, abstracts

        
        
        


def keywordCorpus(df):
    keyword_corpus = df['keywords'].values.tolist()
    keywords = [' '.join(title) for title in keyword_corpus]

    return keyword_corpus, keywords
    



#     all_paper_text = []
#     for index, rec in enumerate(title_corpus):
#         entry_text = rec + keyword_corpus[index] + abstract_corpus[index]
#         all_paper_text.append(" ".join(entry_text))
    
def allTextCorpus(title_corpus, abstract_corpus, keyword_corpus):

    all_text_corpus = []

    for index, title in enumerate(title_corpus):
        corpus = title + keyword_corpus[index] + abstract_corpus[index]
        corpus = list(set(corpus))
        all_text_corpus.append(corpus)
        
        
    all_papers_text = [" ".join(paper_text) for paper_text in all_text_corpus]
    return all_text_corpus, all_papers_text


def createVocab(all_text_corpus):


    # each entry will have uni and bigrams from a paper's title, keywords, and abstract
    flat_corpus = [item for sublist in all_text_corpus for item in sublist]    
    
    vocab = [word for word in flat_corpus if len(word) > 1]
    return vocab

# CORPUS IS NOW LIST OF ALL UNI AND BIGRAMS THAT APPEAR IN THE TITLES, KEYWORDS, AND ABSTRACTS OF ALL PAPERS
# CORPUS CAN NOW BE USED AS THE VOCABULARY
    
# custom method for getting enforcing max_features on our corpus
# necessary because we're using custom vocab

def nMostCommon(term_list, n):
    '''
        term_list: list of ngrams
        n: number of most common terms to return
        e.g. n = 1000 returns list of 1000 most common terms in corpus        
    
    '''
    count = Counter(term_list)    
    highest_freq_terms = count.most_common(n)
    
    # get just the txt
    highest_freq_terms = [term[0] for term in highest_freq_terms]
    return highest_freq_terms
 
        
def genIDs(num):
    index = list(np.arange(1,num+1, 1))
    max_id_length = (len(str(index[-1])))
    IDs = []
    for ID in index:

    #     print(max_id_length - len(str(ID)))
        prepend = ''
        for index in range(max_id_length - len(str(ID))):
            prepend += '0'
    #     print(prepend)
        IDs.append(prepend+str(ID))
    return IDs



def genDF(initialXML, finalXML): # 'initialLibrary.xml' , 'finalLibrary.xml'
    finaltree = ET.parse(finalXML)
    finalroot = finaltree.getroot()
    initialtree = ET.parse(initialXML)
    initialroot = initialtree.getroot()
    
    
    paper_df = pd.DataFrame(data = (getENLdata(finalroot, True)+getENLdata(initialroot, False)),
                       columns = ['original title', 'title', 'original keywords','keywords','original abstract', 'abstract', 'relevant'])
    
    title_corpus, titles = titleCorpus(paper_df)
    abstract_corpus, abstracts = abstractCorpus(paper_df)
    keyword_corpus, keywords = keywordCorpus(paper_df)
    
    
    all_text_corpus, all_papers_text = allTextCorpus(title_corpus, abstract_corpus, keyword_corpus)
    
    vocab = createVocab(all_text_corpus)
    
    
    vect = CountVectorizer(binary = True, vocabulary = nMostCommon(vocab, 10000))

    title_vectors = vect.fit_transform(titles)
    abstract_vectors = vect.fit_transform(abstracts)
    keyword_vectors = vect.fit_transform(keywords)
    combined_text_vectors = vect.fit_transform(all_papers_text)
    
    
    
    paper_df = paper_df.assign(title_corpus = title_corpus)
    paper_df = paper_df.assign(keyword_corpus = keyword_corpus)
    paper_df = paper_df.assign(abstract_corpus = abstract_corpus)
    paper_df = paper_df.assign(all_text_corpus = all_text_corpus)
    paper_df = paper_df.assign(all_text = all_papers_text)
    
    paper_df.insert(1, 'paper_id', genIDs(len(paper_df)))
    
    
    title_vectors         = [title for title in title_vectors.toarray()]
    keyword_vectors       = [keywords for keywords in keyword_vectors.toarray()]
    abstract_vectors      = [abstract for abstract in abstract_vectors.toarray()]
    combined_text_vectors = [combined_text for combined_text in combined_text_vectors.toarray()]


    paper_df = paper_df.assign(title_vector = title_vectors)
    paper_df = paper_df.assign(keyword_vector = keyword_vectors)
    paper_df = paper_df.assign(abstract_vector = abstract_vectors)

    paper_df = paper_df.assign(combined_text_vector = combined_text_vectors)
    
    return paper_df


    
    

