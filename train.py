from gensim import corpora,models
import jieba,os
import jieba.posseg as pseg
import time
import sys
reload(sys)
sys.setdefaultencoding('utf-8')


#rootdir = 'data'
#docsdir = rootdir + '\docs'
#corpusdir = rootdir + '\corpus'
#dicsdir = rootdir + '\dic'
#stopwords = {}.fromkeys([line.rstrip() for line in open(dicsdir +'\stopwords.txt')])


def train_corpus(ntopics, docsdir, corpusdir, dicsdir):
    stopwords = {}.fromkeys([line.rstrip() for line in open(dicsdir +'\stopwords.txt')])
    train_set = []
    walk = os.walk(docsdir)
    flagset=['a','t','s','f','v','b','z','r','m','q','d','p','c','u','e','y','o','h','k','x','w','i']
    for root, dirs, files in walk:
        for name in files:
            file = open(os.path.join(root,name),'r')
            raw = pseg.cut(file.read())
            word_list = []
            for item in raw:
                word_u = item.word.encode('utf-8')
                if (item.flag[0] not in flagset or item.flag == 'vn') and len(item.word) > 1 and word_u not in stopwords:
                    word_list.append(item.word)
#            raw_list = list(jieba.cut(raw, cut_all = False))
#            word_list = []
#            for word in raw_list:
#                word_u = word.encode('utf-8')
#                if word_u not in stopwords:
#                    word_list.append(word)
#            train_set.append(word_list)
            train_set.append(word_list)
            file.close
            
    dic = corpora.Dictionary(train_set)
    corpus = [dic.doc2bow(text) for text in train_set]
    tfidf = models.TfidfModel(corpus)
    corpus_tfidf = tfidf[corpus]
    lda = models.LdaModel(corpus_tfidf, id2word =dic, num_topics = ntopics)
    corpus_lda = lda[corpus]
    corpora.MmCorpus.serialize(corpusdir + '\corpus.mm', corpus)
    lda.save(corpusdir + '\lda.txt')
    dic.save(corpusdir + '\word2id.dict')
    
if __name__ == "__main__":
    args = sys.argv
    train_corpus(int(args[1]), args[2], args[3], args[4])