import jieba.posseg as pseg
import jieba
from gensim import corpora,models
import time,numpy
import json
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

rootdir = 'data'
docsdir = rootdir + '\docs'
corpusdir = rootdir + '\corpus'

def words_probability_of_topic(lda_result, topicid):
    """
    Return a list of `(words_probability, word)` 2-tuples for the most probable
    words in topic `topicid`.

    Only return 2-tuples for the topn most probable words (ignore the rest).

    """
    topic = lda_result.state.get_lambda()[topicid]
    topic = topic / topic.sum() # normalize to probability dist
    beststr = {}
    for id in range(0,len(topic)):
        beststr[(lda_result.id2word[id])] = topic[id]
        #beststr .append((lda_result.id2word[id], topic[id]))
    return beststr


def tags(file, topicnum=10, tagnum=500, docsdir='data\docs', dicsdir='data\dics', corpusdir='data\corpus'): 
    s_tick1 = time.time()
    _lda = models.LdaModel.load(corpusdir + '\lda.txt', mmap = 'r')
    _dic = corpora.Dictionary.load(corpusdir + '\word2id.dict')
    e_tick1 = time.time()
    print (e_tick1-s_tick1)
    new_wordlist = []
    doc_string = open('C:\Users\Ivar\Desktop'+'\\'+file,'r').read()
    new_doc = jieba.cut(doc_string)
    #new_doc = jieba.cut(file)
    for word in new_doc:
        new_wordlist.append(word)
    new_vec = _dic.doc2bow(new_wordlist)
    doc_lda = _lda[new_vec]
    topics_probability = sorted(doc_lda, key = lambda t:t[1], reverse = True)
    
    doc_tags = {}
    for topic_id,topic_p in topics_probability[:topicnum]:
        s_tick2 = time.time()
        print topic_id,'=',topic_p
        words_probability = words_probability_of_topic(_lda, topic_id)
        for word in new_wordlist:
            if doc_tags.has_key(word):
                doc_tags[word] += topic_p*words_probability.get(word, 0)
            else:
                doc_tags[word] = topic_p*words_probability.get(word,0)
        e_tick2 = time.time()
        print (e_tick2 - s_tick2)
#        sort1 = sorted(doc_tags.items(), key = lambda e:e[1],reverse =True)
#        for item in sort1[:tagnum]:
#            if item[1]>0:
#                print item[0],'<=>',item[1]
#        print ''
    
    print '**********************************************************************'
    sort = sorted(doc_tags.items(), key = lambda e:e[1],reverse =True)
    tags = []
    for item in sort[:tagnum]:
        if item[1]>0:
            print item[0],'=>',item[1]
            tags.append(item)
    s = [{"tags": tags}]
    tag_json = json.dumps(s)
    return tag_json

if __name__ == "__main__":
    args = sys.argv
    file = args[1]
    topicnum = int(args[2])
    tagnum = int(args[3])
    docsdir = args[4]
    dicsdir = args[5]
    corpusdir = args[6]
    tags(file, topicnum, tagnum, docsdir, dicsdir, corpusdir)
