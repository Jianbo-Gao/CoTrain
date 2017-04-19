#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Author: gaojianbo@pku.edu.cn

import copy, json
from jieba import posseg
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import linear_model
import numpy as np

from setting import *
from XMLParser import XMLParser
from Translator import Translator
from Timer import Timer


class CoTrain:
    def __init__(self):
        # Init file names and datasets
        self.train_file_en=[TRAIN_DATA_BOOK,TRAIN_DATA_DVD,TRAIN_DATA_MUSIC]
        self.unlabel_file_cn=[UNLABEL_DATA_BOOK,UNLABEL_DATA_DVD,UNLABEL_DATA_MUSIC]
        self.validation_file_cn=[VALIDATION_DATA_BOOK,VALIDATION_DATA_DVD,VALIDATION_DATA_MUSIC]
        self.test_file_cn=[TEST_DATA_BOOK,TEST_DATA_DVD,TEST_DATA_MUSIC]

        self.train_data_en=[]
        self.train_data_cn=[]

        self.unlabel_data_cn=[]
        self.unlabel_data_en=[]

        self.validation_data_cn=[]
        self.validation_data_en=[]

        self.test_data_cn=[]
        self.test_data_en=[]

        # train, unlabel, validation, test
        self.vec_en=[[],[],[],[]]
        self.vec_cn=[[],[],[],[]]
        self.goal_en=[[],[],[],[]]
        self.goal_cn=[[],[],[],[]]

        self.classifier_cn=linear_model.LogisticRegression()
        self.classifier_en=linear_model.LogisticRegression()

        self.vectorizer_cn=CountVectorizer()
        self.vectorizer_en=CountVectorizer()

        self.transformer_cn=TfidfTransformer()
        self.transformer_en=TfidfTransformer()


    def read_data(self):
        # Read data from file
        timer = Timer("reading data")
        timer.start()
        parser=XMLParser()

        for filename in self.train_file_en:
            self.train_data_en+=parser.parse(filename)
        for filename in self.unlabel_file_cn:
            self.unlabel_data_cn+=parser.parse(filename)
        for filename in self.validation_file_cn:
            self.validation_data_cn+=parser.parse(filename)
        for filename in self.test_file_cn:
            self.test_data_cn+=parser.parse(filename)
        timer.finish()

    def __translate_dataset(self, translate_method, src_dataset):
        # Use translate_method to translate src_dataset into dst_dataset
        dst_dataset=[]
        for item in src_dataset:
            dst_item={}
            for attr in item:
                if attr in ['summary', 'text']:
                    dst_item[attr]=translate_method(item[attr])
                else:
                    dst_item[attr]=item[attr]
            dst_dataset.append(copy.deepcopy(dst_item))
        return dst_dataset

    def translate(self):
        # Translate all datasets
        timer = Timer("translating data")
        timer.start()
        translator=Translator()

        self.train_data_cn = self.__translate_dataset(translator.en_to_cn, self.train_data_en)
        self.unlabel_data_en = self.__translate_dataset(translator.cn_to_en, self.unlabel_data_cn)
        self.validation_data_en = self.__translate_dataset(translator.cn_to_en, self.validation_data_cn)
        self.test_data_en = self.__translate_dataset(translator.cn_to_en, self.test_data_cn)

        timer.finish()

    def __vectorize_datasets(self, language, use_jieba, train_data, unlabel_data, validation_data, test_data):
        # Vectorize datasets
        datasets=[train_data, unlabel_data, validation_data, test_data]

        if language=="en":
            vecs=self.vec_en
            goals=self.goal_en
            vectorizer=self.vectorizer_en
            transformer=self.transformer_en

        elif language=="cn":
            vecs=self.vec_cn
            goals=self.goal_cn
            vectorizer=self.vectorizer_cn
            transformer=self.transformer_cn

            if use_jieba:
                jieba_timer=Timer("using jieba")
                jieba_timer.start()

                # Read jieba cache
                jieba_cache={}
                with open(CACHE_JIEBA, 'r') as cachefile:
                    try:
                        jieba_cache=json.loads(cachefile.read())
                    except Exception, e:
                        print "Read Cache ERROR"
                        print e

                # Use jieba in CN
                for i in xrange(4):
                    for item in datasets[i]:
                        for key in ['summary', 'text']:
                            if item.has_key(key):
                                if not jieba_cache.has_key(item[key]):
                                    cut_obj=posseg.cut(item[key])
                                    temp_str=''
                                    for word_obj in cut_obj:
                                        temp_str+=word_obj.word+' '
                                    jieba_cache[item[key]]=copy.deepcopy(temp_str)
                                item[key]=copy.deepcopy(jieba_cache[item[key]])

                # Write jieba cache
                with open(CACHE_JIEBA, 'w') as cachefile:
                    try:
                        cache_data=json.dumps(jieba_cache)
                        cachefile.write(cache_data)
                    except Exception, e:
                        print "Write Cache ERROR"
                        print e

                jieba_timer.finish()


        for i in xrange(4):
            summary=[]
            text=[]
            goal=[]
            for item in datasets[i]:
                if item.has_key("summary"):
                    summary.append(copy.deepcopy(item["summary"]))
                else:
                    summary.append('')

                if item.has_key("text"):
                    text.append(copy.deepcopy(item["text"]))
                else:
                    text.append('')

                if item.has_key("polarity"):
                    goal.append(copy.deepcopy(item["polarity"]))
                else:
                    goal.append('')

            if i == 0:
                transformer.fit(vectorizer.fit_transform(summary+text))

            vec_summary=transformer.transform(vectorizer.transform(summary))
            vec_text=transformer.transform(vectorizer.transform(text))
            vecs[i]=copy.deepcopy(hstack((vec_summary, vec_text)))
            goals[i]=copy.deepcopy(goal)

            print "vec %d size: %s" % (i, str(vecs[i].shape))


    def vectorize(self, use_jieba=True):
        # Vectorize all data
        timer=Timer("vectorizing data")
        timer.start()

        self.__vectorize_datasets("en", use_jieba, self.train_data_en,
            self.unlabel_data_en, self.validation_data_en, self.test_data_en)

        self.__vectorize_datasets("cn", use_jieba, self.train_data_cn,
            self.unlabel_data_cn, self.validation_data_cn, self.test_data_cn)

        timer.finish()

    def __train_model(self, classifier, train_vec, train_goal):
        classifier.fit(train_vec, train_goal)

    def __test_model(self, classifier, test_vec):
        return classifier.predict(test_vec), classifier.predict_proba(test_vec)

    def train(self):
        # Train model

        """
        for i in xrange(0,4):
            print self.vec_en[i].shape
            print len(self.goal_en[i])

        for i in xrange(0,4):
            print self.vec_cn[i].shape
            print len(self.goal_cn[i])
        """

        timer=Timer("training")
        timer.start()

        # Train with train_vec_en/cn and train_goal_en/cn
        self.__train_model(self.classifier_en, self.vec_en[0], self.goal_en[0])
        self.__train_model(self.classifier_cn, self.vec_cn[0], self.goal_cn[0])

        timer.finish()


    def test(self):
        # Test with test_vec_en/cn
        timer=Timer("testing")
        timer.start()

        self.predict_en, self.proba_en = self.__test_model(self.classifier_en, self.vec_en[3])
        self.predict_cn, self.proba_cn = self.__test_model(self.classifier_cn, self.vec_cn[3])

        timer.finish()


    def cotrain(self, n=5, p=5):
        # 1. Test with unlabel_vec_en/cn
        # 2. Add best n/p unlabel samples into train_vec_en/cn
        # 3. ReTrain model with new train_vec_en/cn
        timer=Timer("cotraining")
        timer.start()

        # 1. Test
        unlabel_predict_en, unlabel_proba_en = self.__test_model(self.classifier_en, self.vec_en[1])
        unlabel_predict_cn, unlabel_proba_cn = self.__test_model(self.classifier_cn, self.vec_cn[1])

        tmp_nega_en=np.hsplit(unlabel_proba_en,2)[0].transpose()[0]
        tmp_posi_en=np.hsplit(unlabel_proba_en,2)[1].transpose()[0]

        tmp_nega_cn=np.hsplit(unlabel_proba_cn,2)[0].transpose()[0]
        tmp_posi_cn=np.hsplit(unlabel_proba_cn,2)[1].transpose()[0]

        best_nega_en=[]
        best_nega_cn=[]

        best_posi_en=[]
        best_posi_cn=[]

        # 2. Add
        for i in xrange(n):
            # Add n best negative unlabel samples
            index_nega_en=tmp_nega_en.argmax()
            tmp_nega_en[index_nega_en]=-1
            best_nega_en.append(copy.deepcopy(index_nega_en))

            index_nega_cn=tmp_nega_cn.argmax()
            tmp_nega_cn[index_nega_cn]=-1
            best_nega_cn.append(copy.deepcopy(index_nega_cn))

        for j in xrange(p):
            # Add p best positive unlabel samples
            index_posi_en=tmp_posi_en.argmax()
            tmp_posi_en[index_posi_en]=-1
            best_posi_en.append(copy.deepcopy(index_posi_en))

            index_posi_cn=tmp_posi_cn.argmax()
            tmp_posi_cn[index_posi_cn]=-1
            best_posi_cn.append(copy.deepcopy(index_posi_cn))

        # Best negative CN samples
        best_nega_cn_set=set(best_nega_cn)-set(best_posi_cn)
        tmp_remove=[]
        for best in best_nega_cn_set:
            item=self.unlabel_data_cn[best]
            item['polarity']='N'
            self.train_data_cn.append(copy.deepcopy(item))
            tmp_remove.append(copy.deepcopy(item))
        for item_remove in tmp_remove:
            self.unlabel_data_cn.remove(item_remove)

        # Best negative EN samples
        best_nega_en_set=set(best_nega_en)-set(best_posi_en)
        tmp_remove=[]
        for best in best_nega_en_set:
            item=self.unlabel_data_en[best]
            item['polarity']='N'
            self.train_data_en.append(copy.deepcopy(item))
            tmp_remove.append(copy.deepcopy(item))
        for item_remove in tmp_remove:
            self.unlabel_data_en.remove(item_remove)

        # Best positive CN samples
        best_posi_cn_set=set(best_posi_cn)-set(best_nega_cn)
        tmp_remove=[]
        for best in best_posi_cn_set:
            item=self.unlabel_data_cn[best]
            item['polarity']='P'
            self.train_data_cn.append(copy.deepcopy(item))
            tmp_remove.append(copy.deepcopy(item))
        for item_remove in tmp_remove:
            self.unlabel_data_cn.remove(item_remove)

        # Best positive EN samples
        best_posi_en_set=set(best_posi_en)-set(best_nega_en)
        tmp_remove=[]
        for best in best_posi_en_set:
            item=self.unlabel_data_en[best]
            item['polarity']='P'
            self.train_data_en.append(copy.deepcopy(item))
            tmp_remove.append(copy.deepcopy(item))
        for item_remove in tmp_remove:
            self.unlabel_data_en.remove(item_remove)

        # 3. Retrain
        self.vectorize(False)
        self.train()

        timer.finish()

    def evaluate(self):
        # Evaluate for test result
        timer=Timer("evaluating")
        timer.start()

        test_len=len(self.test_data_cn)
        book_sum=dvd_sum=music_sum=0
        book_right_num=dvd_right_num=music_right_num=0


        for i in xrange(test_len):
            n=self.proba_cn[i][0]+self.proba_en[i][0]
            p=self.proba_cn[i][1]+self.proba_en[i][1]
            if n > p:
                result='N'
            else:
                result='P'

            result_flag=False
            if result == self.test_data_cn[i]['polarity']:
                result_flag=True

            if self.test_data_cn[i]['category']=="book":
                book_sum+=1
                if result_flag:
                    book_right_num+=1
            elif self.test_data_cn[i]['category']=="dvd":
                dvd_sum+=1
                if result_flag:
                    dvd_right_num+=1
            elif self.test_data_cn[i]['category']=="music":
                music_sum+=1
                if result_flag:
                    music_right_num+=1


        book_right_rate=1.0*book_right_num/book_sum
        dvd_right_rate=1.0*dvd_right_num/dvd_sum
        music_right_rate=1.0*music_right_num/music_sum

        results={}
        results['book']=(book_right_rate, book_right_num, book_sum)
        results['dvd']=(dvd_right_rate, dvd_right_num, dvd_sum)
        results['music']=(music_right_rate, music_right_num, music_sum)

        timer.finish()

        return results




def main(iterator_times):
    timer=Timer("CoTraining")
    timer.start()

    trainer=CoTrain()
    trainer.read_data()
    trainer.translate()

    trainer.vectorize()
    trainer.train()
    #trainer.test()
    #results.append(copy.deepcopy(trainer.evaluate()))

    for i in xrange(iterator_times):
        iterator_timer=Timer("CoTrain iterating "+str(i+1))
        iterator_timer.start()

        trainer.cotrain()
        #trainer.test()
        #results.append(copy.deepcopy(trainer.evaluate()))

        iterator_timer.finish()

    trainer.test()
    results=trainer.evaluate()
    print results

    with open(ITERATE_RESULT, 'w') as resultfile:
        try:
            resultfile.write(json.dumps(results))
        except Exception, e:
            print "Write Result Cache ERROR"
            print e

    timer.finish()





if __name__ == '__main__':
    main(40)



