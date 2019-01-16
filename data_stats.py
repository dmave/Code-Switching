"""
Utility functions for data set statistics and Code-Mixing statistics.
reference: 1. Code-Mixing Index (CMI) - B. Gambäck and A. Das. On Measuring the Complexity of
           Code-Mixing. - http://amitavadas.com/Pub/CMI.pdf
           2. M-Index, I-Index - Gualberto Guzmán, Joseph Ricard, Jacqueline Serigos, Barbara E. Bullock, Almeida Jacqueline Toribio. (2017)
           Metrics for modeling code-switching across corpora

"""

from  collections import Counter, defaultdict
from operator import itemgetter
from data_analysis import data_utils
import numpy as np
import math
import matplotlib
matplotlib.use('Tkagg')
import matplotlib.pyplot as plt

def get_vocabulary(sentences):
    """
    :param sentences: list of sentences/utterances in a data set
    :return: Frequency dictionary of words
    """
    vocabulary = {}
    for s in sentences:
        # for w in s.split():
        for w in s:
            vocabulary[w] = vocabulary.get(w, 0) + 1
    return vocabulary

def label_vocabulary(data):
    """
    Returns dictionary of words for each of the majority labels.
    """
    en = {}
    hi = {}
    ne = {}
    other = {}

    for item in data:
        for word in item:
            if word[-1] == 'lang1':
                en[word[1].lower()] = en.get(word[1], 0) + 1
            elif word[-1] == 'lang2':
                hi[word[1].lower()] = hi.get(word[1], 0) + 1
            elif word[-1] == 'ne':
                ne[word[1].lower()] = ne.get(word[1], 0) + 1
            elif word[-1] == 'other':
                other[word[1].lower()] = other.get(word[1], 0) + 1
    return [en, hi, ne, other]

def get_word_count(data):
    """
    :param data: list of list of words (dataset)
    :return: count of tokens
    """
    word_count = 0
    for line in data:
        for word in line:
            word_count += 1
    return word_count

def get_label_counts(label_list):
    """
    :param: list of list of labels
    :return: dict of label:count
    """
    llist = [l for s in label_list for l in s]
    return Counter(llist)

def train_test_label_ratio(ytrain, ytest):
    """
    :param ytrain: list of list of labels
    :param ytest: list of list of labels
    :return: dict of ratio of test:train labels
    """
    train_count = get_label_counts(ytrain)
    test_count = get_label_counts(ytest)
    print("***Label distribution-train = ", train_count)
    print("***Label distribution-test = ", test_count)

    ratio = {}
    for k,v in train_count.items():
        ratio[k] = format(float(test_count[k])/float(v), '.4f')


    return sorted(ratio.items(), key=itemgetter(0))



def count_freq_ngrams(ngram_dict, freq_threshold):
    count = 0
    freq_bigrams = []
    for k, v in ngram_dict.items():
        if v < freq_threshold:
            count += 1
            freq_bigrams.append(k)
    return count, freq_bigrams

########################################################################################################################
#               Code-Mixing Statistics Functions
########################################################################################################################

def has_code_mixing(utterance, label_idx=0):
    """
    :param utterance: list of words and its labels
    :return: True if utterance has code mixing
    """
    if label_idx:
        utterance_labels = data_utils.sent_to_labels(utterance, label_idx)
    else:
        utterance_labels = utterance
    # print(utterance_labels[0])
    if "lang1" in utterance_labels and "lang3" in utterance_labels:
        return True
    else:
        return False

def is_en_monolingual(utterance, label_idx=0):
    """
    :param utterance: list of words and its labels
    :return: True if utterance has only english words
    """
    if label_idx:
        utterance_labels = data_utils.sent_to_labels(utterance, label_idx)
    else:
        utterance_labels = utterance
    if "lang1" in utterance_labels and "lang3" not in utterance_labels:
        return True
    else:
        return False

def is_hi_monolingual(utterance, label_idx=0):
    """
        :param utterance: list of words and its labels
        :return: True if utterance has only hindi words
        """
    if label_idx:
        utterance_labels = data_utils.sent_to_labels(utterance, label_idx)
    else:
        utterance_labels = utterance

    if "lang3" in utterance_labels and "lang1" not in utterance_labels:
        return True
    else:
        return False

    return [total_count, cs_2_count, cs_2_list, cs_3_count, cs_3_list, more_than_cs_3_count, more_than_cs_3_list]

def get_utterance_level_language_distribution(data, label_idx=0):
    """
    :param data: list of list of utterances in the dataset
    :return: list with following statistics in order - CM sentence count and fraction,
                lang1 monolingual sentences count and fraction,
            lang2 monolingual sentences count and fraction,
             count and fraction of sentences without any language tags.
    """
    cm_sent_count = 0
    lang1_sent_count = 0
    lang2_sent_count = 0
    other_sent_count = 0
    for line in data:
        if has_code_mixing(line, label_idx):
            cm_sent_count += 1
        elif is_en_monolingual(line, label_idx):
             lang1_sent_count += 1

        elif is_hi_monolingual(line, label_idx):
            lang2_sent_count += 1
        else:
            other_sent_count += 1
    total_sent_count = len(data)
    # print(total_sent_count)

    return [cm_sent_count, round(100 * float(cm_sent_count)/total_sent_count, 2), lang1_sent_count, round(100 * float(lang1_sent_count)/total_sent_count, 2),
            lang2_sent_count, round(100 * float(lang2_sent_count)/total_sent_count,2), other_sent_count, round(100 * float(other_sent_count)/total_sent_count,2)]

def count_cs_points(utterance, numfields):
    """
    Note: Extended version of the method implementation by Gustavo.
    :param utterance: list of utterance_id, words and language labels
    :param numfields: number of fields in each  unit of utterance. data formats :[word, label] or [pots_id, word, label]
    :return: list of total CS points count, consecutive CS points count, list of consecutive CS points (word, label),
            1 word in between CS point count and its (word, label) list, more than 1 word in between CS point count and its (word,label) list
    """
    cs_2 = 2 #consecutive word switch
    cs_3 = 3 #1 word with other than lang label in between
    cs_2_count = 0
    cs_3_count = 0
    more_than_cs_3_count = 0 #more than word with other than lang label in between

    cs_2_list = []
    cs_3_list = []
    more_than_cs_3_list = []
    prev_lang_label = ''
    prev_lang_label_idx = -1
    if numfields == 3:
        for i, (post_id, word, lang_id) in enumerate(utterance):
            if (i == 0 and (lang_id not in ['lang1', 'lang2'])) or (lang_id not in ['lang1', 'lang2']):
                continue

            if not prev_lang_label:
                prev_lang_label = lang_id
                prev_lang_label_idx = i

            if prev_lang_label != lang_id:
                temp = [post_id]
                for j in range(prev_lang_label_idx, i+1):
                    temp.append((utterance[j][1], utterance[j][2]))
                if len(temp) == cs_2 + 1:
                    cs_2_count += 1
                    cs_2_list.append(temp)
                elif len(temp) == cs_3 + 1:
                    cs_3_count += 1
                    cs_3_list.append(temp)
                elif len(temp) > cs_3 + 1:
                    more_than_cs_3_count += 1
                    more_than_cs_3_list.append(temp)
                    # more_than_cs_3_list += temp
            prev_lang_label = lang_id
            prev_lang_label_idx = i
        total_count = cs_2_count + cs_3_count + more_than_cs_3_count
    else:
        for i, (word, lang_id) in enumerate(utterance):
            if (i == 0 and (lang_id not in ['lang1', 'lang2'])) or (lang_id not in ['lang1', 'lang2']):
                continue

            if not prev_lang_label:
                prev_lang_label = lang_id
                prev_lang_label_idx = i

            if prev_lang_label != lang_id:
                temp = []
                for j in range(prev_lang_label_idx, i + 1):
                    temp.append((utterance[j][0], utterance[j][1]))
                if len(temp) == cs_2:
                    cs_2_count += 1
                    cs_2_list.append(temp)
                elif len(temp) == cs_3:
                    cs_3_count += 1
                    cs_3_list.append(temp)
                elif len(temp) > cs_3:
                    more_than_cs_3_count += 1
                    more_than_cs_3_list.append(temp)
                    # more_than_cs_3_list += temp
            prev_lang_label = lang_id
            prev_lang_label_idx = i
        total_count = cs_2_count + cs_3_count + more_than_cs_3_count

    return [total_count, cs_2_count, cs_2_list, cs_3_count, cs_3_list, more_than_cs_3_count, more_than_cs_3_list]


def get_code_switching_point_distribution(data, numfields):
    """
    :param data: list of list of utterances in the dataset
    :param numfields: depending on format of each unit of list of list - [word, label] or [post_id, word, label
    :return: list with following statistics in order - total CS point count,
                consecutive word CS point count and nested list of CS points tuples,
                 1 word in between CS point count and nested list of CS points tuples,
                 more than 1 word in between CS point count and nested list of CS points tuples.
    """
    consec_csp_count = 0
    word1_csp_count = 0
    word2_more_csp_count = 0
    consec_csp_list = []
    word1_csp_list = []
    word2_more_csp_list = []
    total_csp_count = 0
    if numfields == 2:
        label_idx = 1
    elif numfields == 3:
        label_idx = 2
    integration_index_list = []
    for line in data:
        if has_code_mixing(line,label_idx ):
            csp_stat = count_cs_points(line, numfields)
            total_csp_count += csp_stat[0]
            integration_index_list.append(csp_stat[0]/float(len(line)-1))
            consec_csp_count += csp_stat[1]
            if csp_stat[2]:
                consec_csp_list.append(csp_stat[2])
            word1_csp_count += csp_stat[3]
            if csp_stat[4]:
                word1_csp_list.append(csp_stat[4])
            word2_more_csp_count += csp_stat[5]
            if csp_stat[6]:
                word2_more_csp_list.append(csp_stat[6])
        else:
            integration_index_list.append(0.0)
    avg_i_index =  np.mean(integration_index_list)
    i_index_std = np.std(np.array(integration_index_list), axis=0)


    return [total_csp_count, consec_csp_count, consec_csp_list, word1_csp_count,
            word1_csp_list, word2_more_csp_count, word2_more_csp_list, avg_i_index, i_index_std]


def get_utterance_CMI(utterance, label_idx):
    utterance_labels = data_utils.sent_to_labels(utterance, label_idx)
    token_count = len(utterance_labels)
    label_counts = {'other': 0}

    utterance_CMI = 0.0
    for label in utterance_labels:
        if label == 'lang1':
            label_counts['lang1'] = label_counts.get('lang1', 0) + 1

        elif label == 'lang2':
            label_counts['lang2'] = label_counts.get('lang2', 0) + 1
        elif label == 'fw':
            label_counts['fw'] = label_counts.get('fw', 0) + 1
        else:
            label_counts['other'] = label_counts.get('other', 0) + 1  #u - ne, other, mixed, ambiguous, unk

    lang_label_counts = [label_counts[key] for key in label_counts.keys() if key not in 'other']
    if lang_label_counts: #if utterance doesn't have any language labels
        max_lang_count = float(max(lang_label_counts))
    else:
        max_lang_count = 0.0

    if token_count > label_counts['other']:
        temp = max_lang_count / float(token_count-label_counts['other']) #max{wi}/n-u
        utterance_CMI = (1-temp) * 100

    else:
        utterance_CMI = 0.0

    return utterance_CMI

def get_dataset_CMI(data, label_idx, cm_sent_count):
    """
    :param data:
    :param label_idx:
    :return: CMI_all = average over all sentences in dataset,
             CMI_mixed = average only for the sentences that contain code-mixing.
    """
    all_sents_count = float(len(data))
    CMI = 0.0
    for line in data:
        CMI += get_utterance_CMI(line, label_idx)
    return [round(CMI/all_sents_count, 3), round(float(CMI)/cm_sent_count, 3)]

def prob_lang_j(langj_counts, all_count):
    return [lc/all_count for lc in langj_counts]


def language_word_counts(data_labels, lang_labels):
    p_lang = defaultdict(list)
    word_count = 0
    for sentence in data_labels:
        word_count += len(sentence)
        counts = Counter(sentence)
        p_lang[0].append(counts[lang_labels[0]])
        p_lang[1].append(counts[lang_labels[1]])

    return [sum(val) for key, val in p_lang.items()]

def get_m_index(data_labels, lang_labels):
    """
    Multilingual Index is a word-count-based measure that quantifies the inequality of the distribution of language tags
    in a corpus of at least two languages. Range: [0:Monolingual,1:each lang represented equally]
    Ref:  R. Barnett et.al, “The LIDES Coding Manual : A document for preparing and analyzing language interaction data"
    :param data_labels: list of list of labels for each utterance
    :param lang_labels: list of language labels
    :return: M_index of corpus
    """
    word_count = sum([len(y) for y in data_labels])
    langj_c = language_word_counts(data_labels, lang_labels)
    lang_word_count = sum(langj_c)
    p_j_all_words = prob_lang_j(langj_c, word_count)
    print(p_j_all_words)
    p_j_lang_words = prob_lang_j(langj_c, lang_word_count)
    print(p_j_lang_words)
    p_j_all_words_sqr = [p*p for p in p_j_all_words]
    p_j_lang_words_sqr = [p*p for p in p_j_lang_words]
    print("Total word_count = ", word_count)
    print("Total number of language words = ", lang_word_count)
    print(langj_c)
    # print(p_j_all_words, p_j_all_words_sqr)
    # print(p_j_lang_words, p_j_lang_words_sqr)
    m_idx_all = (1- sum(p_j_all_words_sqr))/ ((len(lang_labels)-1) * sum(p_j_all_words_sqr))
    m_idx_lang = (1 - sum(p_j_lang_words_sqr)) / ((len(lang_labels) - 1) * sum(p_j_lang_words_sqr))

    return list(map(lambda x: round(x,4), [m_idx_all, m_idx_lang]))

def get_language_entropy(data_labels, lang_labels):

    """

    :param data_labels: list of list of labels for each utterance
    :param lang_labels: list of language labels
    :return: Language entropy of the corpus and the upperbound for LE
    """
    # word_count = sum([len(y) for y in data_labels])
    langj_c = language_word_counts(data_labels, lang_labels)
    lang_word_count = sum(langj_c)
    pj_lang = prob_lang_j(langj_c, lang_word_count)
    # print(pj_lang)
    # print(prob_lang_j(langj_c, word_count))
    le = -sum([pj*math.log(pj,2) for pj in pj_lang])
    # print(-sum([pj*math.log(pj,2) for pj in prob_lang_j(langj_c, word_count)]))
    return round(le, 4), math.log(len(lang_labels), 2)

def language_span(utterance, lang_label):
    temp = []
    spans = []
    for l in utterance:
        if l != lang_label:
            if temp:
                spans.append(len(temp))
            temp = []
        else:
            temp.append(l)
    return Counter(spans)




if __name__ == '__main__':

    data, word_count = data_utils.load_data(data_utils.data_path + 'all_dataset.tsv', 3)
    y = [data_utils.sent_to_labels(s, 2) for s in data]
    midx = get_m_index(y, ['lang1', 'lang2'])
    print("M-Index for FB = ", midx)
    

    





