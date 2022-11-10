import torch
import spacy
import difflib
import itertools
import re
import json
import string
import pycountry
import pandas as pd
import numpy as np

train_pt = torch.load('../../NetKu/full_content/train.pt')
# train_pt = torch.load('../../NetKu/full_content/check_cands.pt')
# train_pt = torch.load('../../edit_NetKu/util/train_labeled.pt')


# test_pt = torch.load('../../NetKu/full_content/test.pt')
# val_pt = torch.load('../../NetKu/full_content/val.pt')

spacy_package = 'en_core_web_sm'
nlp_ner = None

def get_nlp():
    nlp = None
    if not nlp:
        try:
            nlp = spacy.load(spacy_package, disable=["tagger" "ner"])
        except:
            import subprocess
            print('downloading spacy...')
            subprocess.run("python3 -m spacy download %s" % spacy_package, shell=True)
            nlp = spacy.load(spacy_package, disable=["tagger" "ner"])
    return nlp

def get_nlp_ner():
    global nlp_ner
    if not nlp_ner:
        nlp_ner = spacy.load(spacy_package, disable=["tagger"])  # just the parser
    return nlp_ner

to_filter = [
    'Share on WhatsApp',
    'Share on Messenger',
    'Reuse this content',
    'Share on LinkedIn',
    'Share on Pinterest' ,
    'Share on Google+',
    'Listen /',
    '– Politics Weekly',
    'Sorry your browser does not support audio',
    'https://flex.acast.com',
    '|',
    'Share on Facebook',
    'Share on Twitter',
    'Share via Email',
    'Sign up to receive',
    'This article is part of a series',
    'Follow Guardian',
    'Twitter, Facebook and Instagram',
    'UK news news',
    'Click here to upload it',
    'Do you have a photo',
    'Listen /',
    'Email View',
    'Read more Guardian',
    'This series is',
    'Readers can recommend ',
    'UK news news',
    'Join the debate',
    'guardian.letters@theguardian.com',
    'More information',
    'Close',
    'All our journalism is independent',
    'is delivered to thousands of inboxes every weekday',
    'with today’s essential stories',
    'Newsflash:',
    'You can read terms of service here',
    'Guardian rating:',
    'By clicking on an affiliate link',
    'morning briefing news',
    'Analysis:',
    'Good morning, and welcome to our rolling coverage',
    'South and Central Asia news',
    'f you have a direct question',
    'sign up to the',
    'You can read terms of service here.',
    'If you want to attract my attention quickly, it is probably better to use Twitter.',
    'UK news',
]
to_filter = list(map(lambda x: x.lower(), to_filter))
starts_with = [
    'Updated ',
    'Here’s the sign-up',
    '[Read more on',
    '[Here’s the list of',
    '[Follow our live coverage',
    '[',
]
contains = [
    'Want to get this briefing by email',
    'Thank youTo'
]
ends_with = [
    ']',
]
last_line_re = re.compile('Currently monitoring (\d|\,)+ news articles')
version_re = re.compile('Version \d+ of \d+')
#---
## general res
clean_escaped_html = re.compile('&lt;.*?&gt;')
end_comma = re.compile(',$')
stopwords = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself",
             "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself",
             "they",
             "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those",
             "am",
             "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did",
             "doing",
             "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for",
             "with",
             "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to",
             "from",
             "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here",
             "there",
             "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some",
             "such",
             "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "can", "will", "just", "should",
             "now"]
stopwords_lemmas = list(set(map(lambda x: x.lemma_, get_nlp()(' '.join(stopwords)))))
## lambdas
filter_sents = lambda x: not (
    any(map(lambda y: y in x, contains)) or
    any(map(lambda y: x.startswith(y), starts_with)) or
    any(map(lambda y: x.endswith(y), ends_with))
)
#--
def get_words(s, split_method='spacy'):
    if split_method == 'spacy':
        return list(map(lambda x: x.text, get_nlp()(s)))
    else:
        return s.split()

get_lemmas = lambda s: list(map(lambda x: x.lemma_.lower(), get_nlp()(s)))
filter_stopword_lemmas = lambda word_list: list(filter(lambda x: x not in stopwords_lemmas, word_list))
filter_punct = lambda word_list: list(filter(lambda x: x not in string.punctuation, word_list))

# Convert string into pars -> do filtering than convert back to string format
def filter_lines(a):
    if isinstance(a, list):
        pars = a
    else:
        # pars = a.split('</p>')
        pars = a.split('\n\n')
    output = []
    for p in pars:
        if not any(map(lambda x: x in p.lower(), to_filter)):
            output.append(p)
    if isinstance(a, list):
        return output
    else:
        return '\n\n'.join(output)

def is_dateline(x):
    ## is short enough
    length = len(x.split()) < 6
    # has a country name
    # 1. Does it have an uppercase word?
    has_gpe = any(map(lambda x: x.isupper(), x.split()))
    # 2. Is there a country name?
    if not has_gpe:
        for word in get_words(x):
            try:
                pycountry.countries.search_fuzzy(word)
                has_gpe = True
                break
            except LookupError:
                has_gpe = False
    # 3. Is there a GPE?
    if not has_gpe:
        doc = get_nlp_ner()(x)
        for ent in doc.ents:
            if ent.label_ == 'PERSON':
                has_gpe = True
    ##
    if length and has_gpe:
        return True
    else:
        return False

# Split into sentences
def split_sents(a, perform_filter=True):
    nlp = get_nlp()
    output_sents = []

    # deal with dateline (this can really mess things up...)
    dateline_dashes = ['—', '–']
    for d in dateline_dashes:
        dateline = a.split(d)[0]
        if is_dateline(dateline): ## find the dateline
            ## dateline.
            output_sents.append(dateline.strip())
            ## all other sentences.
            a = d.join(a.split(d)[1:]).strip()
            break

    # get sentences from each paragraph
    # pars = a.split('.\n\n')
    # get pars, then read the sentences from each par
    pars = a.split('.\n\n')
    for p in pars:
        doc = nlp(p)
        sents = list(map(lambda x: x.text, doc.sents))
        output_sents += sents

    
    # filter out garbage/repetitive sentences
    if perform_filter:
        output_sents = filter_lines(output_sents)

    # last-minute processing
    output_sents = list(map(lambda x: x.strip(), output_sents))

    # merge dateline in with the first sentence
    if len(output_sents) > 0:
        if is_dateline(output_sents[0]):
            output_sents = ['—'.join(output_sents[:2])] + output_sents[2:]
    # output_sents = '.\n\n'.join(output_sents)
    return output_sents

def get_word_diff_ratio(s_old, s_new):
    s_old_words, s_new_words = get_words(s_old), get_words(s_new)
    return difflib.SequenceMatcher(None, s_old_words, s_new_words).ratio()

def get_list_diff(l_old, l_new):
    vars_old = []
    vars_new = []
    diffs = list(difflib.ndiff(l_old, l_new))
    in_question = False
    for idx, item in enumerate(diffs):
        label, text = item[0], item[2:]
        if label == '?':
            continue

        elif label == '-':
            vars_old.append({
                'text': text,
                'tag': '-'
            })
            if (
                    ## if something is removed from the old sentnece, a '?' will be present in the next idx
                    ((idx < len(diffs) - 1) and (diffs[idx + 1][0] == '?'))
                    ## if NOTHING is removed from the old sentence, a '?' might still be present in 2 idxs, unless the next sentence is a - as well.
                 or ((idx < len(diffs) - 2) and (diffs[idx + 2][0] == '?') and diffs[idx + 1][0] != '-')
            ):
                in_question = True
                continue

            ## test if the sentences are substantially similar, but for some reason ndiff marked them as different.
            if (idx < len(diffs) - 1) and (diffs[idx + 1][0] == '+'):
                _, text_new = diffs[idx + 1][0], diffs[idx + 1][2:]
                if get_word_diff_ratio(text, text_new) > .9:
                    in_question = True
                    continue

            vars_new.append({
                'text': '',
                'tag': ''
            })


        elif label == '+':
            old_text, new_text = diffs[idx-2][2:], diffs[idx][2:]
            sents_ratio = get_word_diff_ratio(old_text, new_text) 
            
            if sents_ratio >= .8:
                vars_new.append({
                    'text': new_text,
                    'tag': ' '
                })
            elif sents_ratio < .3:
                vars_new.append({
                    'text': new_text,
                    'tag': '+'
                })
            else:
                vars_new.append({
                    'text': new_text,
                    'tag': '*'
                })
            # if in_question:
            #     in_question = False
            # else:
            #     vars_old.append({
            #         'text':'',
            #         'tag': ' '
            #     })
        else:
            vars_old.append({
                'text': text,
                'tag': ' '
            })
            vars_new.append({
                'text': text,
                'tag': ' '
            })

    return vars_old, vars_new

def cluster_edits(vo, vn):
    clustered_edits = []
    current_cluster = []
    for o, n in list(zip(vo, vn)):
        if (o['tag'] in ['+', '-']) or (n['tag'] in ['+', '-']):
            current_cluster.append((o, n))
        ##
        if o['tag'] == ' ' and n['tag'] == ' ':
            if len(current_cluster) > 0:
                clustered_edits.append(current_cluster)
                current_cluster = []
            clustered_edits.append([(o, n)])
    if len(current_cluster) > 0:
        clustered_edits.append(current_cluster)
    return clustered_edits


def merge_sents(idx_i, idx_j, a, c):
    """Merges two sentences without spacing errors."""
    si_text = c[idx_i][a]['text']
    sj_text = c[idx_j][a]['text']

    if isinstance(si_text, (list, tuple)):
        output_list = list(si_text)
    else:
        output_list = [(idx_i, si_text)]
    if isinstance(sj_text, (list, tuple)):
        output_list += sj_text
    else:
        output_list.append((idx_j, sj_text))
    return output_list

def merge_sents_list(t):
    t = sorted(t, key=lambda x: x[0])
    t = list(map(lambda x: x[1].strip(), t))
    t = ' '.join(t)
    return ' '.join(t.split())

def text_in_interval(c, idx_i, idx_j, version):
    idx_small, idx_large = min([idx_i, idx_j]), max([idx_i, idx_j])
    return any(map(lambda idx: c[idx][version]['text'].strip() != '',  range(idx_small+1, idx_large)))

def lemmatize_sentence(s, cache):
    if isinstance(s, str) and s in cache:
        return cache[s], cache
    if isinstance(s, list):
        s = merge_sents_list(s)
    s_lemmas = get_lemmas(s)
    s_lemmas = filter_stopword_lemmas(s_lemmas)
    s_lemmas = filter_punct(s_lemmas)
    cache[s] = s_lemmas
    return cache[s], cache

def check_subset(s1_lemmas, s2_lemmas, slack=.5):
    """Checks if the second sentence is nearly a subset of the first, with up to `slack` words different."""
    ### get all text (might be a list).
    if len(s2_lemmas) > len(s1_lemmas):
        return False
    if len(s2_lemmas) > 50:
        return False
    ### check match.
    matches = sum(map(lambda word: word in s1_lemmas, s2_lemmas))
    return matches >= (len(s2_lemmas) * (1 - slack))

def swap_text_spots(c, old_spot_idx, new_spot_idx, version):
    ## swap text
    text_old = c[old_spot_idx][version]['text']
    text_new = c[new_spot_idx][version]['text']
    c[new_spot_idx][version]['text'] = text_old
    c[old_spot_idx][version]['text'] = text_new
    ## swap tags
    tag_new = c[new_spot_idx][version]['tag']
    tag_old = c[old_spot_idx][version]['tag']
    c[new_spot_idx][version]['tag'] = tag_old
    c[old_spot_idx][version]['tag'] = tag_new
    return c

import copy
def merge_cluster(c, slack=.5):
    c = list(filter(lambda x: x[0]['text'] != '' or x[1]['text'] != '', c))
    old_c = copy.deepcopy(c)
    r_c = range(len(c))
    keep_going = True
    loop_idx = 0
    cache = {}

    while keep_going:
        for active_version in [0, 1]:
            inactive_version = abs(active_version - 1)
            for idx_i, idx_j in itertools.product(r_c, r_c):
                # [(0, 0), (0, 1), (1, 0), (1, 1)]
                idx_i, idx_j = (idx_i, idx_j) if active_version == 0 else (idx_j, idx_i)
                if (
                        (idx_i != idx_j)
                        and (c[idx_j][active_version]['text'] != '')
                        # and (c[idx_j][inactive_version]['text'] == '')
                        and (c[idx_i][inactive_version]['text'] != '')
                ):

                    # print('active: %s, idx_i: %s, idx_j: %s' % (active_version, idx_i, idx_j))
                    s1_lemmas, cache = lemmatize_sentence(c[idx_i][inactive_version]['text'], cache)
                    s2_lemmas, cache = lemmatize_sentence(c[idx_j][active_version]['text'], cache)
                    if check_subset(s1_lemmas, s2_lemmas, slack=slack):
                        # if there's a match, first check:
                        combined_text_active = merge_sents(idx_i, idx_j, active_version, c)
                        combined_text_inactive = merge_sents(idx_i, idx_j, inactive_version, c)
                        c[idx_j][active_version]['text'] = combined_text_active
                        c[idx_i][active_version]['text'] = ''
                        c[idx_i][inactive_version]['text'] = combined_text_inactive
                        c[idx_j][inactive_version]['text'] = ''
                        # print('FOUND')
                        # print(c)
                        # print('active: %s, idx_i: %s, idx_j: %s' % (active_version, idx_i, idx_j))

                        #    1. if the two idx's are adjacent, then move the active.
                        if abs(idx_i - idx_j) == 1:
                            # print('1.')
                            c = swap_text_spots(c, new_spot_idx=idx_i, old_spot_idx=idx_j, version=active_version)

                        #    2. if there's both >=1 active AND >=1 inactive in between, don't do anything.
                        elif text_in_interval(c, idx_i, idx_j, active_version) and text_in_interval(c, idx_i, idx_j, inactive_version):
                            # print('2.')
                            pass

                        #    3. if there's text in the active version between the two idx's, move the inactive.
                        elif text_in_interval(c, idx_i, idx_j, active_version):
                            # print('3.')
                            c = swap_text_spots(c, new_spot_idx=idx_j, old_spot_idx=idx_i, version=inactive_version)

                        #    4. if there's text in the inactive in between the two idx's, move the active.
                        elif text_in_interval(c, idx_i, idx_j, inactive_version):
                            # print('4.')
                            c = swap_text_spots(c, new_spot_idx=idx_i, old_spot_idx=idx_j, version=active_version)

                        #   5. if there's no text inbetween the idx's in either the active or the inactive, move the active.
                        elif not (
                                text_in_interval(c, idx_i, idx_j, active_version) and
                                text_in_interval(c, idx_i, idx_j, inactive_version)
                        ):
                            # print('5.')
                            c = swap_text_spots(c, new_spot_idx=idx_i, old_spot_idx=idx_j, version=active_version)

                        ## merge list/text
                        for idx, version in itertools.product([idx_i, idx_j], [active_version, inactive_version]):
                            if isinstance(c[idx][version]['text'], list):
                                c[idx][version]['text'] = merge_sents_list(c[idx][version]['text'])

        ## one more merge for safety
        for idx, version in itertools.product(r_c, [active_version, inactive_version]):
            if isinstance(c[idx][version]['text'], list):
                c[idx][version]['text'] = merge_sents_list(c[idx][version]['text'])

        if (c == old_c) or (loop_idx > 10000):
            # print('done, idx: %s' % loop_idx)
            keep_going = False
            loop_idx = 0
        else:
            loop_idx += 1
            # print('one more')
            old_c = copy.deepcopy(c)

    return c

def merge_all_clusters(vo, vn, slack=.5):
    clustered_edits = cluster_edits(vo, vn)
    output_edits = []
    for c in clustered_edits:
        if len(c) == 1:
            c_i = c[0]
            if not (c_i[0]['text'] == '' and c_i[1]['text'] == ''):
                output_edits.append(c_i)
        else:
            c_new = merge_cluster(c, slack=slack)
            for c_i in c_new:
                if not (c_i[0]['text'] == '' and c_i[1]['text'] == ''):
                    output_edits.append(c_i)

    if len(output_edits) == 0:
        return None, None

    return zip(*output_edits)

def get_sentence_diff(a_old, a_new, filter_common_sents=True, merge_clusters=True, slack=.5):
    ## split sentences
    a_old_sents = split_sents(a_old)
    a_new_sents = split_sents(a_new)
    if filter_common_sents:
        a_old_sents = list(filter(filter_sents, a_old_sents))
        a_new_sents = list(filter(filter_sents, a_new_sents))
    ## group list
    vers_old, vers_new = get_list_diff(a_old_sents, a_new_sents)
    ## fix errors/ align sentences
    if merge_clusters:
        vers_old, vers_new = merge_all_clusters(vers_old, vers_new, slack=slack)
    return vers_old, vers_new 

def addMark(src):
    input_docs = src.replace('\n\n', '\n\n#####')
    input_docs = input_docs.replace('[citation needed]', '')
    return input_docs

def fixMark(src):
    # input_docs = src.replace('\n\n', '.\n\n')
    input_docs = src.replace('#####', '.\n\n')
    return input_docs

# Main 

keep, add, sub = [], [], []
un_labeled = []
for idx in range(len(train_pt)):
    old_ver = addMark(train_pt[idx]['document'])
    new_ver = addMark(train_pt[idx]['summary'])
    old, new = get_sentence_diff(old_ver, new_ver)
    un_labeled.append(new)
    ke, ad, su = 0, 0, 0
    for i in range(len(new)):
        if new[i]['tag']==' ': ke+=1
        elif new[i]['tag']=='+': ad+=1
        else: su+=1
    keep.append(ke)
    add.append(ad)
    sub.append(su)

with open('output.txt', 'w') as fwrite:
    keep_mean = np.mean(keep)
    keep_max = np.max(keep)
    add_mean = np.mean(add)
    add_max = np.max(add)
    sub_mean = np.mean(sub)
    sub_max = np.max(sub)
    fwrite.write(f'The mean of keep is {keep_mean}, mean of add is {add_mean}, mean of sub is {sub_mean}, max of each is {keep_max}, {add_max}, {sub_max}')

df_keep = pd.DataFrame(keep)
df_add = pd.DataFrame(add)
df_sub = pd.DataFrame(sub)
df_keep.to_csv('log_keep.csv', header=False, index=False)
df_add.to_csv('log_add.csv', header=False, index=False)
df_sub.to_csv('log_sub.csv', header=False, index=False)

def Labeling(new):
    labeled_data = []
    for idx in range(len(new)):
        if new[idx]['tag']==' ' and new[idx]['text']!='':
            # labeled_data.append(' [KEEP] ' + new[idx]['text'] + ' [/KEEP]')
            labeled_data.append(' [KEEP] ' + new[idx]['text'])
        # elif new[idx]['tag']=='-' and new[idx]['text']!='':
        #     labeled_data.append(' [RM] '+new[idx]['text']+' [/RM]')
        elif new[idx]['tag']=='+' and new[idx]['text']!='': 
            # labeled_data.append(' [ADD] '+new[idx]['text']+' [/ADD]')
            labeled_data.append(' [ADD] '+new[idx]['text'])
        # elif new[idx]['tag']=='*' and new[idx]['text']!='':
        else:
            # labeled_data.append(' [SUB] '+new[idx]['text']+' [/SUB]')
            labeled_data.append(' [SUB] '+new[idx]['text'])
    return ''.join(labeled_data)

labeled = []
for idx in range(len(train_pt)):
     labeled.append(Labeling(un_labeled[idx]))

wrap = []
for idx in range(len(train_pt)):
    idx_content = {}
    idx_content['content'] = labeled[idx].replace('.\n\n#####', '.\n\n')
    wrap.append(idx_content)
torch.save(wrap, './train_labeled.pt')
