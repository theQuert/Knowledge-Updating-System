import torch
# import nltk
# nltk.download('punkt')
# from nltk import word_tokenize, sent_tokenize
from transformers import AutoTokenizer, BartTokenizerFast

train_pt = torch.load('./train_relabeled.pt')
test_pt = torch.load('./test_relabeled.pt')
val_pt = torch.load('./val_relabeled.pt')

# train_pt = torch.load('/home/a97041304/MDS_PRIMER/primer/dataset/NetKu_git/train.pt')
# test_pt = torch.load('/home/a97041304/MDS_PRIMER/primer/dataset/NetKu_git/test.pt')
# val_pt = torch.load('/home/a97041304/MDS_PRIMER/primer/dataset/NetKu_git/val.pt')

model_pth = '/home/quert/MDS_PRIMER/primer/PRIMER_wcep/new'
tokenizer = AutoTokenizer.from_pretrained(model_pth)
# tokenizer = AutoTokenizer.from_pretrained('facebook/bart-base')

def trunc_pt(relabeled_pt):
	# For train_pt, test_pt, val_pt
	wrap = []

	for idx in range(len(relabeled_pt)):
		wrap_lst = []
		s = '.\c\c'
		sents = []
		for sent in relabeled_pt[idx]['document'].split('.\n\n'):
			sent = sent.replace('[ KEEP ]', '[KEEP]').replace('[ ADD ]', '[ADD]').replace('[ SUB ]', '[SUB]').replace(' .', '.').replace(' ,', ',').replace('[KEEP].', '. KEEP').replace(' .', '.')
			sent = sent.replace(' [ADD].', '. [ADD]').replace(' [KEEP].', '. [KEEP]').replace(' [SUB].', '. [SUB]')
			sent = sent.replace('KEEP', '[KEEP]').replace('ADD', '[ADD]').replace('SUB', '[SUB]')
			sent = sent.replace('[[KEEP]]', '[KEEP]').replace('[[ADD]]', '[ADD]').replace('[[SUB]]', '[SUB]')
			sent = sent.replace('[KEEP]', '<KEEP>').replace('[ADD]', '<ADD>').replace('[SUB]', '<SUB>')
			sents.append(sent)
		single = s.join(sents)
		encoded_single = tokenizer(single, truncation=True, max_length=4096, padding=True)
		single_out = tokenizer.decode(encoded_single['input_ids']).replace('<s>', '').replace('</s>', '')

		sents_sec = []
		for sent_sec in relabeled_pt[idx]['summary'].split('.\n\n'):
			sents_sec.append(sent_sec)
		single_sec = s.join(sents_sec)
		encoded_single_sec = tokenizer(single_sec, truncation=True, max_length=1024, padding=True)
		single_out_sec = tokenizer.decode(encoded_single_sec['input_ids']).replace('<s>', '').replace('</s>', '')

		idx_content = {}
		idx_content['document'] = single_out
		idx_content['summary'] = single_out_sec
		wrap.append(idx_content)
	torch.save(wrap, 'wcepvalid_trunc.pt')


# Main
path = val_pt 
trunc_pt(path)
