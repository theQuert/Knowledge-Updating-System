import math
import numpy as np
import torch

def pars_split(pt_file):
	output_str = []
	for idx in range(len(pt_file)):
		single_sent = pt_file[idx]['document'].split('.\c\c')
		num_articles = math.ceil(len(single_sent)/10) # might be 0 if #articles<10
		check_less=0
		articles = []
		# define idex
		if num_articles==0: # number of paragraphs < 10
			start_ids = [0]
		else:
			start_ids = [x*10 for x in range(num_articles)]
		s = '.\c\c'
		for id in range(len(start_ids)):
			if len(start_ids)==1:
				single_str = s.join(single_sent)
			else:
				try:
					if len(single_sent)>=start_ids[id] and len(single_sent)<start_ids[id+1]: 
						single_sent_lst = single_sent[start_ids[id]:len(single_sent)]
						single_str = s.join(single_sent_lst)
					if len(single_sent)>start_ids[id] and len(single_sent)>start_ids[id+1]: 
						single_sent_lst = single_sent[start_ids[id]:start_ids[id+1]]
						single_str = s.join(single_sent_lst)
				except:
					if len(single_sent)>start_ids[id]:
						single_sent_lst = single_sent[start_ids[id]:len(single_sent)]
						single_str = s.join(single_sent_lst)
					else:
						single_str = str(single_sent[-1])
			articles.append(single_str)
		output_str.append(articles)
	return output_str

def wrap_pt(input_str, pt_file):
	wrap = []
	input_src = input_str.copy()
	for idx in range(len(input_src)):
		idx_content = {}
		idx_content['document'] = input_src[idx]
		idx_content['summary'] = pt_file[idx]['summary']
		wrap.append(idx_content)
	torch.save(wrap, '/home/quert/edit_NetKu/util/4096_1024_fbbart/wcepvalid.pt')

pt_file = torch.load('/home/quert/edit_NetKu/util/4096_1024_fbbart/wcepvalid_trunc.pt')
input_str = pars_split(pt_file)
wrap_pt(input_str, pt_file)

