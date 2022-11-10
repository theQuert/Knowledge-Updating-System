import torch

# train_pt = torch.load('./train_relabeled.pt')
test_pt = torch.load('./test_relabeled.pt')
val_pt = torch.load('./val_relabeled.pt')

def trunc_pt(relabeled_pt):
	# For train_pt, test_pt, val_pt
	wrap = []

	for idx in range(len(relabeled_pt)):
		wrap_lst = []
		# for id in range(len(relabeled_pt[idx]['document'])):
		s = ' '
			# filtered_str = s.join(train_pt[idx]['document'][id].split()[:1200])
		filtered_str = s.join(relabeled_pt[idx]['document'].split()[:4096])
		# wrap_lst.append(filtered_str)
		idx_content = {}
		idx_content['document'] = filtered_str 
		idx_content['summary'] = relabeled_pt[idx]['summary']
		wrap.append(idx_content)
	torch.save(wrap, 'wcepvalid_trunc.pt')

# Main
path = val_pt 
trunc_pt(path)
