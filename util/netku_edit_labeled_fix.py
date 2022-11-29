import torch

train_labeled_pt = torch.load('../dataset/train_log/train_labeled.pt')
test_labeled_pt = torch.load('../dataset/test_log/test_labeled.pt')
val_labeled_pt = torch.load('../dataset/val_log/val_labeled.pt')

wrap_pt = []
for idx in range(len(train_labeled_pt)):
    idx_content = {}
    train_labeled_pt[idx]['content'] = train_labeled_pt[idx]['content'].replace('# ####', '#####')
    train_labeled_pt[idx]['content'] = train_labeled_pt[idx]['content'].replace('##### ', '#####')
    train_labeled_pt[idx]['content'] = train_labeled_pt[idx]['content'].replace(' [KEEP] ### [KEEP] ##', '.##### [KEEP]')
    train_labeled_pt[idx]['content'] = train_labeled_pt[idx]['content'].replace('.##### [KEEP] [KEEP] ', '.##### [KEEP] ')
    train_labeled_pt[idx]['content'] = train_labeled_pt[idx]['content'].replace('[KEEP] #####', '##### [KEEP] ')
    train_labeled_pt[idx]['content'] = train_labeled_pt[idx]['content'].replace('##### [KEEP] [KEEP] ', '##### [KEEP] ')
    train_labeled_pt[idx]['content'] = train_labeled_pt[idx]['content'].replace(' ##### [KEEP]', '.##### [KEEP]')
    train_labeled_pt[idx]['content'] = train_labeled_pt[idx]['content'].replace('..#####', '.#####')
    train_labeled_pt[idx]['content'] = train_labeled_pt[idx]['content'].replace(' [ADD] #####', '##### [ADD] ')
    train_labeled_pt[idx]['content'] = train_labeled_pt[idx]['content'].replace(' [SUB] #####', '##### [SUB] ')
    train_labeled_pt[idx]['content'] = train_labeled_pt[idx]['content'].replace(' [KEEP] #### [KEEP] #', '##### [KEEP] ')
    train_labeled_pt[idx]['content'] = train_labeled_pt[idx]['content'].replace('[KEEP] [ADD]', '[KEEP]')

    idx_content['content'] = train_labeled_pt[idx]['content']
    wrap_pt.append(idx_content)
torch.save(wrap_pt, './train_labeled_fixed.pt')
