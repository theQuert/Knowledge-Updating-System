from rouge import FilesRouge

files_rouge = FilesRouge()
# hyp_path = "/home/quert/edit_NetKu/dataset/same_secs_insert_labeled/final_hyp.src"
hyp_path = "/home/quert/edit_NetKu/dataset/same_secs_insert_labeled/restricted/extracted_hyp.src"
# ref_path = "/home/quert/edit_NetKu/dataset/same_secs_insert_labeled/test_text.txt.tgt"
ref_path = "/home/quert/edit_NetKu/dataset/same_secs_insert_labeled/restricted/extracted_ref.tgt"
scores = files_rouge.get_scores(hyp_path, ref_path, avg=True)
print(scores)
