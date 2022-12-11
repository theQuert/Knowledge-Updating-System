- `train.pt`: paragraphs are separated into '. \\c\\c'.
- `netku_edit.py`: Only the labeled data (without truncation) (paragraphs splitted with '#####'). Output: `train_labeled.pt`
- `netku_edit_labeled_fix.py`: Some `#####` is mis-replaced with `[KEEP] ### [KEEP] ## [KEEP]`, so we have to fix this to align paragraphs. Output: `train_labeled_fixed.pt`; #Paragraphs in the updated-version is more than the non-updated content. (DONE)
- `netku_edit_post.py`: Combine the labeled.pt and the updated information to construct ['document'], ['summary'], paragraphs splitted with '.\n\n' We may replace `. \n\n` to `.\n\n`, and replace `\n\n` with `\c\c`. Input: `train_labeled_fixed.pt` **Re**-Output: `train_relabeled.pt`
- `netku_edit_beftrunc.py`: Truncate the input and output for PRIMERA (input max_len is 4096, output max_len is 1024), we use the updated custom tokenizer. Using '.\c\c' to solve the PRIMERA problem to split paragraphs. Output: `wceptrain_trunc.pt`.
- `netku_edit_afttrunc.py`: Apply 10 paragrphs in a list to contrunct the ['summary'], and use the ['document'] from the last step. Output: `wceptrain.pt`

