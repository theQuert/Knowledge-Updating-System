### Ideas
- `src`: We keep the contents except of `trigger`, and replace '\n\n' with '\c\c'.
- `tgt`: We clean the contents, replace '\n\n' with '\c\c'.
- For ROUGE scoring...
- `rebuild_pt_without_trigger.py`: Rebuild pt file for `util/netku_edit.py` to do sentence-level difference labeling.
- After labeling, we get `train_labeled_without_trigger.pt`
- Then, we calculcate the edit actions with `count_edits.py`
