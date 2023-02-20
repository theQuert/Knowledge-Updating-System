import pandas as pd
from sklearn.metrics import classification_report,confusion_matrix

test_data = pd.read_csv('/home/quert/edit_NetKu/dataset/same_secs_insert_labeled/merged_updated_test.csv')
submission = pd.read_csv('/home/quert/edit_NetKu/util/submission.csv')

ans = test_data.target.values
pred = submission.target.values

report = classification_report(ans, pred, output_dict=True)
pd.DataFrame(report).transpose().to_csv("/home/quert/edit_NetKu/util/classification_report.csv")


