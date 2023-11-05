
import numpy as np
metrics = ['rouge1', 'rouge2', 'rougeL','BERTscore']
score_dict = {}
for metric in metrics:
    filename = rf'/home/akoirala/Thesis/Generating_summary/records/{metric}.npy'
    score_dict[metric] = np.load(filename)
print(score_dict)

# def remove_hash_concat(row):
#     def remove_unwanted_chars(text):
#         if isinstance(text, str):
#             cleaned_text = text.replace('#', '').replace('\n', '')
#             return cleaned_text
#         return text

#     for col in ['ground_truth_summary','docs_sent','input_seq','Masked Background']:
#         row[col] = remove_unwanted_chars(row[col])
#     return row

# import pandas as pd
# df = pd.read_csv(r'/home/akoirala/Thesis/Testing/result/model_with_test/PICO_0.15_test.csv')
# #df = remove_hash_concat(df)
# df_final = df.apply(remove_hash_concat, axis=1)
# print("# have been removed")
# df_final.to_csv(r'/home/akoirala/Thesis/Testing/result/model_with_test/PICO_0.15_test.csv',index=False)
