def remove_hash_concat(row):
    def remove_unwanted_chars(text):
        if isinstance(text, str):
            cleaned_text = text.replace('#', '').replace('\n', '')
            return cleaned_text
        return text

    for col in ['docs_sent', 'Masked Background', 'ground_truth_summary','input_seq']:
        row[col] = remove_unwanted_chars(row[col])
    return row
import pandas as pd

input_df = pd.read_csv(r'/home/akoirala/Thesis/Data-Preprocessing-Pipeline/result/PICO_0.15_dev.csv')
input_df = input_df.apply(remove_hash_concat, axis=1)
print("Done")
input_df.to_csv(r'/home/akoirala/Thesis/Data-Preprocessing-Pipeline/result/PICO_0.15_dev.csv',index=False)
print("saving output")