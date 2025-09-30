import pandas as pd

# 讀取兩個 CSV 檔案
df1 = pd.read_csv('image_descriptions.csv')
df2 = pd.read_csv('image_descriptions_second.csv')

# 合併資料
df_all = pd.concat([df1, df2], ignore_index=True)

# 儲存合併後的 CSV
df_all.to_csv('image_descriptions_all.csv', index=False)