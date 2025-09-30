# step: 
describe_image.py ->

(merge_img_describe.py) -> 

cleaned_text.py -> 

train_bertopic.py ->

docTopic.py



## describe_image.py
在 google cloud control 上申請 gemini-2.0-flash 的 api，並針對每張照片進行詳細描述

{ prompt 重點：

    詳細描述圖片，重點放在主要物體及其背景。
    
    使用英文，提供完整的描述性回复
    
    不要使用項目符號。

}



## (merge_img_describe.py)
不一定會使用到，用於合併 image_descriptions.csv



## cleaned_text.py
因為 gemini 對於圖片的描述是口語化的，存在不重要的介係詞等內容，故需先進行內容的整理

{ 

    在 remove_stopwords 中清除英語停用詞；後續 build_phrases 以 gensim Phrases 訓練雙詞與三詞片語，捕捉常見詞組並轉成 Phraser。
    
    apply_phrases 將片語模型套回每筆 token，把常見連續詞合併成單一 token（例如 bigram trigram）。

    lemmatize_and_pos_filter 透過 spaCy 詞形還原，只保留名詞/形容詞/副詞，片語中的單詞各自還原後再重新組合。
    
    frequency_filter_docs 使用 CountVectorizer 以文件頻率篩選 token（出現比例 <1.5% 或 >80% 的詞會被移除），減少過稀或過常的詞。
    
    load_and_clean 串起上述流程：讀取 CSV Description 欄位→tokenize→去停用詞→訓練並套用片語→詞形還原與詞性篩選→文件頻率過濾，最後回傳乾淨 token。
    
}



## train_bertopic.py
使用預處理過的資料 fine tune bertopic

輸出 csv ，紀錄每個 topic 的關鍵字



## docTopic.py
讓已經 fine tune 後的 model 判斷每篇文章的 topic 占比