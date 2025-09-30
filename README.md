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