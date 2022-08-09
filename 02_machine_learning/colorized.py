from PIL import Image
import numpy as np
import glob

def colorize_files(file_path):
    file_list = glob.glob(file_path+'*')
    for item in file_list:
        img = Image.open(item)
        color = img.convert("RGB")
        color.save(item) #★元の画像が上書きされるので注意！

# カラー化したい画像データを入れたフォルダを指定する
file_path = '******/'
# カラー化処理関数
colorize_files(file_path)

