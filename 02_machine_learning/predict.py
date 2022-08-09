import torch
import torchvision.models as models
from torchvision import transforms
import os
from PIL import Image
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import torch.utils.data as data
import torch.nn as nn
import torch.optim as optim
import glob
import time

class ImageTransform(object):
    """
    入力画像の前処理クラス
    画像のサイズをリサイズする
    
    Attributes
    ----------
    resize: int
        リサイズ先の画像の大きさ
    mean: (R, G, B)
        各色チャンネルの平均値
    std: (R, G, B)
        各色チャンネルの標準偏差
    """
    def __init__(self, resize, mean, std):
        self.data_trasnform = {
            'train': transforms.Compose([
                # データオーグメンテーション
                transforms.RandomHorizontalFlip(),
                # 画像をresize×resizeの大きさに統一する
                transforms.Resize((resize, resize)),
                # Tensor型に変換する
                transforms.ToTensor(),
                # 色情報の標準化をする
                transforms.Normalize(mean, std)
            ]),
            'valid': transforms.Compose([
                # 画像をresize×resizeの大きさに統一する
                transforms.Resize((resize, resize)),
                # Tensor型に変換する
                transforms.ToTensor(),
                # 色情報の標準化をする
                transforms.Normalize(mean, std)
            ])
        }
    
    def __call__(self, img, phase='train'):
        return self.data_trasnform[phase](img)

class BirdDataset(data.Dataset):
    """
    鳥のDataseクラス。
    PyTorchのDatasetクラスを継承させる。
    
    Attrbutes
    ---------
    file_list: list
        画像のファイルパスを格納したリスト
    classes: list
        鳥の特徴のラベル名
    transform: object
        前処理クラスのインスタンス
    phase: 'train' or 'valid'
        学習か検証化を設定
    """
    def __init__(self, file_list, classes, transform=None, phase='train'):
        self.file_list = file_list
        self.transform = transform
        self.classes = classes
        self.phase = phase
    
    def __len__(self):
        """
        画像の枚数を返す
        """
        return len(self.file_list)
    
    def __getname__(self, index):
        return self.file_list[index]
    
    def __getitem__(self, index):
        """
        前処理した画像データのTensor形式のデータとラベルを取得
        """
        # 指定したindexの画像を読み込む
        img_path = self.file_list[index]
        img = Image.open(img_path)
        
        # 画像の前処理を実施
        img_transformed = self.transform(img, self.phase)
        
        # 画像ラベルをファイル名から抜き出す
        label = self.file_list[index].split('/')[2]
        
        # ラベル名を数値に変換
        label = self.classes.index(label)
        
        return img_transformed, label

#---------------モデルの読み込み---------------#

# 学習済みモデルの読み込み
model_path = './non-weight/model-4_valid.pth'  #epoch5のpretrainなしモデル
#model_path = './with-weight/model-4_valid.pth'#epoch5のpretrainありモデル
model_ft = models.resnet50()
model_ft.fc = nn.Linear(model_ft.fc.in_features, 2)
model_ft.load_state_dict(torch.load(model_path))
# GPUの利用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
net = model_ft.to(device)
net.eval()

#---------------評価---------------#
folder = '../03_testset/01_kawaii/'#可愛い鳥のテストセット
#folder = '../03_testset/02_kakkoii/'#かっこいい鳥のテストセット
files = glob.glob(folder+'*')
for item in files:
    print(item)
    time.sleep(1)
    img = Image.open(item)
    # リサイズ先の画像サイズ
    resize = 300
    # 今回は簡易的に(0.5, 0.5, 0.5)で標準化
    mean = (0.5, 0.5, 0.5)
    std = (0.5, 0.5, 0.5)

    transform = ImageTransform(resize, mean, std)
    img_transformed = transform(img, 'valid').unsqueeze(0)#次元を追加。batch化の代わり
    
    pred = []
    with torch.no_grad():
        output = net(img_transformed)
        _, preds = torch.max(output, 1)
        pred += [int(l.argmax()) for l in output]
        print(pred)#[0]だと「可愛い」判定、[1]だと「かっこいい」判定です
        m = nn.Softmax(dim=1)
        print(m(output))#可愛いとかっこいい判定の度合いを表す評価値です。

