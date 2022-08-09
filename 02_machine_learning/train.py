import torch
import torchvision.models as models
from torchvision import transforms
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch.utils.data as data
import torch.nn as nn
import torch.optim as optim
import random

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

def make_filepath_list(file_path):
    """
    学習データ、検証データそれぞれのファイルへのパスを格納したリストを返す
    
    Returns
    -------
    train_file_list: list
        学習データファイルへのパスを格納したリスト
    valid_file_list: list
        検証データファイルへのパスを格納したリスト
    """
    train_file_list = []
    valid_file_list = []

    for top_dir in os.listdir(file_path):
        file_dir = os.path.join(file_path, top_dir)
        file_list = os.listdir(file_dir)
        random.shuffle(file_list)

        # 各犬種ごとに8割を学習データ、2割を検証データとする
        num_data = len(file_list)
        num_split = int(num_data * 0.8)

        train_file_list += [os.path.join(file_path, top_dir, file).replace('\\', '/') for file in file_list[:num_split]]
        valid_file_list += [os.path.join(file_path, top_dir, file).replace('\\', '/') for file in file_list[num_split:]]
    
    return train_file_list, valid_file_list

#---------------データ格納のテスト---------------#
# 画像データへのファイルパスを格納したリストを取得する
file_path = '../01_train/'
train_file_list, valid_file_list = make_filepath_list(file_path)

print('学習データ数 : ', len(train_file_list))
print('検証データ数 : ', len(valid_file_list))

#---------------学習セット作成のテスト---------------#
# 動作確認
# クラス名
bird_classes = [
    '01_kawaii',  '02_kakkoii'
]

# リサイズ先の画像サイズ
resize = 300

# 今回は簡易的に(0.5, 0.5, 0.5)で標準化
mean = (0.5, 0.5, 0.5)
std = (0.5, 0.5, 0.5)

# Datasetの作成
train_dataset = BirdDataset(
    file_list=train_file_list, classes=bird_classes,
    transform=ImageTransform(resize, mean, std),
    phase='train'
)

valid_dataset = BirdDataset(
    file_list=valid_file_list, classes=bird_classes,
    transform=ImageTransform(resize, mean, std),
    phase='valid'
)

#---------------データローダー作成---------------#
# バッチサイズの指定
batch_size = 64

# DataLoaderを作成
train_dataloader = data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True)

valid_dataloader = data.DataLoader(
    valid_dataset, batch_size=32, shuffle=False)

# 辞書にまとめる
dataloaders_dict = {
    'train': train_dataloader, 
    'valid': valid_dataloader
}

#---------------モデルの作成---------------#

# 学習済みモデルの読み込み
# Resnet50を重み付きで読み込むor読み込まない
model_ft = models.resnet50(pretrained = True)

# 最終ノードの出力を2に変更する(「可愛い」と「かっこいい」)
model_ft.fc = nn.Linear(model_ft.fc.in_features, 2)

# GPUの利用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
net = model_ft.to(device)

# 損失関数に交差エントロピーを利用
criterion = nn.CrossEntropyLoss()

# 最適化に関しては、いくつかのパターンを調べた結果、下記が一番結果がよかった
optimizer = optim.SGD(net.parameters(),lr=0.01,momentum=0.9)


#---------------学習---------------#
# エポック数
num_epochs = 5

for epoch in range(num_epochs):
    print('Epoch {}/{}'.format(epoch+1, num_epochs))
    print('-------------')
    
    for phase in ['train', 'valid']:
        
        if phase == 'train':
            # モデルを訓練モードに設定
            net.train()
        else:
            # モデルを推論モードに設定
            net.eval()
        
        # 損失和
        epoch_loss = 0.0
        # 正解数
        epoch_corrects = 0
        
        # DataLoaderからデータをバッチごとに取り出す
        for inputs, labels in dataloaders_dict[phase]:
            #inputsにcudaを当てはめる
            #inputs = inputs.to('cuda')
            # optimizerの初期化
            optimizer.zero_grad()
            
            # 学習時のみ勾配を計算させる設定にする
            with torch.set_grad_enabled(phase == 'train'):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = net(inputs)
                
                # 損失を計算
                loss = criterion(outputs, labels)
                
                # ラベルを予測
                _, preds = torch.max(outputs, 1)
                
                # 訓練時はバックプロパゲーション
                if phase == 'train':
                    # 逆伝搬の計算
                    loss.backward()
                    # パラメータの更新
                    optimizer.step()
                
                # イテレーション結果の計算
                # lossの合計を更新
                # PyTorchの仕様上各バッチ内での平均のlossが計算される。
                # データ数を掛けることで平均から合計に変換をしている。
                # 損失和は「全データの損失/データ数」で計算されるため、
                # 平均のままだと損失和を求めることができないため。
                epoch_loss += loss.item() * inputs.size(0)
                
                # 正解数の合計を更新
                epoch_corrects += torch.sum(preds == labels.data)

        # epochごとのlossと正解率を表示
        epoch_loss = epoch_loss / len(dataloaders_dict[phase].dataset)
        epoch_acc = epoch_corrects.double() / len(dataloaders_dict[phase].dataset)
        torch.save(net.state_dict(), 'model-'+str(epoch)+'_'+phase+'.pth')
        torch.save(net, 'model-'+str(epoch)+'_'+phase+'.ptori')
        print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
        