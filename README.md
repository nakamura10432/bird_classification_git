# 可愛い鳥とかっこいい鳥を二値分類するpytorchプログラム

Resnet50を用いた画像の二値分類スクリプトです。

フォルダ構成は下記の通り。

- 01_train：学習・検証用データを格納しています
 - 01_kawaii：可愛い鳥の画像フォルダ
 - 02_kakkoii：かっこいい鳥の画像フォルダ
- 02_machine_learning：学習・予測用のプログラムを格納しています
 - train.py：学習用スクリプト
 - predict.py：予測用スクリプト
 - colorized.py：画像を白黒からカラープロパティにする用
 - non-weight/：pretrainモデルを使用せず学習したモデル
 - with-weight/：pretrainモデルを使用して学習したモデル
- 03_testset：テスト用データセット
 - 01_kawaii：可愛い鳥テストセット
 - 02_kakkoii：かっこいい鳥テストセット

詳しくは下記のURLで説明記事をあげています。

https://www.data-flake.com/2022/08/09/bird_classification_resnet50_pytorch/