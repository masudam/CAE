# CAE
Convolutional Autoencoderをtensorflow2系(2.1)とPytorch(1.6.0)で実装してみた（python3.7）

データセットは、それぞれのライブラリのコードで用意してくれるcifar10を使用しました。

lossを下げることより中間層の次元圧縮を重視したためそこまで復元された画像はきれいではありませんが、実験など環境に多様性がない場合では十分だと思います（多分）。

もし復元を重視するならbatch normalizationを加えたり、中間で線形層を入れずに複数次元のまま復元することで、lossがより下がってきれいな復元画像になるでしょう。

基本的に同じパラメータで作成しましたが、pytochでtf2のConv2DTransposeと同じサイズになるように調整した結果、paddingとoutput_paddingを使用することとなり、微妙に異なるかもしれません。

## tensorflow2
run
```
python main_tf2.py
```
tf.keras.Sequentialでencoderとdecoderをそれぞれ構成して、callで出力（生成画像）と中間層の値を返すようにしました。
tf.keras.datasetsは最初からtrainとtestを分けて渡してくれるのでデータの用意が比較的楽ですが、初回はダウンロードがあります。

#### 元画像と生成画像
![元画像](https://github.com/masudam/CAE/blob/master/result_tf2/test_sample.png "元画像")![tensorflow2による生成画像（30epoch）](https://github.com/masudam/CAE/blob/master/result_tf2/image_at_epoch_0030.png "tensorflow2による生成画像（30epoch）")

## Pytorch
run
```
python main_.py
```
nn.Sequentialでencoderとdecoderをそれぞれ構成して、forwardで出力（生成画像）と中間層の値を返すようにしました。
datasetは特別な型に従って作り、train phaseとval phaseに分けて学習を行いました。データはdata/ディレクトリが作成されそこに自動でダウンロードされます。

### 元画像と生成画像
![元画像](https://github.com/masudam/CAE/blob/master/result_torch/test_sample.png "元画像")![Pytorchによる生成画像（30epoch）](https://github.com/masudam/CAE/blob/master/result_torch/image_at_epoch_0030.png "Pytorchによる生成画像（30epoch）")

