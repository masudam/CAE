# CAE
Convolutional Autoencoderをtensorflow2系(2.1)とPytorchで実装してみた

データセットは、それぞれのライブラリのコードで用意してくれるcifar10を使用しました。
lossを下げることより中間層の次元圧縮を重視したためそこまで復元された画像はきれいではありませんが、実験など環境に多様性がない場合では十分だと思います。
もし復元を重視するならbatch normalizationを加えたり、中間で線形層を入れずに複数次元のまま復元することで、lossがより下がってきれいな復元画像になるでしょう。

基本的に同じパラメータで作成しましたが、pytochでtf2のConv2DTransposeと同じサイズになるように調整した結果、paddingとoutput_paddingを使用することとなり、微妙に異なるかもしれません。

## tensorflow2


#### 元画像と生成画像
![元画像](https://github.com/masudam/CAE/blob/master/result_tf2/test_sample.png "元画像")![tensorflow2による生成画像（30epoch）](https://github.com/masudam/CAE/blob/master/result_tf2/image_at_epoch_0030.png "tensorflow2による生成画像（30epoch）")


### Pytorchでの結果
![元画像](https://github.com/masudam/CAE/blob/master/result_torch/test_sample.png "元画像")![Pytorchによる生成画像（30epoch）](https://github.com/masudam/CAE/blob/master/result_torch/image_at_epoch_0030.png "Pytorchによる生成画像（30epoch）")

