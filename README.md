

### 使い方

```bash
python pascalvoc_metrics.py \
    --gt_dir <ground truth の矩形の情報を表すテキストファイルがあるディレクトリ> \
    --gt_format xyrb \
    --det_dir <検出したの矩形の情報を表すテキストファイルがあるディレクトリ> \
    --det_format xyrb \
    --output <出力するディレクトリ>
```

`--gt_format`、`--det_format` は矩形の表現形式を指定します。

* "xyrb" の場合 (デフォルト)、矩形は左上及び右下の (x, y) 座標で表します。
* "xywh" の場合、矩形は左上の (x, y) 座標及び幅、高さで表します。

### ディレクトリ構成

画像ファイルに対する [ground truth](https://ejje.weblio.jp/content/ground+truth+data) 及び検出した矩形の情報は、同名のテキストファイルで `groundtruths` 及び `detections` ディレクトリに保存されているとします。
例えば、画像 `00001.jpg` に対する ground truth の矩形の情報は `groundtruths/00001.txt`、検出した矩形の情報は `detections/00001.txt` になります。

例: ground truth の矩形の情報を表すテキストファイルがあるディレクトリ

```
groundtruths
├── 00001.txt
├── 00002.txt
├── 00003.txt
├── 00004.txt
├── 00005.txt
├── 00006.txt
└── 00007.txt
```

例: 検出した矩形の情報を表すテキストファイルがあるディレクトリ

```
detections
├── 00001.txt
├── 00002.txt
├── 00003.txt
├── 00004.txt
├── 00005.txt
├── 00006.txt
└── 00007.txt
```

### ファイルの形式

各テキストファイルは以下の形式で1行に1つの矩形の情報が記載されているとします。

* "xyrb" の場合
  * ground truth の矩形: `<ラベル名> <xmin> <ymin> <xmax> <ymax>`
  * 検出した矩形: `<ラベル名> <スコア> <xmin> <ymin> <xmax> <ymax>`
* "xywh" の場合
  * ground truth の矩形: `<ラベル名> <xmin> <ymin> <width> <height>`
  * 検出した矩形: `<ラベル名> <スコア> <xmin> <ymin> <width> <height>`

例: ground truth の矩形の情報を表すテキストファイル

```
person 25 16 38 56
person 129 123 41 62
```

例: 検出した矩形の情報を表すテキストファイル

```
person 0.88 5 67 31 48
person 0.70 119 111 40 67
person 0.80 124 9 49 67
```
