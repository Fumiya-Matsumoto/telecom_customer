# 目的
1. Gitの扱いに慣れる
2. 機械学習でのフォルダ構成・管理方法を画一する
3. 仮想企業に対してAIを用いた課題解決・新規事業の提案を練習

# 1. Gitの扱いに慣れる
* 使用した教材
    * [Railsチュートリアル：Git/GitHub編](https://railstutorial.jp/git/getting_started#cha-getting_started)

# 2. フォルダ構成・管理方法を画一する

## フォルダ構成
<details><summary>詳細</summary><div>

```
.
├── configs
│   └── default.json
├── data
│   ├── input
│   │   ├── sample_submission.csv
│   │   ├── train.csv
│   │   └── test.csv
│   └── output
├── features
│   ├── __init__.py
│   ├── base.py
│   └── create.py
├── logs
│   └── logger.py
├── models
│   └── lgbm.py
├── notebooks
│   └── eda.ipynb
├── scripts
│   └── convert_to_feather.py
├── utils
│   └── __init__.py
├── .gitignore
├── .pylintrc
├── LICENSE
├── README.md
├── run.py
└── tox.ini
```

</div></details>

## configs
jsonファイルで書設定を記載。
* 利用している特徴量
* 学習器のパラメータなど

## data
dataフォルダはinput/outputに分けている。
### input
元データのcsvファイルや、feather形式に変換したファイルなどを配置
### output
提出用のcsvファイルを出力

## features
train/testから作成した各特徴量を保存するためのフォルダ
* 流れ
    1. trainとtestにデータが分割されていない場合には、scripts/train_test_split.pyを実行し、データを分割する
    2. scripts/convert_to_feather.pyでcsv形式からfeather形式に変換する
    3. create.pyで特徴量をfeatrer形式でfeaturesフォルダ内に保存

特徴量管理のポイントは以下の3点
* Feather形式でシリアライズする
* 特徴量は基底クラスを継承して実装
* 特徴量作成スクリプトはargparseを利用してコマンドラインツールとして実行可能にする

### Feather形式でシリアライズ
Featherは読み込みが非常に高速。C++で実装されており、Pythonのラッパーが提供されている。

Pythonで利用する場合は以下のコマンドでインストール可能。
```
$ pip install -U feather-format
```
これにより、pandasのDataFrameに対して、`df.to_feather(filepath)`とすることにより、Feather形式でのシリアライズが可能。

読み込みの場合は、`pd.read_feather(filepath)`を利用する。

### 基底クラスの実装
同じようなコードを何度も書くことは、メンテナンスの都合上、望ましくないため、以下のような基底クラス継承して特徴量を作成することを考える。

<details><summary>コード詳細</summary><div>

```
import re
import time
from abc import ABCMeta, abstractmethod
from pathlib import Path
from contextlib import contextmanager

import pandas as pd


@contextmanager
def timer(name):
    t0 = time.time()
    print(f'[{name}] start')
    yield
    print(f'[{name}] done in {time.time() - t0:.0f} s')


class Feature(metaclass=ABCMeta):
    prefix = ''
    suffix = ''
    dir = '.'
    
    def __init__(self):
        self.name = self.__class__.__name__
        self.train = pd.DataFrame()
        self.test = pd.DataFrame()
        self.train_path = Path(self.dir) / f'{self.name}_train.ftr'
        self.test_path = Path(self.dir) / f'{self.name}_test.ftr'
    
    def run(self):
        with timer(self.name):
            self.create_features()
            prefix = self.prefix + '_' if self.prefix else ''
            suffix = '_' + self.suffix if self.suffix else ''
            self.train.columns = prefix + self.train.columns + suffix
            self.test.columns = prefix + self.test.columns + suffix
        return self
    
    @abstractmethod
    def create_features(self):
        raise NotImplementedError
    
    def save(self):
        self.train.to_feather(str(self.train_path))
        self.test.to_feather(str(self.test_path))
```

</div></details>

#### timer()
処理時間を簡単に計測するための関数
#### Featureクラス
特徴量の基底クラス。

このクラスを継承して`create_features`メソッドを実装すれば利用可能になる。自身のクラス名から自動でファイル名を生成し、`save`メソッドで保存まで行うことができる。

利用例
```
class FamilySize(Feature):
    def create_features(self):
        self.train['family_size'] = train['SibSp'] + train['Parch'] + 1
        self.test['family_size'] = test['SibSp'] + test['Parch'] + 1

FamilySize().run().save()
```
このように書くと、`FamilySize_train.ftr`と`FamilySize_test.ftr`が作成される。
#### argparseを利用したコマンドラインツール化
特徴量を実装したらワンコマンドで実行できることが望ましい。また、すでに計算された特徴量は再度計算したくない。
```
import argparse
import inspect

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--force', '-f', action='store_true', help='Overwrite existing files')
    return parser.parse_args()


def get_features(namespace):
    for k, v in namespace.items():
        if inspect.isclass(v) and issubclass(v, Feature) and not inspect.isabstract(v):
            yield v()


def generate_features(namespace, overwrite):
    for f in get_features(namespace):
        if f.train_path.exists() and f.test_path.exists() and not overwrite:
            print(f.name, 'was skipped')
        else:
            f.run().save()
```
* get_arguments()
  * コマンドライン引数を解析する関数。`python hoge.py -f`のように`-f`オプションを付けることで、上書きモードになる。
* get_features()
  * `Feature`を継承したクラスをインスタンス化して返すイテレータ
* generate_features()
  * `namespace`を渡すと、そこに含まれる特徴量がすでに保存済みかどうか確認して、存在していない場合は計算する。

利用例
```
import pandas as pd

from .base import Feature, get_arguments, generate_features

Feature.dir = 'features'

class FamilySize(Feature):
    def create_features(self):
        self.train['family_size'] = train['SibSp'] + train['Parch'] + 1
        self.test['family_size'] = test['SibSp'] + test['Parch'] + 1


if __name__ == '__main__':
    args = get_arguments()

    train = pd.read_csv('input/train.csv')
    test = pd.read_csv('input/test.csv')

    generate_features(globals(), args.force)
```
pyファイルを実行すると、features内に`FamilySize_train.ftr`と`FamilySize_test.ftr`が保存される。
```
$ python titanic.py
[FamilySize] start
[FamilySize] done in 0 s
```
### 特徴量の読み込み
出力した特徴量は以下のようにして読み込むことができる
```
def load_datasets(feats):
    dfs = [pd.read_feather(f'features/{f}_train.ftr') for f in feats]
    X_train = pd.concat(dfs, axis=1)
    dfs = [pd.read_feather(f'features/{f}_test.ftr') for f in feats]
    X_test = pd.concat(dfs, axis=1)
    return X_train, X_test

feats = ['FamilySize', 'Hoge', 'Fuga', 'Piyo']
X_train, X_test = load_datasets(feats)
```
一番深いimportancesフォルダは、特徴量の重要度を出力するために用意。
## logs
logsフォルダには、計算の実行ごとに下記の情報を出力する
* 利用した特徴量
* trainのshape
* 学習器のパラメータ
* cvのスコア
実装
```
import logging

from lightgbm.callback import _format_eval_result


def log_evaluation(logger, period=1, show_stdv=True, level=logging.DEBUG):
    def _callback(env):
        if period > 0 and env.evaluation_result_list and (env.iteration + 1) % period == 0:
            result = '\t'.join([_format_eval_result(x, show_stdv) for x in env.evaluation_result_list])
            logger.log(level, '[{}]\t{}'.format(env.iteration+1, result))
    _callback.order = 10
    return _callback
```
### 使い方
ロガーをあらかじめ作成しておき、log_evaluationの引数を渡すと動く
```
# ロガーの作成
logger = logging.getLogger('main')
logger.setLevel(logging.DEBUG)
sc = logging.StreamHandler()
logger.addHandler(sc)
fh = logging.FileHandler('hoge.log')
logger.addHandler(fh)

# データのロードなどは省略

# 訓練時にコールバックのリストを渡す
clf = lgb.LGBMClassifier()
callbacks = [log_evaluation(logger, period=10)]
clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=callbacks)
```

## models
modelフォルダには学習器を用意する。
* 入力：pandas.DataFrame、パラメータ
* 出力

## notebook
探索用データ分析などで利用したJupyter Notebookを配置する。

ここで試行錯誤した結果を適切なフォルダ内のpythonファイルに取り込んでいく。

## scripts
汎用的なpythonファイルを配置する。
* convert_to_feather.py：csvファイルをfeathre形式のファイルに変換
* train_test_split.py：元データ（csv形式）をtrain.csvとtest.csvに変換

## utils
汎用的に使える関数を書いておく。

## 計算の実行
リポジトリのルートで以下を実行する。
```
python run.py
```

## 参考
* u++さん：[【Kaggleのフォルダ構成や管理方法】タイタニック用のGitHubリポジトリを公開しました](https://upura.hatenablog.com/entry/2018/12/28/225234)
* amaotoneさん：[Kaggleで使えるFeather形式を利用した特徴量管理法](https://amalog.hateblo.jp/entry/kaggle-feature-management)

# 3. AI事業の提案（シミュレーション）

