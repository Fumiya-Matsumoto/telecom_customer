## Structures
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
### configs
jsonファイルで書設定を記載。
* 利用している特徴量
* 学習器のパラメータなど

### data
dataフォルダはinput/outputに分けている。
#### input
元データのcsvファイルや、feather形式に変換したファイルなどを配置
#### output
提出用のcsvファイルを出力

### features
train/testから作成した各特徴量を保存。

特徴量管理のポイントは以下の3点
* Feather形式でシリアライズする
* 特徴量は基底クラスを継承して実装
* 特徴量作成スクリプトはargparseを利用してコマンドラインツールとして実行可能にする
#### Feather形式でシリアライズ
Featherは読み込みが非常に高速。C++で実装されており、Pythonのラッパーが提供されている。

Pythonで利用する場合は以下のコマンドでインストール可能。
```
$ pip install -U feather-format
```
これにより、pandasのDataFrameに対して、`df.to_feather(filepath)`とすることにより、Feather形式でのシリアライズが可能。

読み込みの場合は、`pd.read_feather(filepath)`を利用する。

#### 基底クラスの実装
同じようなコードを何度も書くことは、メンテナンスの都合上、望ましくないため、以下のような基底クラス継承して特徴量を作成することを考える。

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
