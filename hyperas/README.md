# Hyperas - 在Keras中自動選擇超參數
deep learning做到後面都剩下調參數
而參數又不是那麼容易調整，是個廢力又廢時的工作
這邊將介紹透過Hyperas這個套件，自動選擇符合Model最好的參數

## 安裝Hyperas
使用`pip`進行安裝

```
$ pip install hyperas
```

## Import Hyperas

```python
from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform
```

## Import Keras
```python
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop

from keras.datasets import mnist
from keras.utils import np_utils
```

之後我們會依序
1. 定義Data
2. 定義Model
3. Optimize model hyperparameters

## 定義Data
使用`MNIST`的data

```python
def data():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train.reshape(60000, 784)
    X_test = X_test.reshape(10000, 784)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255
    nb_classes = 10
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)
    return X_train, Y_train, X_test, Y_test
```
## 定義Model
這邊除了定義Model外，還需完成training及testing，所以需把data傳進來
最後回傳一個dictionary，其中包含：
* loss: Hyperas會去選擇最小值的model
* status: 直接回傳`STATUS_OK`
* model: 可不回傳(option)

```python
def create_model(X_train, Y_train, X_test, Y_test):
    model = Sequential()
    model.add(Dense(512, input_shape=(784,)))
    model.add(Activation('relu'))
    model.add(Dropout({{uniform(0, 1)}}))
    model.add(Dense({{choice([256, 512, 1024])}}))
    model.add(Activation('relu'))
    model.add(Dropout({{uniform(0, 1)}}))
    model.add(Dense(10))
    model.add(Activation('softmax'))

    rms = RMSprop()
    model.compile(loss='categorical_crossentropy', optimizer=rms, metrics=['accuracy'])
    
    model.fit(X_train, Y_train,
              batch_size={{choice([64, 128])}},
              nb_epoch=1,
              verbose=2,
              validation_data=(X_test, Y_test))
    score, acc = model.evaluate(X_test, Y_test, verbose=0)
    print('Test accuracy:', acc)
    return {'loss': -acc, 'status': STATUS_OK, 'model': model}
```

原本Dropout需要傳入一個0-1的機率
但我們這邊不直接指定一個數字
而是透過`uniform`幫我們產生一個0-1的數字

```python
model.add(Dropout({{uniform(0, 1)}}))
```

Dense擇是透過`choice`
傳入我們要哪些值

```python
model.add(Dense({{choice([256, 512, 1024])}}))
```

最後回傳的dictionary
我們目標是選擇最高的accuracy的model
但因為Huperas他會去選擇`loss`這個value **最小的** 的model
所以將accuracy直接變 **負號**
再丟給`loss`

```python
return {'loss': -acc, 'status': STATUS_OK, 'model': model}
```

## Optimize model hyperparameters
最後透過`optim.minimize()`來找出最好的model
* model: 我們定義的model
* data: 我們定義的data
* algo: 使用TPE algorithm
* max_evals: evaluation次數

```python
X_train, Y_train, X_test, Y_test = data()

best_run, best_model = optim.minimize(model=create_model,
                                      data=data,
                                      algo=tpe.suggest,
                                      max_evals=5,
                                      trials=Trials())

print("Evalutation of best performing model:")
print(best_model.evaluate(X_test, Y_test))
```

## 最後
1. Hyperas好像跟註解很不合，在跑程式時需把註解都刪掉，以免發生錯誤
2. 如果是使用jupyter notebook需在`optim.minimize()`多加入`notebook_name`這個參數且設定為`ipynb`的檔名，假如目前為`Untitled.ipynb`就設定為：
    
```python
best_run, best_model = optim.minimize(model=create_model,
                                      data=data,
                                      algo=tpe.suggest,
                                      max_evals=5,
                                      trials=Trials(),
                                      notebook_name='Untitled')
```

## 參考
https://github.com/maxpumperla/hyperas



