import numpy as np
import typing
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from flytekit import task, workflow
from sklearn.datasets import load_iris
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split


@task
def preprocess() -> (typing.Dict[str, typing.List[typing.List[float]]], typing.Dict[str, typing.List[typing.List[float]]]):
    iris_data = load_iris()
    x = iris_data.data
    y = iris_data.target.reshape(-1, 1)
    encoder = OneHotEncoder(sparse=False)
    y = encoder.fit_transform(y)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20)
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)
    data = {'x_train': x_train.tolist(), 'x_test': x_test.tolist()}
    label = {'y_train': y_train.tolist(), 'y_test': y_test.tolist()}
    return data, label


@task
def dl(data: typing.Dict[str, typing.List[typing.List[float]]],
       label: typing.Dict[str, typing.List[typing.List[float]]]) -> float:
    model = Sequential()
    model.add(Dense(10, input_shape=(4,), activation='relu', name='fc1'))
    model.add(Dense(10, activation='relu', name='fc2'))
    model.add(Dense(3, activation='softmax', name='output'))
    optimizer = Adam(lr=0.001)
    model.compile(optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    print('Neural Network Model Summary: ')
    print(model.summary())
    model.fit(np.array(data['x_train']), np.array(label['y_train']), verbose=2, batch_size=5, epochs=200)
    results = model.evaluate(np.array(data['x_test']), np.array(label['y_test']))
    score = float(results[1])
    return score


@workflow
def iris_workflow() -> float:
    data, label = preprocess()
    score = dl(data=data, label=label)
    return score


print(f"iris_workflow dl algorithm output auc:{iris_workflow()}")
