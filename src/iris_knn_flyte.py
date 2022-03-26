import typing
import numpy as np
from flytekit import task, workflow
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier


@task
def preprocess() -> (typing.Dict[str, typing.List[typing.List[float]]], typing.Dict[str, typing.List[int]]):
    iris = load_iris()
    x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=22)
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)
    data = {'x_train': x_train.tolist(), 'x_test': x_test.tolist()}
    label = {'y_train': y_train.tolist(), 'y_test': y_test.tolist()}
    return data, label


@task
def ml(data: typing.Dict[str, typing.List[typing.List[float]]], label: typing.Dict[str, typing.List[int]]) -> float:
    estimator = KNeighborsClassifier(n_neighbors=9)
    estimator.fit(np.array(data['x_train']), np.array(label['y_train']))
    score = estimator.score(np.array(data['x_test']), np.array(label['y_test']))
    return score


@workflow
def iris_workflow() -> float:
    data, label = preprocess()
    score = ml(data=data, label=label)
    return score


print(f"iris_workflow knn algorithm output auc:{iris_workflow()}")
