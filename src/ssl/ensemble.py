from typing import Any, Dict, List

import numpy as np
from sklearn.metrics import accuracy_score

from src.utils import compare_labels


class Ensemble:
    """
    Classe responsável por criar um comitê de classificadores e implementar
    seus métodos
    """

    def __init__(self, ssl_algorithm: callable, **params: Dict[str, Any]):
        self.ensemble: list[str] = []
        self.weights: list = []
        self.is_weights = params.get('is_weight', False)
        self.ssl_algorithm = ssl_algorithm
        self.ssl_params = params.get('ssl_params', {})
        self.q_measure_funcs = {
            "classifier_ensemble": self.q_measure_classifier_vs_ensemble,
            "classifier_classifier": self.q_measure_classifier_vs_classifier,
        }

    def add_classifier(self, classifier, need_train: bool = True):
        """
        Adiciona um classificador ao comitê, podendo já estar treinado
        ou não.

        Parameters
        ----------
        classifier : Classifier
            Classificador a ser adicionado no comitê.
        need_train : bool, optional
            Flag para indicar se é necessário treinar o classificador
            com o algoritmo semissupervisionado, por default True.
        """

        if need_train:
            self.ssl_params["base_estimator"] = classifier
            flexconc = self.ssl_algorithm(self.ssl_params)
        else:
            flexconc = classifier
        self.ensemble.append(flexconc)
        self.weights.append(1)

    def add_fit_classifier(
        self,
        classifier,
        instances: np.ndarray,
        classes: np.ndarray
    ):
        """
        Treina um classificador e o adiciona no comitê já existente,
        evitando ter que treinar todo o comitê novamente.

        Parameters
        ----------
        classifier : Classifier
            Classificador a ser treinado e adicionado.
        instances : ndarray
            Instâncias para serem utilizadas no treinamento.
        classes : ndarray
            Rótulos dos instâncias.
        """
        classifier.fit(instances, classes)
        self.ensemble.append(classifier)
        self.weights.append(1)

    def remover_classifier(self, classifier):
        """
        Remove um classificador específico do comitê atual.

        Parameters
        ----------
        classifier : Classificador
            Objeto classificador a ser removido do comitê.

        Raises
        ------
        ValueError
            Quando o classificador a ser removido não existe na lista.
        """
        try:
            self.weights.pop(self.ensemble.index(classifier))
            self.ensemble.remove(classifier)
        except ValueError as err:
            raise ValueError(
                f"Classificador não existe no comitê.\nErro: {err}"
            ) from err

    def measure_ensemble(
        self,
        instances: np.ndarray,
        classes: np.ndarray
    ) -> List[float]:
        """
        Calcula a acurácia dos classificadores base do comitê.

        Parameters
        ----------
        instances : ndarray
            Instâncias para serem utilizadas no treinamento.
        classes : ndarray
            Rótulos dos instâncias.

        Returns
        -------
        List[float]
            Acurácia de cada classificador base do comitê.
        """
        ensemble_metric = []

        for classifier in self.ensemble:
            y_pred = self.predict_one_classifier(classifier, instances)
            ensemble_metric.append(accuracy_score(classes, y_pred))

        return ensemble_metric

    def drop_ensemble(self):
        """Esvazia o comitê de classificadores"""
        self.ensemble = []
        self.weights = []

    def fit_ensemble(self, instances: np.ndarray, classes: np.ndarray):
        """
        Treina todos os classificadores de uma vez, com um único
        conjunto de instâncias e classes. Essa função chama n vezes a
        função `fit_single_classifier`.

        Parameters
        ----------
        instances : ndarray
            Instâncias para serem utilizadas no treinamento.
        classes : ndarray
            Rótulos dos instâncias.
        """

        for classifier in self.ensemble:
            self.fit_single_classifier(classifier, instances, classes)

    def fit_single_classifier(
        self,
        classifier,
        instances: np.ndarray,
        classes: np.ndarray
    ):
        """
        Treina um classificador a partir de um conjunto de instâncias e
        seus respectivos rótulos

        Parameters
        ----------
        classifier : Classifier
            Classificador a ser treinado.
        instances : ndarray
            Instâncias para serem utilizadas no treinamento.
        classes : ndarray
            Rótulos dos instâncias.

        Returns
        -------
            Classificador treinado e pronto para ser utilizado para
            predizer novas instâncias da base de dados.
        """

        return classifier.fit(instances, classes)

    def predict_one_classifier(
        self,
        classifier,
        instances: np.ndarray
    ) -> List[int]:
        """
        Realiza a predição de um único classificador a partir de um
        grupo de instâncias.

        Parameters
        ----------
        classifier : Classifier
            Classificador a ser utilizado para predizer as instâncias.
        instances : ndarray
            Instâncias para serem preditas.

        Returns
        -------
        List[int]
            Lista com os rótulos das instâncias que foram preditas pelo
            classificador
        """
        y_pred = classifier.predict(instances)

        return y_pred

    def predict_ensemble(self, instances: np.ndarray) -> List[int]:
        """
        Realiza a predição de instâncias do comitê. A estratégia de
        agregação é a votação simples.

        Parameters
        ----------
        instances : np.ndarray
            Instâncias a serem avaliadas.

        Returns
        -------
        List[int]
            Predição do comitê de classificadores, a partir de uma
            votação simples para decidir os rótulos das instâncias.
        """
        y_pred = []

        for instance in instances:
            pred = []

            for classifier in self.ensemble:
                pred.append(
                    classifier.predict(instance.reshape(1, -1)).tolist()[0]
                )
            aux = {}
            for y_cl, w in zip(pred, self.weights):
                if y_cl in aux.keys():
                    aux[y_cl] += w
                else:
                    aux[y_cl] = w
            y_pred.append(max(aux, key=aux.get))

        return np.array(y_pred, dtype="int64")

    def update_weights(
        self,
        instances: np.ndarray,
        classes: np.ndarray,
        metric: callable = None,
        **metric_args,
    ) -> None:
        """Realiza o ajuste nos pesos dos classificadores base do
        comitê. Por meio da eficácia de classificação dos modelos.
        Caso não seja informada nenhuma métrica, será utilizada a
        acurácia. Além disso, é interessante pensar em métricas na
        escala [0, 1].

        Parameters
        ----------
        instances : np.ndarray
            Instâncias a serem utilizadas no treinamento.
        classes : np.ndarray
            Rótulos das instâncias.
        metric : callable
            Métrica a ser utilizada.
        """
        if metric is None:
            metric = accuracy_score
        for i, cl in enumerate(self.ensemble):
            y_pred = self.predict_one_classifier(cl, instances)
            self.weights[i] = metric(classes, y_pred, **metric_args)

    def swap(
        self,
        classifier: List,
        pos: List[int],
        instances: np.ndarray,
        classes: np.ndarray,
        retrain: bool = True,
    ) -> None:
        """
        Função para trocar um classificador do comitê por outro já
        treinado.

        Parameters
        ----------
        classifier : List
            Lista de classificadores, ou classificador, para serem
            incluídos no comitê.
        pos : List[int]
            Lista indicando quais classificadores do comitê devem ser
            substituídos pelos novos.
        """

        if len(pos) == len(classifier):
            for i, j in enumerate(pos):
                if retrain:
                    self.ensemble[i].partial_fit(instances, classes)
                else:
                    self.ensemble[i] = classifier[j]
        else:
            for i in pos:
                if retrain:
                    self.ensemble[i].partial_fit(instances, classes)
                else:
                    self.ensemble[i] = classifier[0]

    def partial_fit(
        self,
        instances: np.ndarray,
        classes: np.ndarray,
        labels=None,
        sample_weight=None,
    ):
        """
        Função que realiza um treinamento parcial dos classificadores
        base do comitê.

        Parameters
        ----------
        instances : np.ndarray
            Instâncias a serem utilizadas no treinamento.
        classes : np.ndarray
            Rótulos das instâncias.
        """

        for model in self.ensemble:
            model.partial_fit(instances, classes, labels, sample_weight)

    def fit(
        self,
        instances: np.ndarray,
        classes: np.ndarray,
        labels=None,
        sample_weight=None,
    ):
        """
        Função para fazer o treinamento dos classificadores do comitê.

        Parameters
        ----------
        instances : np.ndarray
            Instâncias a serem utilizadas no treinamento.
        classes : np.ndarray
            Rótulos das instâncias.
        """

        for model in self.ensemble:
            model.fit(instances, classes, labels, sample_weight)

        return self

    def compute_pareto_frontier(
        self,
        instances: np.ndarray,
        classes: np.ndarray,
        q_measure_func: str,
        minimization: bool = False,
        **q_measure_params,
    ):
        """_summary_

        Parameters
        ----------
        instances : np.ndarray
            instâncias a serem comparadas.
        classes : np.ndarray
            classes das instâncias.
        """
        acc = self.measure_ensemble(instances, classes)
        q_measure = (
            1
            - abs(
                self.q_measure_funcs.get(
                    q_measure_func, self.q_measure_classifier_vs_ensemble
                )(instances, classes, **q_measure_params)
            )
        ).tolist()
        measures = [tuple((x, y)) for x, y in zip(acc, q_measure)]
        ensemble_classifier = self.pareto_frontier(measures, minimization)
        best_cls = list(set(measures).intersection(ensemble_classifier))
        best_cls_pos = [measures.index(cl) for cl in best_cls]
        new_ensemble = []

        for cl in best_cls_pos:
            new_ensemble.append(self.ensemble[cl])

        self.ensemble = new_ensemble

    def pareto_frontier(
        self,
        measures: List[tuple],
        minimization: bool = False
    ) -> List[tuple]:
        """
        TODO: realizar o 1 - abs(classifier_similarity()) e então
            computar o Pareto disso. Ao computar o Pareto, manter
            apenas as soluções que são não dominadas, ou seja, estão
            na primeira fronteira de Pareto.
        """

        non_dominated = []
        dominated = []

        for x, y in measures:
            if tuple((x, y)) not in dominated:
                # calcula os pontos dominados pelo atual...

                if self.check(
                    (x, y), list(set(measures) - set(dominated)), minimization
                ):
                    dominated.append((x, y))
                else:
                    non_dominated.append((x, y))
            else:
                pass

        return non_dominated

    def check(
        self,
        dot: tuple,
        data: List[tuple],
        minimization: bool = False,
    ) -> bool:
        if dot in data:
            data.remove(dot)

        if minimization:
            for x, y in data:
                if x < dot[0] and y < dot[1]:
                    return True

            return False
        else:
            for x, y in data:
                if x > dot[0] and y > dot[1]:
                    return True

            return False

    def q_measure_classifier_vs_ensemble(
        self,
        instances: np.ndarray,
        classes: np.ndarray,
        **params,
    ) -> np.ndarray:
        """

        Parameters
        ----------
        instances : np.ndarray
            instâncias a serem comparadas.
        classes : np.ndarray
            classes das instâncias.

        Returns
        -------
        List[float]
            _description_
        """
        similarity = []
        ensemble_labels = self.predict_ensemble(instances)
        ensemble_pred = compare_labels(classes, ensemble_labels)

        for classifier in self.ensemble:
            classifier_labels = self.predict_one_classifier(
                classifier,
                instances
            )
            classifier_pred = compare_labels(classes, classifier_labels)

            similarity.append(
                self._evaluate_similarity(ensemble_pred, classifier_pred)
            )

        return np.array(similarity)

    def q_measure_classifier_vs_classifier(
        self,
        instances: np.ndarray,
        classes: np.ndarray,
        **params
    ) -> np.ndarray:
        """_summary_

        Parameters
        ----------
        instances : np.ndarray
            instâncias a serem comparadas.
        classes : np.ndarray
            classes das instâncias.
        average : bool, optional
            deve utilizar o menor valor da matriz, by default False.
        average : bool, optional
            deve utilizar o valor absoluto da medida, by default False.

        Returns
        -------
        np.ndarray
            lista com os resultado da medida de diversidade Q.
        """
        similarity = np.array(
            [
                [0.0 for _ in range(len(self.ensemble))]

                for _ in range(len(self.ensemble))
            ]
        )

        for cl1 in range(len(self.ensemble) - 1):
            cl1_labels = self.predict_one_classifier(
                self.ensemble[cl1], instances
            )
            cl1_pred = compare_labels(classes, cl1_labels)

            for cl2 in range(cl1 + 1, len(self.ensemble)):
                cl2_labels = self.predict_one_classifier(
                    self.ensemble[cl2], instances
                )
                cl2_pred = compare_labels(classes, cl2_labels)

                cl_similarity = self._evaluate_similarity(cl1_pred, cl2_pred)
                similarity[cl1, cl2] = cl_similarity
                similarity[cl2, cl1] = cl_similarity

        absolute = params.get("absolute", False)

        if params.get("average", False):
            return np.array(
                [
                    np.mean(abs(i)) if absolute else np.mean(i)

                    for i in similarity
                ]
            )

        return np.array(
            [min(abs(i)) if absolute else min(i) for i in similarity]
        )

    def _evaluate_similarity(
        self,
        ensemble_pred: np.ndarray,
        classifier_pred: np.ndarray,
    ) -> float:
        """
        Calcula a similaridade entre o output do classificador com o
        output do comitê para validar o nível de concordância.

        Parameters
        ----------
        ensemble_pred : np.ndarray
            Predição do comitê
        classifier_pred : np.ndarray
            Predição do classificador

        Returns
        -------
        float
            Valor da similaridade entre as duas predições.
        """
        n11, n10, n00, n01 = 0, 0, 0, 0

        for ensemble_lab, class_lab in zip(ensemble_pred, classifier_pred):
            if ensemble_lab == class_lab:
                if ensemble_lab + class_lab:
                    n11 += 1
                else:
                    n00 += 1
            else:
                if ensemble_lab > class_lab:
                    n10 += 1
                else:
                    n01 += 1

        try:
            return (n11*n00 - n01*n10) / (n11*n00 + n01*n10)
        except ZeroDivisionError:
            print(f"N11 = {n11}\nN00 = {n00}\nN10 = {n10}\nN01 = {n01}\n")
            return 1
