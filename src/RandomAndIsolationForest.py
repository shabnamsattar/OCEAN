class RandomAndIsolationForest:
    def __init__(self, randomForest=None, isolationForest=None):
        """
        Combines a randomForest and an isolationForest inputs into
        a completeForest object.
        """
        self.randomForest = randomForest
        self.isolationForest = isolationForest

        self.estimators_ = []
        self.randomForestEstimatorsIndices = []
        self.isolationForestEstimatorsIndices = []

        self.n_estimators = 0

        if randomForest:
            self.n_estimators += randomForest.n_estimators
            self.randomForestEstimatorsIndices = list(range(randomForest.n_estimators))
            self.estimators_ += randomForest.estimators_

        if isolationForest:
            offset = len(self.estimators_)
            self.n_estimators += isolationForest.n_estimators
            self.isolationForestEstimatorsIndices = [
                i + offset for i in range(isolationForest.n_estimators)
            ]
            self.estimators_ += isolationForest.estimators_
