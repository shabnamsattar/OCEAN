class IsolationOnlyForest:
    def __init__(self, isolationForest):
        self.isolationForest = isolationForest
        self.n_estimators = isolationForest.n_estimators
        self.estimators_ = [est for est in isolationForest.estimators_]
        # indices used by RandomForestCounterfactualMilp helpers
        self.isolationForestEstimatorsIndices = list(range(self.n_estimators))
        # For RF-only loops that iterate over "all trees"
        self.randomForestEstimatorsIndices = []  # empty
