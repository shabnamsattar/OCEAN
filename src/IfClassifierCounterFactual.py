import gurobipy as gp
from gurobipy import GRB
import numpy as np
from sklearn.ensemble._iforest import _average_path_length

from src.ClassifierCounterFactual import ClassifierCounterFactualMilp
from src.RandomForestCounterfactual import RandomForestCounterfactualMilp
from src.RandomAndIsolationForest import RandomAndIsolationForest
from src.CounterFactualParameters import TreeConstraintsType, BinaryDecisionVariables

class IfClassifierCounterFactualMilp(ClassifierCounterFactualMilp, RandomForestCounterfactualMilp):
    def __init__(self, classifier, sample, anomaly_threshold_log2,
                 objectiveNorm=2, verbose=False,
                 featuresType=False, featuresPossibleValues=False,
                 featuresActionnability=False, oneHotEncoding=False,
                 constraintsType=TreeConstraintsType.LinearCombinationOfPlanes,
                 binaryDecisionVariables=BinaryDecisionVariables.LeftRight_lambda):

        ClassifierCounterFactualMilp.__init__(
            self, classifier, sample, 0,  # outputDesired dummy (not used)
            objectiveNorm, verbose, featuresType, featuresPossibleValues,
            featuresActionnability, oneHotEncoding)

        RandomForestCounterfactualMilp.__init__(
            self, mutuallyExclusivePlanesCutsActivated=False,
            constraintsType=constraintsType,
            binaryDecisionVariables=binaryDecisionVariables)

        self.model.modelName = "IsolationForestCounterFactualMilp"
        self.completeForest = RandomAndIsolationForest(randomForest=None, isolationForest=classifier)
        self.anomaly_threshold_log2 = anomaly_threshold_log2
        self.isolationForest = classifier

    def __addAnomalyScoreConstraint(self):
        expr = gp.LinExpr(0.0)
        c = _average_path_length([self.isolationForest.max_samples_])[0]

        for t in self.completeForest.isolationForestEstimatorsIndices:
            tm = self.treeManagers[t]
            tree = self.completeForest.estimators_[t]
            for v in range(tm.n_nodes):
                if tm.is_leaves[v]:
                    depth = tm.node_depth[v] + _average_path_length([tree.tree_.n_node_samples[v]])[0]
                    expr += depth * tm.y_var[v] / self.completeForest.n_estimators

        log2_delta = self.anomaly_threshold_log2
        constant = -c * log2_delta
        self.model.addConstr(expr >= constant, name="log2_anomaly_score_constraint")

    def buildModel(self):
        self.initSolution()
        self._RandomForestCounterfactualMilp__buildTrees()
        self.__addAnomalyScoreConstraint()
        self.addActionnabilityConstraints()
        self.addOneHotEncodingConstraints()
        self.initObjective()

    def solveModel(self):
        self.model.write("if.lp")
        self.model.optimize()
        self.runTime = self.model.Runtime

        if self.model.status != GRB.OPTIMAL:
            self.objValue = "inf"
            self.x_sol = self.x0
            return False

        self.objValue = self.model.ObjVal
        self.x_sol = [[]]
        for f in range(self.nFeatures):
            self.x_sol[0].append(self.x_var_sol[f].getAttr(GRB.Attr.X))
        if self.verbose:
            print("Solution built\n", self.x_sol)
        self.__checkIfBadPrediction(self.x_sol)
        return True

    def getAnomalyScore(self):
        """Return the Isolation Forest anomaly score of the counterfactual solution (original s(x) ∈ (0,1])"""
        x = np.array(self.x_sol)
        path_lengths = []
        for estimator in self.isolationForest.estimators_:
            leaf_index = estimator.apply(x)[0]
            node_depth = 0
            while leaf_index != 0:
                parent = np.where((estimator.tree_.children_left == leaf_index) |
                                  (estimator.tree_.children_right == leaf_index))[0][0]
                leaf_index = parent
                node_depth += 1
            path_lengths.append(node_depth)
        avg_path_length = np.mean(path_lengths)
        c = _average_path_length([self.isolationForest.max_samples_])[0]
        return 2 ** (-avg_path_length / c)

    def __checkIfBadPrediction(self, x_sol):
        if self.verbose:
            score = self.getAnomalyScore()
            print("Anomaly score (s(x)):", score)
            print("log2(score):", np.log2(score))
            print("Target log2 threshold:", self.anomaly_threshold_log2)
            if np.log2(score) > self.anomaly_threshold_log2:
                print("Warning: counterfactual is too anomalous (violates constraint).")
            else:
                print("Counterfactual is plausible under the threshold.")
