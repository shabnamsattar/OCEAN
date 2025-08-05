import gurobipy as gp
from gurobipy import GRB
import numpy as np
from sklearn.ensemble._iforest import _average_path_length
import math

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

    def __addAnomalyScoreConstraint(self,*, threshold=0.0):
        expr = gp.LinExpr(0.0)
        for t in self.completeForest.isolationForestEstimatorsIndices:
            tm   = self.treeManagers[t]
            tree = self.completeForest.estimators_[t]

            for v in range(tm.n_nodes):
                if tm.is_leaves[v]:
                    depth = (tm.node_depth[v] +
                         _average_path_length([tree.tree_.n_node_samples[v]])[0])
                    expr += (depth / self.completeForest.n_estimators) * tm.y_var[v]

    # ----------------------------------------------------------------------
    # 2.  Convert “decision ≥ threshold” to a linear inequality on ⟨h(x)⟩
    #     decision(x) = −2−⟨h(x)⟩/c  − offset_
    # ----------------------------------------------------------------------
        c_n  = _average_path_length([self.isolationForest.max_samples_])[0]
        delta = threshold + float(self.isolationForest.offset_)   # RHS inside brackets
        if delta >= 0:
            raise ValueError("threshold + offset_ must be negative for a valid cut-off")

        log2_delta = math.log2(-delta)          # log₂(−delta)
        constant   = -c_n * log2_delta          # −c · log₂(−delta)

    # ----------------------------------------------------------------------
    # 3.  Finally add  ⟨h(x)⟩ ≥ constant
    # ----------------------------------------------------------------------
        self.model.addConstr(expr >= constant,
                         name="log2_anomaly_score_constraint")


    def buildModel(self):
        self.initSolution()
        self._RandomForestCounterfactualMilp__buildTrees()
        self.anomaly_threshold = 0.2    # or 0.1, 0.2, …
        self.__addAnomalyScoreConstraint(threshold=self.anomaly_threshold)

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

    
    def getAnomalyScore(self, *, return_raw=False):
        
        x = np.asarray(self.x_sol).reshape(1, -1)

    # --- average path length ⟨h(x)⟩ ----------------------------------------
        depths = []
        for t in self.completeForest.isolationForestEstimatorsIndices:
            tm    = self.treeManagers[t]
            tree  = self.completeForest.estimators_[t]

            leaf  = tree.apply(x)[0]          # leaf that x lands in
            depth = tm.node_depth[leaf]       # #edges from root to leaf
            # Expected extra splits to isolate the nₗ points inside that leaf
            depth += _average_path_length([tree.tree_.n_node_samples[leaf]])[0]

            depths.append(depth)

        h_bar = float(np.mean(depths))        # ⟨h(x)⟩
        c_n   = _average_path_length([self.isolationForest.max_samples_])[0]

        # --- scikit-learn scores ----------------------------------------------
        score_samples = -2.0 ** (-h_bar / c_n)        # same as notebook
        if return_raw:
            return score_samples

        return score_samples - float(self.isolationForest.offset_)



    def __checkIfBadPrediction(self, x_sol):
        if self.verbose:
            score = self.getAnomalyScore()
            print("Anomaly score (s(x)):", score)
            #print("log2(score):", np.log2(score))
            #print("Target log2 threshold:", self.anomaly_threshold_log2)
            if score >= 0:
                print("Counterfactual is plausible under the threshold.")
            else:
                print("Warning: counterfactual is too anomalous (violates constraint).")
