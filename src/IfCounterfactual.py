import math
import gurobipy as gp
from gurobipy import GRB
from sklearn.ensemble._iforest import _average_path_length

from src.CounterfactualMilp import CounterfactualMilp
from src.RandomForestCounterfactual import RandomForestCounterfactualMilp
from src.IsolationOnlyForest import IsolationOnlyForest
from src.CounterFactualParameters import TreeConstraintsType, BinaryDecisionVariables, FeatureType

class IfCounterfactualMilp(CounterfactualMilp, RandomForestCounterfactualMilp):
    """
    Find the closest x' to x0 such that an Isolation Forest would treat x' as an inlier.
    
    """
    def __init__(self,
                 isolationForest,              # REQUIRED: trained sklearn IF
                 sample,                       # x0 (normalized & one-hot as used to train IF)
                 objectiveNorm=2,
                 verbose=False,
                 featuresType=False,
                 featuresPossibleValues=False,
                 featuresActionnability=False,
                 oneHotEncoding=False,
                 constraintsType=TreeConstraintsType.LinearCombinationOfPlanes,
                 binaryDecisionVariables=BinaryDecisionVariables.LeftRight_lambda):
        # Base: features, variables, objective holder, model
        CounterfactualMilp.__init__(
            self, sample, objectiveNorm, verbose,
            featuresType, featuresPossibleValues,
            featuresActionnability, oneHotEncoding)
        self.outputDesired = 0 

        # Reuse the ensemble→MILP machinery, but WITHOUT plausibility add-on
        RandomForestCounterfactualMilp.__init__(
            self,
            mutuallyExclusivePlanesCutsActivated=False,
            constraintsType=constraintsType,
            binaryDecisionVariables=binaryDecisionVariables,
            addPlausibilityConstraint=False  # <— IMPORTANT: turn off old plausibility
        )

        self.model.modelName = "IsolationForestCounterfactualMilp"
        self.isolationForest = isolationForest
        self.completeForest = IsolationOnlyForest(isolationForest)
        
        self.randomCostsActivated = False
        
        self.greaterCosts = [1.0] * self.nFeatures
        self.smallerCosts = [1.0] * self.nFeatures


        # Convenience partitions from CounterfactualMilp
        self.continuousFeatures = [f for f in range(self.nFeatures)
                                   if self.featuresType[f] == FeatureType.Numeric]
        self.discreteFeatures = [f for f in range(self.nFeatures)
                                 if self.featuresType[f] in (FeatureType.Discrete,)]
        self.categoricalNonOneHotFeatures = [f for f in range(self.nFeatures)
                                 if self.featuresType[f] == FeatureType.CategoricalNonOneHot]
        self.binaryFeatures = [f for f in range(self.nFeatures)
                               if self.featuresType[f] == FeatureType.Binary]

    # ---------- Key new constraint: log2 anomaly score ----------
    def __addAnomalyScoreConstraint(self, threshold=0.0):
        """
        Enforce decision_function(x') >= threshold.

        Sklearn IF: decision_function(x) = score_samples(x) - offset_
        and score_samples(x) = 2^(- E[h(x)] / c(n))
        We require: 2^(-E[h]/c_n) - offset_ >= threshold
                  => 2^(-E[h]/c_n) >= threshold + offset_ =: delta
        We assume delta < 0 (typical when 'threshold' is 0 and offset_ > 0) so that -delta in (0, 1),
        then:   -E[h]/c_n >= log2(delta)
                E[h] >= - c_n * log2(-delta)      with delta = threshold + offset_  (delta < 0)
        """
        expr = gp.LinExpr(0.0)  # will hold the average depth E[h]
        for t in self.completeForest.isolationForestEstimatorsIndices:
            tm   = self.treeManagers[t]
            tree = self.completeForest.estimators_[t]
            for v in range(tm.n_nodes):
                if tm.is_leaves[v]:
                    depth = tm.node_depth[v] + _average_path_length([tree.tree_.n_node_samples[v]])[0]
                    expr += depth * tm.y_var[v] / self.completeForest.n_estimators

        c_n   = _average_path_length([self.isolationForest.max_samples_])[0]
        delta = threshold + float(self.isolationForest.offset_)
        if delta >= 0:
            raise ValueError("threshold + offset_ must be negative for a valid cut-off")

        log2_delta = math.log2(-delta)       # log2(−delta) < 0
        constant   = -c_n * log2_delta       # positive
        self.model.addConstr(expr >= constant, name="log2_anomaly_score_constraint")

    # ---------- Build & Solve ----------
    def buildModel(self, decision_threshold=0.0):
        # variables & per-feature feasible sets
        self.initSolution()
        # build per-tree path variables & inter-tree consistency
        self.buildForest()
        # feature constraints
        self.addActionnabilityConstraints()
        self.addOneHotEncodingConstraints()
        # anomaly-→inlier constraint
        self.__addAnomalyScoreConstraint(threshold=decision_threshold)
        # objective (closest x' to x0)
        self.initObjective()  # from CounterfactualMilp

    def solveModel(self, time_limit=600, threads=4):
        self.model.setParam(GRB.Param.TimeLimit, time_limit)
        self.model.setParam(GRB.Param.Threads, threads)
        self.model.optimize()
        if self.model.status != GRB.OPTIMAL:
            self.x_sol = self.x0
            return False
        self.x_sol = [[self.x_var_sol[f].getAttr(GRB.Attr.X) for f in range(self.nFeatures)]]
        return True
