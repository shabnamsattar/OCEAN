import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import IsolationForest
from pathlib import Path
# Import OCEAN functions
from src.DatasetReader import DatasetReader
from src.RfClassifierCounterFactual import RfClassifierCounterFactualMilp
from src.CounterFactualParameters import TreeConstraintsType
from src.CounterFactualParameters import BinaryDecisionVariables


def check_counterfactuals_feasibility(clf, ilf, reader,
                                      indices, desiredOutcome):
    allSolved = True
    count = 1
    for index in indices:
        print("Start checking", count, "out of", len(indices))
        count += 1
        x0 = [reader.data.loc[index, reader.data.columns != 'Class']]
        randomForestMilp = RfClassifierCounterFactualMilp(
            clf, x0, desiredOutcome,
            isolationForest=ilf,
            constraintsType=TreeConstraintsType.LinearCombinationOfPlanes,
            objectiveNorm=1, mutuallyExclusivePlanesCutsActivated=True,
            strictCounterFactual=True, verbose=False,
            binaryDecisionVariables=BinaryDecisionVariables.PathFlow_y,
            featuresActionnability=reader.featuresActionnability,
            featuresType=reader.featuresType,
            featuresPossibleValues=reader.featuresPossibleValues)
        randomForestMilp.buildModel()
        if not randomForestMilp.solveModel():
            print("Warning, index", index, "could not be solved")
            allSolved = False
    if allSolved:
        print("All models could be solved")
    else:
        print("Not all models could be solved")


def build_counterfactual_file(datasetFile, desiredOutcome, nbCounterFactuals,
                              use_isolation_forest=False, anomaly_threshold=0.4):
    print("Start treating", datasetFile)
    reader = DatasetReader(datasetFile)

    clf = RandomForestClassifier(max_leaf_nodes=50, random_state=1, n_estimators=100)
    clf.fit(reader.X_train, reader.y_train)

    ilf = IsolationForest(random_state=1, max_samples=100, n_estimators=100)
    ilf.fit(reader.X_train)

    data = pd.DataFrame(reader.X_test)
    predictions = clf.predict(data)
    data['clf_result'] = predictions
    data['Class'] = reader.y_test

    dataWithoutDesiredResults = data.loc[(data['Class'] != desiredOutcome) & (data['clf_result'] != desiredOutcome)]
    data.drop(['clf_result'], axis=1, inplace=True)
    if len(dataWithoutDesiredResults) > nbCounterFactuals:
        dataWithoutDesiredResults = dataWithoutDesiredResults.sample(n=nbCounterFactuals)

    for index in dataWithoutDesiredResults.index:
        x0 = [reader.data.loc[index, reader.data.columns != 'Class']]
        if use_isolation_forest:
            cf = IfClassifierCounterFactualMilp(
                classifier=ilf,
                sample=x0,
                anomaly_threshold_log2=np.log2(anomaly_threshold),
                featuresType=reader.featuresType,
                featuresPossibleValues=reader.featuresPossibleValues,
                featuresActionnability=reader.featuresActionnability,
                oneHotEncoding=reader.oneHotEncoding,
                objectiveNorm=1,
                verbose=True
            )
        else:
            cf = RfClassifierCounterFactualMilp(
                classifier=clf,
                sample=x0,
                outputDesired=desiredOutcome,
                isolationForest=ilf,
                constraintsType=TreeConstraintsType.LinearCombinationOfPlanes,
                objectiveNorm=1,
                mutuallyExclusivePlanesCutsActivated=True,
                strictCounterFactual=True,
                verbose=True,
                binaryDecisionVariables=BinaryDecisionVariables.PathFlow_y,
                featuresActionnability=reader.featuresActionnability,
                featuresType=reader.featuresType,
                featuresPossibleValues=reader.featuresPossibleValues
            )

        cf.buildModel()
        solved = cf.solveModel()
        if solved:
            print("Counterfactual for index", index, ":", cf.x_sol)
            if use_isolation_forest:
                print("Anomaly score:", cf.getAnomalyScore())
        else:
            print("Could not solve for index", index)


if __name__ == '__main__':
    build_counterfactual_file("datasets/test.csv", desiredOutcome=1, nbCounterFactuals=5,
                              use_isolation_forest=True, anomaly_threshold=0.4)
def get_paths_to_counterfactuals_directory(datasetFile):
    """
    Read paths to files and create folder:
    path can be either a string or a Path object.
    """
    if isinstance(datasetFile, Path):
        datasetName = datasetFile.name
        pathToCounterfactual = datasetFile.parent / 'counterfactuals'
        if not os.path.exists(pathToCounterfactual):
            os.mkdir(pathToCounterfactual)
        outputFile = pathToCounterfactual / datasetName
        oneHotDatasetName = "OneHot_" + datasetName
        oneHotOutputFile = pathToCounterfactual / oneHotDatasetName
    else:
        words = datasetFile.split('/')
        path = ""
        for w in words[:-1]:
            path += w + "/"
        path += "counterfactuals/"
        if not os.path.exists(path):
            os.mkdir(path)
        outputFile = path + words[-1]
        oneHotOutputFile = path + "OneHot_" + words[-1]
    return outputFile, oneHotOutputFile
