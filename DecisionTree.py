import pandas as pd
import math

class DecisionTree(object):

    def __init__(self):
        pass

    def makeTree(self, X, Y):
        """
        Construct decision tree

        Parameters
        ----------
        X : matrix, shape = [n_samples, n_features]

        Y : array-like, shape = [n_samples]

        Returns
        -------
        tree: a tree node stored in dictionary.
              key: a tuple contains (name of the classifier, decision if no subtrees)
              value: a list. the element of the list are lists of length 3.
                  List[0]: value of the classifier.
                  List[1]: decision if the tree has no subtrees.
                  List[2]: subtrees.
        """
        attributes = list(X.columns.values)
        label = list(Y.columns.values)[0]
        data = pd.concat([X, Y], axis=1)

        attribute, defaultDecision, decisions = self.getBestAttribute(data, label, attributes)
        # No information gain. Done classifying.
        if not attribute:
            return 

        attributes.remove(attribute)
        dataGroups = data.groupby(attribute)
        tree, branches = {}, []

        for key in dataGroups.groups.keys():
            branch = [key, decisions[key], None]
            branches.append(branch)
        tree[(attribute, defaultDecision)] = branches

        for group in dataGroups.groups:
            subData = dataGroups.get_group(group)
            X = subData.drop([label, attribute], axis=1)
            Y = pd.DataFrame(subData[label])

            # Find which branch the current subtree belongs to
            for i in xrange(len(tree[(attribute, defaultDecision)])):
                if tree[(attribute, defaultDecision)][i][0] == group:
                    tree[(attribute, defaultDecision)][i][2] = self.makeTree(X, Y)
                    break
        return tree

    def computeEntropy(self, data, label):
        """
        Compute entropy of a dataset using the classifier label.

        Parameters
        ----------
        data : matrix, shape = [n_samples, n_features]

        label : class labels in classification.

        Returns
        -------
        entropy : entropy value        
        """
        dataGroups = data.groupby(label)
        total = float(len(data))
        entropy = 0

        for group in dataGroups.groups:
            freq = len(dataGroups.get_group(group))
            if freq == 0:
                continue
            entropy += -freq / total * math.log(freq / total, 2)
        return entropy

    def getBestAttribute(self, data, label, attributes):
        """
        Select best attribute from attributes that maximize entropy gain

        Parameters
        ----------
        data : matrix, shape = [n_samples, n_features]

        label : class labels in classification.

        attributes: array-like, shape = [n_features] or None.

        Returns
        -------
        bestAttr: attribute name that maximize entropy gain or None
        defaultDecision: The decision chosen if bestAttr is None.
        decisions: The most frequent label value in current dataset group by bestAttr values.
        """
        entropy = self.computeEntropy(data, label)
        # Intialize the default values. The default decision is the most frequent label value
        defaultDecision = data[label].value_counts().keys()[0]
        maxGain, bestAttr, decisions = 0, None, defaultDecision

        # Compute entropy gain for each attribute 
        for attribute in attributes:
            tempdecisions = {}
            dataGroups = data.groupby(attribute)
            total = float(len(data))
            subentropy = 0

            for group in dataGroups.groups:
                subData = dataGroups.get_group(group)
                freq = len(dataGroups.get_group(group))
                subentropy += freq / total * self.computeEntropy(subData, label)
                decision = subData[label].value_counts().keys()[0]
                tempdecisions[group] = decision

            gain = entropy - subentropy
            if gain > maxGain:
                bestAttr = attribute
                maxGain = gain
                decisions = tempdecisions

        return bestAttr, defaultDecision, decisions

    def fit(self, tree, data):
        """
        Pred the data with the decision tree.

        Parameters
        ----------
        data : matrix, shape = [n_samples, n_features]

        tree: decsion tree

        Returns
        -------
        decisions: list. Prediction results. shape = [n_samples]
        """
        nrow, _ = data.shape
        res = []

        for i in xrange(nrow):
            d = data[i: i + 1]
            pred = self.__fit(tree, d)
            res.append(pred)
        return pd.Series(res)

    def __fit(self, tree, data):
        """
        Pred the data with the decision tree.

        Parameters
        ----------
        data : array-like, shape = [n_samples]

        tree: decsion tree

        Returns
        -------
        decisions: type of the attribute of interest. Prediction results. shape = [n_samples]
        """
        key, decision = tree.keys()[0]
        values = tree.values()[0]
        prop = list(data[key])[0]
        
        for val in values:
            if val[0] == prop:
                subtree = val[2]
                if not subtree:
                    return val[1]
                else:
                    return self.__fit(subtree, data)
        return decision