import pandas as pd
import math

class DecisionTree(object):

	def __init__(self):
		pass

	def makeTree(self, X, Y):
		"""Construct decision tree

        Parameters
        ----------
        X : matrix, shape = [n_samples, n_features]

        Y : array-like, shape = [n_samples]

        Returns
        -------
        tree: a tree node stored in dictionary.
        """
		attributes = list(X.columns.values)
		label = list(Y.columns.values)[0]
		data = pd.concat([X, Y], axis=1)

		if len(attributes) == 0: 
			return data[label].value_counts().keys()[0]

		attribute, decision = self.getBestAttribute(data, label, attributes)
		if not attribute: 
			return decision

		attributes.remove(attribute)
		dataGroups = data.groupby(attribute)
		tree, leaves = {}, {}
		for key in dataGroups.groups.keys():
			leaves[key] = {}
		tree[attribute] = leaves

		for group in dataGroups.groups:
			subData = dataGroups.get_group(group)
			X = subData.drop([label, attribute], axis=1)
			Y = pd.DataFrame(subData[label])
			tree[attribute][group] = self.makeTree(X, Y)
		return tree

	def computeEntropy(self, data, label):
		"""Compute entropy of a dataset using the classifier label.

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
		"""Select best attribute from attributes that maximize entropy gain

        Parameters
        ----------
        data : matrix, shape = [n_samples, n_features]

        label : class labels in classification.

        attributes: array-like, shape = [n_features] or None.

        Returns
        -------
        bestAttr: attribute name that maximize entropy gain or None
        decision: The most frequent label value in current dataset.
        """
		entropy = self.computeEntropy(data, label)
		maxGain, bestAttr, decision = 0, None, None

		# Compute entropy gain for each attribute
		for attribute in attributes:
			dataGroups = data.groupby(attribute)
			total = float(len(data))
			subentropy = 0

			for group in dataGroups.groups:
				subData = dataGroups.get_group(group)
				freq = len(dataGroups.get_group(group))
				subentropy += freq / total * self.computeEntropy(subData, label)

			gain = entropy - subentropy
			if gain > maxGain:
				bestAttr = attribute
				maxGain = gain

		decision = data[label].value_counts().keys()[0]
		return bestAttr, decision