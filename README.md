```python
import pandas as pd
import math
from IPython.display import Image
from DecisionTree import *
```

To illustrate how the tree works, I use the following example. The example and traing data are from <a href="http://www.cise.ufl.edu/~ddd/cap6635/Fall-97/Short-papers/2.htm" target="_blank">ID3</a>. The training set contains 15 samples.

| Day     | Outlook  | Temperature |Humidity    |Wind		 |Play ball |
| ------- |:--------:| :----------:|:----------:|:----------:|:--------:|
| D1      | Sunny    | Hot         |High		|	Weak     |	No	    |
| D2      | Sunny    | Hot         |High		|	Strong	 |	No	    |
| D3      | Overcast | Hot         |High		|	Weak	 |	Yes	    |
| D4      | Rain     | Mild        |High		|	Weak     |	Yes	    |
| D5      | Rain     | Cool        |Normal		|	Weak	 |	Yes	    |
| D6      | Rain     | Cool        |Noraml		|	Strong	 |	No	    |
| D7      | Overcast | Cool        |Noraml		|	Strong	 |	Yes	    |
| D8      | Sunny    | Mild        |High		|	Weak	 |	No	    |
| D9      | Sunny    | Cool        |Noraml		|	Weak	 |	Yes	    |
| D10     | Rain     | Mild        |Noraml		|	Weak	 |	Yes	    |
| D11     | Sunny    | Mild        |Noraml		|	Strong	 |	Yes	    |
| D12     | Overcast | Mild        |High 		|	Strong	 |	Yes	    |
| D13     | Overcast | Hot         |Noraml		|	Weak 	 |	Yes	    |
| D14     | Rain     | Mild        |High		|	Strong	 |	No	    |

The tree constructed by id3 algorithm looks like

![alt text](http://www.cise.ufl.edu/~ddd/cap6635/Fall-97/Short-papers/Image3.gif)

Read the data and call the makeTree method to build the tree.

```python
train = pd.read_csv("Data/train.csv")
X = train.drop(["Play ball", "Day"], axis=1)
Y = pd.DataFrame(train["Play ball"])
s = DecisionTree()

tree = s.makeTree(X, Y)
print tree
```

The tree looks like
```python
{('Outlook', 'Yes'): [['Overcast', 'Yes', None], 
					  ['Sunny', 'No', {('Humidity', 'No'): [['High', 'No', None], 
					  										['Normal', 'Yes', None]]}], 
					  ['Rain', 'Yes', {('Wind', 'Yes'): [['Strong', 'No', None], 
					  									 ['Weak', 'Yes', None]]}]]}
```

I add 9 samples as the test set

| Day     | Outlook  | Temperature |Humidity    |Wind		 | Pred     |
| ------- |:--------:| :----------:|:----------:|:----------:|:--------:|
| D15     | Hot      | Hot         |Low	    	|	Strong   |	No	    |
| D16     | Hot      | Hot         |High		|	Weak	 |	No	    |
| D17     | Mild     | Hot         |Low		    |	Weak	 |	Yes	    |
| D18     | Cool     | Mild        |Noraml		|	Weak     |	Yes	    |
| D19     | Cool     | Cool        |High		|	Strong	 |	Yes	    |
| D20     | Cool     | Cool        |High		|	Weak	 |	No	    |
| D21     | Hot      | Cool        |Noraml		|	Weak	 |	Yes	    |


We can fit the test set with the tree constructed above

```python
test = pd.read_csv("Data/test.csv")

print s.fit(tree, test)
```

```python
0     No
1     No
2    Yes
3    Yes
4     No
5    Yes
6    Yes
dtype: object
```