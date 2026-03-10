import pandas as pd
import numpy as np

data = pd.read_csv('text.csv')

concept = np.array(data.iloc[:, :-1])
target = np.array(data.iloc[:, -1])

def train(con, tar):
    specific_h = None
    for i, val in enumerate(tar):
        if val.lower() == 'yes':
            specific_h = con[i].copy()
            break

    if specific_h is None:
        return "No positive examples found."

    for i, val in enumerate(con):
        if tar[i].lower() == 'yes':
            for x in range(len(specific_h)):
                if val[x] != specific_h[x]:
                    specific_h[x] = '?'

    return specific_h

print("Final Specific Hypothesis:")
print(train(concept, target))