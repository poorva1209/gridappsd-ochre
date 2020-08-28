import copy
import numpy as np
import pandas as pd

ids = pd.read_csv('servicepoint_ids.csv')['ServicePoint_Id'].values

result = np.array([])
for i, id in enumerate(ids):
	try:
		values = pd.read_csv('~/Desktop/uncontrollable_load/Loadshapes/ls__{}.csv'.format(id), header=None)
		if result.shape[0] == 0:
			result = copy.deepcopy(values)
		else:
			result += copy.deepcopy(values)
	except:
		pass
df =pd.DataFrame(data=result)
df.to_csv('~/Desktop/uncontrollable_load/uncontrollable_load.csv')                          