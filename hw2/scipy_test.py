from scipy.optimize import minimize
import numpy as np
from quad import quad as qd
from rosen import rosen as ros
from brach import brachistochrone as br

# x0 = np.array([5.0, 5.0])
x0 = 0.5 * np.zeros(58)
options = {'eps': 1e-5}
res = minimize(br, x0, options=options)
print('Scipys optimized x = ', res.x)
print('Scipys function calls was: ', res.nfev)