import numpy as np
from BFNet import RBFNet , EBFNet
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

x = [ ]
y = [ ]
for i in range ( 1000 ):
    x.append ( [ 2 + np.random.normal ( 0 , 1 ) , 2 + np.random.normal ( 0 , 1 ) ] )
    y.append ( [ 1 ] )
for i in range ( 1000 ):
    x.append ( [ 5 + np.random.normal ( 0 , 1 ) , 3 + np.random.normal ( 0 , 1 ) ] )
    y.append ( [ 0 ] )
x = np.array ( x )
print ( x.shape )
plt.scatter ( x[ : , 0 ] , x[ : , 1 ] )
plt.show ()

m = RBFNet ( k=100 , y=0.1 , CR='c' )
X_train , X_test , y_train , y_test = train_test_split ( x , y , test_size=0.33 , random_state=42 )
m.train ( X_train , y_train )
print ( m.evaluate ( X_test , y_test ) )
class1 = [ ]
class2 = [ ]
for i in range ( len ( x ) ):
    c = m.predict ( x[ i ] )
    if c >= 0.5:
        class1.append ( x[ i ] )
    else:
        class2.append ( x[ i ] )
class1 = np.array ( class1 )
class2 = np.array ( class2 )
plt.scatter ( class1[ : , 0 ] , class1[ : , 1 ] )
plt.scatter ( class2[ : , 0 ] , class2[ : , 1 ] )
plt.show ()
