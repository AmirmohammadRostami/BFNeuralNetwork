import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from BFNet import RBFNet


x = np.arange ( 0 , 1080 , 5 )
x = np.reshape ( x , (len ( x ) , 1) )
y = np.sin ( x * np.pi / 180 )
y = np.reshape ( y , (len ( x ) , 1) )
x_train , x_test , y_train , y_test = train_test_split ( x , y , test_size=0.3 , random_state=42 )
plt.scatter ( x_train , y_train , label='train data' )
plt.scatter ( x_test , y_test , label='test_data' )
plt.legend ()
plt.title ( 'normal data' )
plt.show ()
m = RBFNet ( k=85 , y=0.0005 , CR='r' )
m.train ( x_train , y_train )
y_prediction = [ ]
for i in range ( len ( x ) ):
    y_prediction.append ( m.predict ( x[ i ] ) )
print(m.evaluate(x_test,y_test))
plt.scatter ( x_train , y_train , label='train data' )
plt.scatter ( x_test , y_test , label='test_data' )
plt.scatter ( x , y_prediction , label='model prediction' )
plt.legend ()
plt.show ()