import tensorflow as tf
import numpy as np
import pickle
from os import mkdir
from scipy.spatial.distance import euclidean
from sklearn.cluster import KMeans


class RBFNet:

    def __init__ ( self , *args , **kwds ):
        """
        :param args:  contain only file path for loading stored model like './model.p'
        :param kwds:
            'k': (single integer) number of RBF centers

            'y': (single float number) lambda variable for RLS phase

            'CR': (single char) determine model is for classification or regression by 'c' or 'r' character
        """

        if len ( args ) == 1:
            file_path = args[ 0 ]
            tf.reset_default_graph ()
            model_dict = pickle.load ( open ( file_path + "/model.p" , "rb" ) )
            self._CR = model_dict[ 'CR' ]
            self._k = model_dict[ 'k' ]
            self._y = model_dict[ 'y' ]
            self._d_max = model_dict[ 'd_max' ]
            self._centers = model_dict[ 'centers' ]
            self._x_train_shape = model_dict[ 'x_train_shape' ]
            self.__create_graph ( self._x_train_shape , self._d_max , self._centers )
            saver = tf.train.Saver ()
            self._sess = tf.Session ()
            saver.restore ( self._sess , file_path + '/tmp/model.ckpt' )

        else:
            self._k = kwds[ 'k' ]
            self._y = kwds[ 'y' ]
            self._CR = kwds[ 'CR' ]
            self._sess = None
            self._d_max = None
            self._centers = None
            self._x_train_shape = None

    def __rbf ( self , x , mu , sigma ):
        with tf.name_scope ( 'RBF_calculator' ):
            sub = x - mu
            norm = tf.reshape ( tf.norm ( sub , 2 , axis=1 ) , [ self._k , 1 ] )
            sigma_2 = tf.pow ( sigma , 2 )
            multi_sigma_2_norm = (-1 / (2 * sigma_2)) * norm
            exp = tf.exp ( multi_sigma_2_norm )
            return exp

    def __create_graph ( self , x_train_shape , d_max , centers ):
        tf.reset_default_graph ()
        x = tf.placeholder ( dtype=tf.float64 , shape=[ None , x_train_shape ] , name='input' )
        y = tf.placeholder ( dtype=tf.float64 , shape=[ None , 1 ] , name='output' )
        sigma = tf.Variable ( initial_value=np.ones ( shape=[ self._k , 1 ] ) * (d_max / (2 * self._k) ** 0.5) )
        mu = tf.Variable ( centers )
        w = tf.Variable ( initial_value=np.zeros ( shape=[ self._k , 1 ] ) )
        p = tf.Variable ( initial_value=np.identity ( self._k ) * (self._y ** -1) )
        with tf.name_scope ( "RLS" ):
            phi = self.__rbf ( x , mu , sigma )
            p_new = p - (tf.matmul ( tf.matmul ( tf.matmul ( p , phi ) , phi , transpose_b=True ) , p )) / (
                    1 + tf.matmul ( phi , tf.matmul ( p , phi ) , transpose_a=True ))
            g = tf.matmul ( p_new , phi )
            alpha = y - tf.matmul ( w , phi , transpose_a=True )
            new_w = w + tf.matmul ( g , alpha )
            tf.assign ( w , new_w , name='assign_w' )
            tf.assign ( p , p_new , name='assign_p' )
        with tf.name_scope ( 'predict' ):
            tf.reduce_sum ( tf.matmul ( phi , w , transpose_a=True ) , name='output' )

    def train ( self , x_train , y_train ):
        """
        :param x_train:
        :param y_train:
        :return: return list of evaluate metric in each iteration of RLS if evaluate== True, else return None
        """
        print ( '(phase 1): start finding clusters centers... ' )
        k_means = KMeans ( n_clusters=self._k , random_state=1010 ).fit ( x_train )
        self._centers = k_means.cluster_centers_
        # FIND D(max) for sigma
        self._d_max = -1
        for i in range ( len ( self._centers ) ):
            for j in range ( len ( self._centers ) ):
                temp = euclidean ( self._centers[ i ] , self._centers[ j ] )
                if self._d_max < temp:
                    self._d_max = temp
        self._x_train_shape = x_train.shape[ 1 ]
        print ( '(phase 2): creating model structure... ' )
        self.__create_graph ( self._x_train_shape , self._d_max , self._centers )
        print ( 'start training...' )
        init = tf.global_variables_initializer ()
        self._sess = tf.Session ()
        self._sess.run ( init )
        for i in range ( len ( x_train ) ):
            _ , _ = self._sess.run ( [ 'RLS/assign_w:0' , 'RLS/assign_p:0' ] ,
                                     feed_dict={'input:0': [ x_train[ i ] ] , 'output:0': [ y_train[ i ] ]} )
        print ( 'MODEL trained successfully!' )

    def evaluate ( self , x_test , y_test ):
        if self._CR == 'c':
            tp = 0
            fp = 0
            for j in range ( len ( x_test ) ):
                label = self._sess.run ( 'predict/output:0' , feed_dict={'input:0': [ x_test[ j ] ]} )
                if label >= 0.5:
                    label = 1
                else:
                    label = 0
                if label == y_test[ j ][0]:
                    tp += 1
                else:
                    fp += 1
            acc = tp / (tp + fp)  # accuracy
            return {'Accuracy': acc}
        else:
            mse = 0  # mean absolute error
            for j in range ( len ( x_test ) ):
                predicted_value = self._sess.run ( 'predict/output:0' , feed_dict={'input:0': [ x_test[ j ] ]} )
                mse += (((predicted_value - y_test[ j ]) ** 2) / len ( x_test ))
            return {'MSE': mse}

    def predict ( self , x_test ):
        return self._sess.run ( 'predict/output:0' , feed_dict={'input:0': [ x_test ]} )

    def load ( self , file_path='' ):
        tf.reset_default_graph ()
        model_dict = pickle.load ( open ( file_path + "/model.p" , "rb" ) )
        self._CR = model_dict[ 'CR' ]
        self._k = model_dict[ 'k' ]
        self._y = model_dict[ 'y' ]
        self._d_max = model_dict[ 'd_max' ]
        self._centers = model_dict[ 'centers' ]
        self._x_train_shape = model_dict[ 'x_train_shape' ]
        self.__create_graph ( self._x_train_shape , self._d_max , self._centers )
        saver = tf.train.Saver ()
        self._sess = tf.Session ()
        saver.restore ( self._sess , file_path + '/tmp/model.ckpt' )

    def save ( self , file_path='' ):
        try:
            saver = tf.train.Saver ()
            saver.save ( self._sess , file_path + '/tmp/model.ckpt' )
            model_dict = {
                'k': self._k ,
                'y': self._y ,
                'CR': self._CR ,
                'centers': self._centers ,
                'd_max': self._d_max ,
                'x_train_shape': self._x_train_shape ,

            }
            pickle.dump ( model_dict , open ( file_path + "/model.p" , "wb" ) )
        except:
            mkdir ( file_path )
            saver = tf.train.Saver ()
            saver.save ( self._sess , file_path + '/tmp/model.ckpt' )
            model_dict = {
                'k': self._k ,
                'y': self._y ,
                'CR': self._CR ,
                'centers': self._centers ,
                'd_max': self._d_max ,
                'x_train_shape': self._x_train_shape ,

            }
            pickle.dump ( model_dict , open ( file_path + "/model.p" , "wb" ) )


class EBFNet:

    def __init__ ( self , *args , **kwds ):
        """

        :param args:  contain only file path for loading stored model like './model.p'
        :param kwds:
            'k': (single integer) number of RBF centers

            'y': (single float number) lambda variable for RLS phase

            'CR': (single char) determine model is for classification or regression by 'c' or 'r' character
        """

        if len ( args ) == 1:
            file_path = args[ 0 ]
            tf.reset_default_graph ()
            model_dict = pickle.load ( open ( file_path + "/model.p" , "rb" ) )
            self._CR = model_dict[ 'CR' ]
            self._k = model_dict[ 'k' ]
            self._y = model_dict[ 'y' ]
            self._d_max = model_dict[ 'd_max' ]
            self._centers = model_dict[ 'centers' ]
            self._inv_cov_matrix = model_dict[ '_inv_cov_matrix' ]
            self._x_train_shape = model_dict[ 'x_train_shape' ]
            self.__create_graph ( self._x_train_shape , self._inv_cov_matrix , self._d_max , self._centers )
            saver = tf.train.Saver ()
            self._sess = tf.Session ()
            saver.restore ( self._sess , file_path + '/tmp/model.ckpt' )

        else:
            self._k = kwds[ 'k' ]
            self._y = kwds[ 'y' ]
            self._CR = kwds[ 'CR' ]
            self._sess = None
            self._d_max = None
            self._centers = None
            self._x_train_shape = None

    def __ebf ( self , x , mu , sigma , inv_cov_matrices , x_train_shape ):
        with tf.name_scope ( 'EBF_calculator' ):
            sub = x - mu
            sigma_2 = tf.pow ( sigma , 2 )
            r = [ ]
            for i in range ( self._k ):
                r.append ( tf.matmul (
                    tf.matmul ( tf.reshape ( sub[ i ] , [ x_train_shape , 1 ] ) , inv_cov_matrices[ i ] ,
                                transpose_a=True ) ,
                    tf.reshape ( sub[ i ] , [ x_train_shape , 1 ] ) ) )
            r = tf.reshape ( r , [ self._k , 1 ] )
            multi_sigma_2_rest = (-1 / (2 * sigma_2)) * r
            exp = tf.exp ( multi_sigma_2_rest )
            return exp

    def __create_graph ( self , x_train_shape , np_inv_cov_matrices , d_max , centers ):
        tf.reset_default_graph ()
        # place holders
        x = tf.placeholder ( dtype=tf.float64 , shape=[ None , x_train_shape ] , name='input' )
        y = tf.placeholder ( dtype=tf.float64 , shape=[ None , 1 ] , name='output' )
        # variables
        inv_cov_matrices = tf.Variable ( initial_value=np_inv_cov_matrices )
        sigma = tf.Variable ( initial_value=np.ones ( shape=[ self._k , 1 ] ) * (d_max / (2 * self._k) ** 0.5) )
        mu = tf.Variable ( centers )
        w = tf.Variable ( initial_value=np.zeros ( shape=[ self._k , 1 ] ) )
        p = tf.Variable ( initial_value=np.identity ( self._k ) * (self._y ** -1) )
        with tf.name_scope ( "RLS" ):
            phi = self.__ebf ( x , mu , sigma , inv_cov_matrices , x_train_shape )
            p_new = p - (tf.matmul ( tf.matmul ( tf.matmul ( p , phi ) , phi , transpose_b=True ) , p )) / (
                    1 + tf.matmul ( phi , tf.matmul ( p , phi ) , transpose_a=True ))
            g = tf.matmul ( p_new , phi )
            alpha = y - tf.matmul ( w , phi , transpose_a=True )
            new_w = w + tf.matmul ( g , alpha )
            tf.assign ( w , new_w , name='assign_w' )
            tf.assign ( p , p_new , name='assign_p' )
        with tf.name_scope ( "predict" ):
            tf.reduce_sum ( tf.matmul ( phi , w , transpose_a=True ) , name='output' )

    def train ( self , x_train , y_train ):

        print ( '(phase 1): start finding clusters centers... ' )
        k_means = KMeans ( n_clusters=self._k , random_state=1010 ).fit ( x_train )
        self._centers = k_means.cluster_centers_
        labels_of_samples = k_means.labels_
        centers_samples = [ ]
        for i in range ( self._k ):
            centers_samples.append ( [ ] )
        for i in range ( len ( x_train ) ):
            centers_samples[ labels_of_samples[ i ] ].append ( x_train[ i ] )
        list_inv_cov_matrices = [ ]
        for i in range ( self._k ):
            list_inv_cov_matrices.append (
                np.linalg.pinv ( np.cov ( np.array ( centers_samples[ i ] ).T , bias=True ) ) )
        self._inv_cov_matrix = np.array ( list_inv_cov_matrices )
        # FIND D(max) for sigma
        self._d_max = -1
        for i in range ( len ( self._centers ) ):
            for j in range ( len ( self._centers ) ):
                temp = euclidean ( self._centers[ i ] , self._centers[ j ] )
                if self._d_max < temp:
                    self._d_max = temp
        print ( '(phase 2): creating model structure... ' )
        self._x_train_shape = x_train.shape[ 1 ]
        self.__create_graph ( self._x_train_shape , self._inv_cov_matrix , self._d_max , self._centers )
        print ( 'start training...' )
        init = tf.global_variables_initializer ()
        self._sess = tf.Session ()
        self._sess.run ( init )
        for i in range ( len ( x_train ) ):
            _ , _ = self._sess.run ( [ 'RLS/assign_w:0' , 'RLS/assign_p:0' ] ,
                                     feed_dict={'input:0': [ x_train[ i ] ] , 'output:0': [ y_train[ i ] ]} )
        print ( 'MODEL trained successfully!' )

    def evaluate ( self , x_test , y_test ):
        if self._CR == 'c':
            tp = 0
            fp = 0
            for j in range ( len ( x_test ) ):
                label = self._sess.run ( 'predict/output:0' , feed_dict={'input:0': [ x_test[ j ] ]} )
                if label > 0.5:
                    label = 1
                else:
                    label = 0
                if label == y_test[ j ]:
                    tp += 1
                else:
                    fp += 1
            acc = tp / (tp + fp)  # accuracy
            return {'Accuracy': acc}
        else:
            mse = 0  # mean absolute error
            for j in range ( len ( x_test ) ):
                predicted_value = self._sess.run ( 'predict/output:0' , feed_dict={'input:0': [ x_test[ j ] ]} )
                mse += (((predicted_value - y_test[ j ]) ** 2) / len ( x_test ))
            return {'MSE': mse}

    def predict ( self , x_test ):
        return self._sess.run ( 'predict/output:0' , feed_dict={'input:0': [ x_test ]} )

    def load ( self , file_path='' ):
        tf.reset_default_graph ()
        model_dict = pickle.load ( open ( file_path + "/model.p" , "rb" ) )
        self._CR = model_dict[ 'CR' ]
        self._k = model_dict[ 'k' ]
        self._y = model_dict[ 'y' ]
        self._d_max = model_dict[ 'd_max' ]
        self._inv_cov_matrix = model_dict[ '_inv_cov_matrix' ]
        self._centers = model_dict[ 'centers' ]
        self._x_train_shape = model_dict[ 'x_train_shape' ]
        self.__create_graph ( self._x_train_shape , self._inv_cov_matrix , self._d_max , self._centers )
        #can also use meta graph
        saver = tf.train.Saver ()
        self._sess = tf.Session ()
        saver.restore ( self._sess , file_path + '/tmp/model.ckpt' )

    def save ( self , file_path='' ):
        try:
            saver = tf.train.Saver ()
            saver.save ( self._sess , file_path + '/tmp/model.ckpt' )
            model_dict = {
                'k': self._k ,
                'y': self._y ,
                'CR': self._CR ,
                'centers': self._centers ,
                'inv_cov_matrix': self._inv_cov_matrix ,
                'd_max': self._d_max ,
                'x_train_shape': self._x_train_shape ,

            }
            pickle.dump ( model_dict , open ( file_path + "/model.p" , "wb" ) )
        except:
            mkdir ( file_path )
            saver = tf.train.Saver ()
            saver.save ( self._sess , file_path + '/tmp/model.ckpt' )
            model_dict = {
                'k': self._k ,
                'y': self._y ,
                'CR': self._CR ,
                'centers': self._centers ,
                'inv_cov_matrix': self._inv_cov_matrix ,
                'd_max': self._d_max ,
                'x_train_shape': self._x_train_shape ,

            }
            pickle.dump ( model_dict , open ( file_path + "/model.p" , "wb" ) )