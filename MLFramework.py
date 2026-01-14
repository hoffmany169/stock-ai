

from abc import abstractmethod, ABCMeta


class MachineLearningFramework(metaclass=ABCMeta):
    @abstractmethod
    def load_train_data(self):
        """
        Docstring for load_train_data
        load the training data from source
        :param self: Description
        """
        pass

    @abstractmethod
    def preprocess_data(self):
        """
        Docstring for preprocess_data
        prepare the data for training and testing
        :param self: Description
        """
        pass

    @abstractmethod
    def create_scaled_data(self, df):
        '''
        function create_scaled_data
        param[data]: raw data to be scaled
        '''
        pass

    @abstractmethod
    def create_train_test_data(self, percent_split=0.2):
        '''
        function create_train_test_data
        param[test_size]: ratio of dividing train- and test-data
        '''
        pass

    @abstractmethod
    def build_model(self, input_shape: tuple):
        """Docstring for build_model
        build the machine learning model
        :param self: Description
        :param input_shape: Description
        """
        pass


    @abstractmethod
    def train_model(self):
        """
        Docstring for train_model
        train the model with training data
        :param self: Description
        """
        pass

    @abstractmethod
    def evaluate_model(self, output=False):
        """
        Docstring for evaluate_model
        evaluate the model performance with test data
        :param self: Description
        :param data: Description
        """
        pass

    @abstractmethod
    def predict(self, data=None):
        """
        Docstring for predict
        predict with trained model on new data
        :param self: Description
        :param data: Description
        """
        pass

    def process_train_data(self, data, output=True):
        # 加载并预处理数据
        self.load_train_data()
        self.preprocess_data()
        
        # 训练模型
        self.train_model()
        self.evaluate_model(output=output)
