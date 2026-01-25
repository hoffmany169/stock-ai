

from abc import abstractmethod, ABCMeta


class MachineLearningFramework(metaclass=ABCMeta):
    def load_historical_data(self, ticker_symbol, start_date, end_date):
        """
        Docstring for load_historical_data
        load the historical data from source
        :param ticker_symbol: ticker symbol to load data for
        :param start_date: start date for the data
        :param end_date: end date for the data
        return: DataFrame
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
        param[df]: raw data to be scaled
        '''
        pass

    @abstractmethod
    def create_train_test_data(self):
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
    def evaluate_model(self):
        """
        Docstring for evaluate_model
        evaluate the model performance with test data
        :param self: Description
        :param data: Description
        """
        pass

    @abstractmethod
    def predict(self, data):
        """
        Docstring for predict
        predict with trained model on new data
        :param self: Description
        :param data: Description
        """
        pass

    def process_train_data(self, evaluation=False):
        """
        ### 加载并预处理数据
        """
        self.preprocess_data()

        # 训练模型
        self.train_model()
        if evaluation:
            self.evaluate_model()