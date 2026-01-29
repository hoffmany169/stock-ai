from Common.AutoNumber import AutoIndex

class FEATURE(AutoIndex):
    Open_Close = ()
    High_Low = ()
    Close_Low = ()
    Close_High = ()
    Avg_Price = ()
    Volume_Change = ()
    MA_5 = ()
    MA_20 = ()
    RSI = ()
    MACD = ()
    Volume_MA_5 = ()
    Price_Volume_Ratio = ()
    PE = ()
    PB = ()
    Volume = ()

class TICKER(AutoIndex):
    ID = ()
    DATA = ()
    FEATURES = ()
    MODEL = ()
    SCALER = ()
    TRAIN_DATA = ()
    TRAIN_TEST_DATA = ()
    HISTORY = ()
    PERFORMANCE = ()
    SELECTED = ()

class StockFeature:
    def __init__(self):
        self._FEATURE_STATE_LIST = self._create_feature_list()

    def _create_feature_list(self)->dict:
        return dict(zip([f for f in FEATURE], [True]*len(FEATURE )))

    def get_features(self)->list:
        features = []
        for f, v in self._FEATURE_STATE_LIST.items():
            if v: features.append(f)
        return features

    def get_feature_count(self)->int:
        return len(self.get_features)

    @staticmethod
    def get_feature_name(feature)->str:
        name = 'No such feature'
        for f in FEATURE:
            if f == feature:
                if f == FEATURE.Open_Close:
                    name = "Open-Close Difference"
                elif f == FEATURE.High_Low:
                    name = "High-Low Difference"
                elif f == FEATURE.Close_Low:
                    name = "Close-Low Difference"
                elif f == FEATURE.Close_High:
                    name = "Close-High Difference"
                elif f == FEATURE.Avg_Price:
                    name = "Average Price"
                else:
                    name = ' '.join(f.name.split('_'))
        return name

    def is_feature_used(self, feature)->bool:
        return self._FEATURE_STATE_LIST[feature]
    
    def enable_feature(self, feature):
        self._FEATURE_STATE_LIST[feature] = True

    def disable_feature(self, feature):
        self._FEATURE_STATE_LIST[feature] = False

class TICKER_DATA_PARAM(AutoIndex):
    ticker_symbol = ()
    start_date = ()
    end_date = ()
    interval = ()

class LTSM_MODEL_PARAM(AutoIndex):
    timestamp = ()
    lookback = ()
    future_days = ()
    threshold = ()
    features = ()
    feature_count = ()

class MODEL_TRAIN_DATA(AutoIndex):
    ticker_data = ()
    ticker_data_params = ()
    model = ()
    scaler = ()
    parameters = ()
    readme = ()
    train_history = ()
    performance = ()
    model_summary = ()

