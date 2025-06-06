from Common.AutoNumber import AutoIndex
from datetime import date, timedelta
from sklearn.preprocessing import StandardScaler
import yfinance as yf
import numpy as np

class FEATURE(AutoIndex):
    Close = ()
    MA_5 = ()
    MA_20 = ()
    RSI = ()
    MACD = ()
    PE = ()
    PB = ()
    Volume = ()

class Select_Stock:
    def __init__(self, tickers:list, start_date, end_date=None, win_size=60):
        self._tickers = tickers
        self._start_date =  start_date
        self._yesterday = date.today() - timedelta(days=1)
        self._end_date = f"{self._yesterday.year}-{self._yesterday.month}-{self._yesterday}" if end_date is None else end_date
        self._window_size = win_size
        self.all_data = None # data downloaded from yfinance
        self._full_dataset = []
        self._threshold = 0.05  # 5%收益率阈值
        self._prediction_threshold = 0.7

    @property
    def threshold(self):
        return self._threshold
    
    @property
    def prediction_threshold(self):
        return self._prediction_threshold

    def _download_data_from_yfinance(self):
        self.all_data = yf.download(self._tickers, start=self._start_date, end=self._end_date, group_by='ticker')

    def create_data_sequence(self, ticker, df):
        '''
        function _create_data_sequence
        param[ticker]: ticker of stock
        param[df]: dataframe of stock data
        '''
        print("Implementation _create_data_sequence")

    def create_train_test_data(self, test_size=0.2):
        '''
        function create_train_test_data
        param[test_size]: ratio of dividing train- and test-data
        '''
        print("Implementation create_train_test_data")


    # compute rsi
    def _compute_rsi(self, data_sequence, period=14):
        delta = data_sequence.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(period).mean()
        avg_loss = loss.rolling(period).mean()
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    # compute macd
    def _compute_macd(self, data_sequence, slow=26, fast=12, signal=9):
        ema_fast = data_sequence.ewm(span=fast).mean()
        ema_slow = data_sequence.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal).mean()
        return macd - signal_line

    # get trailing PE (or forward PE) 市盈率
    # 示例用法
    # print(get_pe_ratio("AAPL"))  # 输出苹果公司市盈率
    def get_pe_ratio(self, ticker):
        try:
            stock = yf.Ticker(ticker)
            pe = stock.info.get('trailingPE')  # 或 'forwardPE'
            return pe if pe is not None else np.nan
        except:
            return np.nan

    # get price to book ratio 市净率
    # 示例用法
    # print("AAPL P/B Ratio:", get_pb_ratio("AAPL"))
    def get_pb_ratio(self, ticker):
        """
        使用 Yahoo Finance API 获取市净率
        """
        try:
            stock = yf.Ticker(ticker)
            # 获取市净率
            pb_ratio = stock.info.get('priceToBook')
            
            # 如果市净率不存在，尝试计算
            if pb_ratio is None:
                # 获取当前股价
                current_price = stock.history(period='1d')['Close'].iloc[-1]
                
                # 获取每股账面价值
                book_value_per_share = stock.info.get('bookValue')
                
                if book_value_per_share and book_value_per_share > 0:
                    pb_ratio = current_price / book_value_per_share
                else:
                    pb_ratio = np.nan
            return pb_ratio
        except Exception as e:
            print(f"Error getting P/B ratio for {ticker}: {str(e)}")
            return np.nan

    def create_scaled_data(self, X):
        features = [f.name for f in FEATURE]
        scaler = StandardScaler()
        return scaler.fit_transform(X)
    
