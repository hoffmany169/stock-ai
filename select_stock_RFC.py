# 导入必要库
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from select_stock import Select_Stock, FEATURE

class Select_Stock_RFC(Select_Stock):
    def __init__(self, tickers, start_date, end_date=None, win_size=5):
        super().__init__(tickers, start_date, end_date=end_date, win_size=win_size)
        self._features = []
        self._stock_data = None

    # 1. 数据获取函数
    def get_tickers_data(self):
        self._download_data_from_yfinance()
        # 创建特征数据集
        for ticker in self._tickers:
            if ticker not in self.all_data:
                print(f"==> Warning: {ticker} data was not downloaded!")
                continue
            # put ticker data
            df = self.all_data[ticker].copy()
            df['Ticker'] = ticker
            self._create_feature_data(ticker, df)
    

    def _create_feature_data(self, ticker, df):
        # put technical data
        df[FEATURE.MA_5.name] = df[FEATURE.Close.name].rolling(5).mean()      # 5 days mean
        df[FEATURE.MA_20.name] = df[FEATURE.Close.name].rolling(20).mean()    # 20 days mean
        df[FEATURE.RSI.name] = self._compute_rsi(df[FEATURE.Close.name])      # rsi
        df[FEATURE.MACD.name] = self._compute_macd(df[FEATURE.Close.name])

        # 添加基本面数据（示例）
        df[FEATURE.PE.name] = self.get_pe_ratio(ticker) 
        df[FEATURE.PB.name] = self.get_pb_ratio(ticker) 
        # scaled_data = self._create_scaled_data(df)
        # self._create_data_sequence(ticker, df, scaled_data)
        self._features.append(df)
        self._stock_data = pd.concat(self._download_data_from_yfinancefeatures)
        self.create_data_sequence(ticker, self._stock_data)

    def create_data_sequence(self, ticker, df):
        data = self._stock_data.dropna()
        ## calculate future return
        data['Future_Return'] = data['Close'].pct_change(self._window_size).shift(-self._window_size)
        ## create binary clafication label
        data['Label'] = np.where(data['Future_Return'] > self._threshold, 1, 0)

        # choose feature columns
        feature_columns = [f.name for f in FEATURE if f.value > 0]
        X = data[feature_columns]
        X_scaled_data = self.create_scaled_data(X)
        y = ['Label']
        return X, y

    def create_train_test_data(self, test_size=0.2):

# 3. 特征工程与标签创建
def prepare_features(data, future_days=5):
    """
    创建特征和标签
    label: 未来5日收益率是否超过阈值（分类问题）
    """
    data = data.dropna()
    
    # 计算未来收益率
    data['Future_Return'] = data['Close'].pct_change(future_days).shift(-future_days)
    
    # 创建二分类标签
    threshold = 0.05  # 5%收益率阈值
    data['Label'] = np.where(data['Future_Return'] > threshold, 1, 0)
    
    # 选择特征列
    feature_columns = ['MA_5', 'MA_20', 'RSI', 'MACD', 'PE', 'PB', 'Volume']
    X = data[feature_columns]
    y = data['Label']
    
    return X, y

# 4. 模型训练
def train_model(X, y):
    # 数据标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 划分训练测试集（时间序列需使用时间分割）
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, shuffle=False)
    
    # 初始化模型
    model = RandomForestClassifier(n_estimators=100,
                                  max_depth=5,
                                  random_state=42)
    
    # 训练模型
    model.fit(X_train, y_train)
    
    # 评估模型
    print("模型评估结果：")
    print(classification_report(y_test, model.predict(X_test)))
    
    return model, scaler

# 5. 选股策略
def stock_selection_strategy(model, scaler, current_data):
    """
    使用训练好的模型进行选股
    """
    # 预处理当前数据
    X_current = current_data[feature_columns]
    X_scaled = scaler.transform(X_current)
    
    # 预测概率
    proba = model.predict_proba(X_scaled)[:, 1]
    
    # 选择概率最高的前10%股票
    selected_stocks = current_data[proba > np.percentile(proba, 90)]
    
    return selected_stocks['Ticker'].unique()

# 主程序
if __name__ == "__main__":
    # 配置参数
    tickers = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'TSLA']  # 示例股票列表
    start_date = '2018-01-01'
    end_date = '2023-01-01'
    ss_rfc = Select_Stock_RFC(tickers, start_date, end_date=end_date, win_size=5)
    # 获取数据
    ss_rfc.get_tickers_data()
    
    # 准备特征
    X, y = ss_rfc.create_data_sequence()
    
    # 训练模型
    model, scaler = train_model(X, y)
    
    # 获取最新数据
    current_data = get_stock_data(tickers, 
                                start_date=pd.to_datetime(end_date) - pd.DateOffset(30),
                                end_date=pd.to_datetime(end_date))
    
    # 执行选股策略
    selected_stocks = stock_selection_strategy(model, scaler, current_data)
    print("推荐股票列表：", selected_stocks)