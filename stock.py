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
    HISTORY = ()
    PERFORMANCE = ()
    SELECTED = ()

def get_chinese_feature_name(feature):
    feature_names = {
        FEATURE.Open_Close: "开盘收盘差",
        FEATURE.High_Low: "最高最低差",
        FEATURE.Close_Low: "收盘最低差",
        FEATURE.Close_High: "收盘最高差",
        FEATURE.Avg_Price: "平均价格",
        FEATURE.Volume_Change: "成交量变化",
        FEATURE.MA_5: "5日均线",
        FEATURE.MA_20: "20日均线",
        FEATURE.RSI: "相对强弱指数",
        FEATURE.MACD: "平滑异同移动平均线",
        FEATURE.Volume_MA_5: "5日成交量均线",
        FEATURE.Price_Volume_Ratio: "价格成交量比率",
        FEATURE.PE: "市盈率",
        FEATURE.PB: "市净率",
        FEATURE.Volume: "成交量"
    }
    return feature_names.get(feature, "未知特征")