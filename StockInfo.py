from Common.AutoNumber import AutoIndex
from Common.Util import CreateChildWindow, CloseChildWindow

USA_STOCK = {
        'AAPL' : 'Aple',
        'GOOGL' : 'Google',
        'MSFT'  : 'Microsoft',
        'AMZN'  : 'Amazon',
        'TSLA'  : 'Tesla', 
        'NVDA'  : 'Nvdia',
        'META'  : 'Meta',
        'NFLX'  : 'Nefix',
        'INTC'  : 'Intc',
        'AMD'   : 'AMD',
        'BABA'  : 'BABA',
        'JD'    : 'JD',
        'PDD'   : 'PDD',
        'BIDU'  : 'BIDU',
        'NTES'  : 'NTES'}
GER_STOCK = {
        "SAP.DE"    : 'SAP SE Software',      # SAP SE - 软件公司（DAX最大成分股）
        "SAP.F"     : 'SAP SE',      # 同一公司，不同交易所后缀
        "SIE.DE"    : 'Siemens AG',      # 西门子
        "ALV.DE"    : 'Allianz SE',      # 安联保险
        "DAI.DE"    : 'Mercedes-Benz Group',      # 戴姆勒（梅赛德斯-奔驰集团）
        "BAS.DE"    : 'BASF SE',      # 巴斯夫
        "BAYN.DE"   : 'Bayer AG',      # 拜耳
        "BMW.DE"    : 'BMW AG',      # 宝马
        "VOW3.DE"   : 'Volkswagen AG (Pref)',      # 大众汽车优先股
        "VOW.DE"    : 'Volkswagen AG',      # 大众汽车普通股
        "CON.DE"    : 'Continent Group',      # 大陆集团
        "ADS.DE"    : 'Adidas AG',      # 阿迪达斯
        "DBK.DE"    : 'Deutsche Bank AG',      # 德意志银行
        "DB1.DE"    : 'Deutsche Bank AG',      # 德意志交易所
        "MRK.DE"    : 'Merk',      # 默克
        "RWE.DE"    : 'RWE Energie',      # RWE能源公司
        "ENR.DE"    : 'Siemens Energie',      # 西门子能源
        "IFX.DE"    : 'Infinex',      # 英飞凌
        "HEI.DE"    : 'Heideburg',      # 海德堡水泥
        "FRE.DE"    : 'Fre',      # 弗雷森纽斯医疗
        "EVK.DE"    : 'Evonik Industries',      # Evonik Industries - 赢创工业
        "ZAL.DE"    : 'Zalando SE',      # Zalando - 德国电商平台
        "HEN3.DE"   : 'Henkel AG (Pref)',      # Henkel - 汉高（优先股）
        "HEN.DE"    : 'Henkel AG',      # 汉高（普通股）
        "EOAN.DE"   : 'E.ON'}      # E.ON - 意昂集团

FUTURES = {"GC=F"   : 'Gold', #(黄金), 
           "CL=F"   : 'Raw Oil'}  #(原油)
INDICES = {'^MDAXI' : 'MDAX 50',       # MDAX指数 - 德国中型股指数（包含50家中型公司）
            '^TECDAX': 'TecDAX Techlogie Index',      # TecDAX指数 - 德国科技股指数
            '^SDAXI': 'SDAX',       # SDAX指数 - 德国小型股指数
            '^GDAXIP': 'DAX 30',      # DAX 30指数 - 法兰克福证券交易所主要指数
            '^GSPC' : 'SPC 500',        # (标普500), 
            '^DJI'  : 'Dow Jones'}          # (道琼斯)
HONGKONG = {'0700.HK' : 'QQ', # (腾讯), 
            '9988.HK' : 'Ali'}  # (阿里)
"""
德国股票通常有以下后缀：

.DE - 法兰克福证券交易所（主要）

.F - 法兰克福（有时也用作德国股票的通用后缀）

.ETR - 德意志交易所（Xetra交易系统）

.DUSS - 杜塞尔多夫证券交易所

.MUN - 慕尼黑证券交易所

.HAM - 汉堡证券交易所

.STU - 斯图加特证券交易所

.BER - 柏林证券交易所
"""
from tkinter.ttk import Combobox
from tkinter import LEFT, Label, Frame, BOTH, TOP, BOTTOM, Listbox, EXTENDED
class StockInfo:
    class PRODUCT_TYPE(AutoIndex):
        usa_stock = () #美股
        ger_stock = () #德股
        futures = ()   #期货
        indices = ()   #指数
        HongKong = ()  #港股

    def __init__(self, parent):
        self.all_stocks = dict(zip([p for p in StockInfo.PRODUCT_TYPE], [USA_STOCK, GER_STOCK, FUTURES, INDICES, HONGKONG]))
        self.parent = parent
        self.root = CreateChildWindow(self.parent, "Stock Info")
        self.create_gui()

    def create_gui(self):
        prod = Frame(self.root)
        prod.pack(fill=BOTH, expand=True, padx=10, pady=10)
        Label(prod, text='Product').pack(side=LEFT)
        self.product_combo = Combobox(prod, width=10, values=[''.join(p.name.split('_')) for p in StockInfo.PRODUCT_TYPE])
        self.product_combo.pack(side=LEFT)

        stock_frame = Frame(self.root)
        stock_frame.pack(fill=BOTH, expand=True, padx=10, pady=10)
        Label(stock_frame, text='Stocks').pack(side=TOP)
        stock_list = Listbox(stock_frame, height=10, width=25,
                                    selectmode=EXTENDED)  # 允许多选
        stock_list.pack(side=BOTTOM, fill=BOTH, expand=True)