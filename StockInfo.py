from Common.AutoNumber import AutoIndex
from Common.Util import CreateChildWindow, CloseChildWindow
from stock_prediction import StockPredictionGUI
from Common.EventHandler import Event

USA_STOCK = {
        'AAPL'  : 'Aple',
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
            '^CXKAX' : 'Automobil',       # 汽车指数 - 包含德国汽车制造商的指数, 奔驰、宝马、大众
            '^CXPBX' : 'Bank',       # 银行指数 - 包含德国主要银行的指数, 德意志银行、商业银行
            '^CXKTX' : 'Tech',       # 科技指数 - 包含德国科技公司的指数, 英飞凌、SAP
            '^CXKIX' : 'Insurance',       # 保险指数 - 包含德国主要保险公司的指数, 安联、慕尼黑再保险
            '^CXKPX' : 'Industrie',       # 能源指数 - 包含德国主要工业公司的指数, 西门子、DHL、空中客车
            '^CXKCX' : 'Chemie',       # 化工指数 - 包含德国主要化工公司的指数, 巴斯夫、赢创工业、科思创
            '^CXKDX' : 'Pharma',       # 医药指数 - 包含德国主要医药公司的指数, 默克、拜耳、弗雷森纽斯医疗
            '^CXKUX' : 'Communication',       # 通信指数 - 包含德国主要通信公司的指数, 德国电信、沃达丰、1&1
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
import os, json
from tkinter.ttk import Combobox
from tkinter import Label, Frame, Listbox, Scrollbar, Entry, Button, StringVar, Widget
from tkinter import messagebox
from tkinter import LEFT, BOTH, TOP, BOTTOM, VERTICAL, RIGHT, Y, WORD, END, INSERT
from tkinter.scrolledtext import ScrolledText
from StockEvent import StockEvent

class StockInfo:
    """
    StockInfo
    manage stock information about ticker
    """
    class PRODUCT_TYPE(AutoIndex):
        indices = ()   #指数
        ger_stock = () #德股
        futures = ()   #期货
        usa_stock = () #美股
        HongKong = ()  #港股

    class SuffixGermanStockExchange(AutoIndex):
        DE = ()
        F = ()
        ETR = ()
        DUSS = ()
        MUN = ()
        HAM = ()
        STU = ()
        BER = () 
    GermanStockExchange = ['Frankfurt', 'Frankfurt', 'Xetra', 'Duesseldorf', 'Munich', 'Hamburg', 'Stuttgart', 'Berlin']
    StockInfoPath = r'./info'
    StockInfoFile = 'stock_info.cfg'

    @staticmethod
    def get_product_type_text(prod_type:PRODUCT_TYPE)->str:
        return ' '.join(prod_type.name.split('_'))
    
    @staticmethod
    def get_product_type_by_index(index:int)->PRODUCT_TYPE:
        for t in StockInfo.PRODUCT_TYPE:
            if t.value == index:
                return t

    @staticmethod
    def get_product_type_by_text(text:str)->PRODUCT_TYPE:
        for t in StockInfo.PRODUCT_TYPE:
            if StockInfo.get_product_type_by_text(t) == text:
                return t

    def __init__(self, parent:Widget=None):
        self.GermanStockExchangeCodes = dict(zip([c.name for c in StockInfo.SuffixGermanStockExchange],
                                                StockInfo.GermanStockExchange))
        self.stock_info_file = os.path.join(StockInfo.StockInfoPath, StockInfo.StockInfoFile)
        self.load_stock_info()
        self.parent = parent
        self.product_index = 0
        self.stock_selected_index = 0
        if self.parent:
            self.root = CreateChildWindow(self.parent, 'stock info', modal=True, XClose=True)
            self.create_gui()

    def total_stock_count(self):
        count = 0
        for prod_type in self.stock_info_data:
            stock_dict = self.stock_info_data[prod_type]
            count += len(stock_dict)
        return count

    def load_stock_info(self):
        if os.path.exists(self.stock_info_file):
            print(f"Loading stock info from {self.stock_info_file}")
            with open(self.stock_info_file, 'r') as info:
                stock_info_info = json.load(info)
            # convert product from string to PRODUCT_TYPE
            self.stock_info_data = dict(zip([c for c in StockInfo.PRODUCT_TYPE], [None] * len(StockInfo.PRODUCT_TYPE)))
            for p in self.stock_info_data:
                for s in stock_info_info:
                    if s == p.name:
                        self.stock_info_data[p] = stock_info_info[s]
        else:
            # reform data
            product_data = [INDICES, GER_STOCK, FUTURES, USA_STOCK, HONGKONG]
            for i, p in enumerate(product_data):
                for s, v in p.items():
                    product_data[i][s] = {'company':v}
            self.stock_info_data = dict(zip([p for p in StockInfo.PRODUCT_TYPE], product_data))
            self.save_stock_info()

    def get_stock_info(self, ticker_symbol):
        for p in self.stock_info_data:
            for s in self.stock_info_data[p]:
                if s == ticker_symbol:
                    return self.stock_info_data[p][s]
        return None

    def update_stock_info(self):
        import threading
        popup = CreateChildWindow(self.root, 'Stock info', geometry='200x50')
        show_text = 'Updating stock info...'
        count = self.total_stock_count()
        text_var = StringVar()
        text_var.set(f"{show_text} (0/{count})")
        Label(popup, textvariable=text_var, justify='center').pack(fill=BOTH, anchor='center', padx=10, pady=10)
        def run_update(callback):
            updated_count = 1
            for prod_type in self.stock_info_data:
                stock_dict = self.stock_info_data[prod_type]
                for stock_code in stock_dict:
                    print(f"Updating info for {stock_code}...")
                    success, result = self.obtain_stock_info(stock_code)
                    if success:
                        stock_dict[stock_code] = result
                    else:
                        print(f"Failed to obtain info for {stock_code}: {result}")
                    print(f"Updated {updated_count}/{count} stocks.")
                    text_var.set(f"{show_text} ({updated_count}/{count})")
                    popup.update()
                    updated_count += 1
            self.save_stock_info()
            callback()  # 在更新完成后调用回调函数关闭弹窗
        func =lambda: self.root.after(0, popup.destroy)
        threading.Thread(target=run_update, daemon=True, args=(func,)).start()

    def obtain_stock_info(self, symbol:str):
        import yfinance as yf
        try:
            # 确保使用正确的后缀
            # if not any(symbol.endswith(suffix) for suffix in ['.DE', '.F', '.ETR', '.DUSS', '.MUN', '.HAM', '.STU', '.BER']):
            #     symbol_with_suffix = symbol + '.DE'
            # else:
            #     symbol_with_suffix = symbol
            
            ticker = yf.Ticker(symbol)
            info = ticker.info
            company_name = info.get('longName', info.get('shortName', symbol))
            sector = info.get('sector', 'N/A')
            industry = info.get('industry', 'N/A')
            result = {
                        'symbol': symbol,
                        'company': company_name,
                        'sector': sector,
                        'industry': industry
                    }
            return True, result
        except Exception as e:
            return False, str(e)

    def save_stock_info(self):
        if not os.path.exists(StockInfo.StockInfoPath):
            os.mkdir(StockInfo.StockInfoPath)
        print(f"Saving stock info to {self.stock_info_file}")
        self.save_stock_info_data = {}
        for prod_type, stock_dict in self.stock_info_data.items():
            if isinstance(prod_type, StockInfo.PRODUCT_TYPE):
                self.save_stock_info_data[prod_type.name] = stock_dict
        with open(self.stock_info_file, 'w+') as info:
            json.dump(self.save_stock_info_data, info)

    def get_stock_list_by_product_type(self, prod_type:PRODUCT_TYPE)->str:
        stock_list = self.stock_info_data[prod_type]
        return '\n'.join(stock_list)

    def get_stock_name_by_selected_stock_index(self):
        if self.product_index < 0:
            return
        # get product type by selected product index
        prod_type = StockInfo.get_product_type_by_index(self.product_index)
        # get stock dict by product type
        stock_dict = self.stock_info_data.get(prod_type, None)
        if stock_dict is None:
            return
        # get stock text by stock index in the stock list of product type in the list box
        # remove prefix
        key_list_text = self.stock_list.get(self.stock_selected_index).strip()
        parts = key_list_text.split('[')
        key_text = parts[1][:-1]
        return key_text

    def get_stock_info_text(self):
        if self.product_index < 0:
            return
        # get product type by selected product index
        prod_type = StockInfo.get_product_type_by_index(self.product_index)
        # get stock dict by product type
        stock_dict = self.stock_info_data.get(prod_type, None)
        if stock_dict is None:
            return
        # get stock text by stock index in the stock list of product type in the list box
        # remove prefix
        key_list_text = self.stock_list.get(self.stock_selected_index).strip()
        parts = key_list_text.split('[')
        key_text = parts[1][:-1]
        info = ''
        for k,v in stock_dict[key_text].items():
            info += f'{k} : {v}\n'

    def create_gui(self):
        prod = Frame(self.root)
        prod.pack(fill=BOTH, expand=True, padx=10, pady=10)
        frame1 = Frame(prod)
        frame1.pack(side=LEFT, fill='x', expand=True)
        Label(frame1, text='Product').pack(side=LEFT)
        self.product_combo = Combobox(frame1, width=10,
                                      values=[''.join(p.name.split('_')) for p in StockInfo.PRODUCT_TYPE],
                                      state='readonly')
        self.product_combo.pack(side=LEFT, padx=5)
        self.product_combo.bind('<<ComboboxSelected>>', self._on_product_change)
        Label(frame1, text='New Stock').pack(side=LEFT, padx=5)
        self.new_stock_entry = Entry(frame1, width=16)
        self.new_stock_entry.pack(side=LEFT, padx=5)
        self.add_stock_button = Button(frame1, text='Add', command=self.add_new_stock)
        self.add_stock_button.pack(side=LEFT, padx=5)
        # frame 2
        stock_frame = Frame(self.root)
        stock_frame.pack(fill=BOTH, expand=True, padx=10, pady=10)
        # left frame of frame 2
        Label(stock_frame, text='Stocks').pack(side=LEFT, anchor='n')
        listbox_frame = Frame(stock_frame)
        listbox_frame.pack(side=LEFT)
        self.stock_list = Listbox(listbox_frame, height=10, width=20, selectmode='single')
        self.stock_list.pack(side=LEFT, fill=BOTH, expand=True)
        self.stock_list.bind('<<ListboxSelect>>', self._on_select_stock)
        # 滚动条
        scrollbar = Scrollbar(listbox_frame, orient=VERTICAL, command=self.stock_list.yview)
        scrollbar.pack(side=RIGHT, fill=Y)
        self.stock_list.config(yscrollcommand=scrollbar.set)
        self.product_combo.current(0)
        self.product_index = 0
        self.update_stock_listbox(self.product_index)

        # right frame of frame 2
        frame2_right = Frame(stock_frame)
        frame2_right.pack(side=LEFT, fill='x', expand=True, padx=10)
        # info field
        self.info_field = ScrolledText(frame2_right, wrap=WORD,
                                        width=40, height=8,
                                        font=("Times New Roman", 15))    
        self.info_field.pack(padx=5, pady=10, anchor='n')
        # button update stock info
        self.update_stock_info_button = Button(frame2_right, text='Update Stock Info', command=self.update_stock_info)
        self.update_stock_info_button.pack(anchor='center')
        # buttons
        btn_frame = Frame(self.root)
        btn_frame.pack(anchor='center', expand=True, padx=10, pady=10)
        self.select_button = Button(btn_frame, text='select', command=self._on_selected, anchor='center')
        self.select_button.pack(side=LEFT, padx=10)
        Button(btn_frame, text='close', command=self.on_exit).pack(side=LEFT, padx=10)

    def _on_product_change(self, event):
        self.product_index = self.product_combo.current()
        print(StockInfo.get_product_type_by_index(self.product_index).name)
        self.update_stock_listbox(self.product_index)

    def _on_select_stock(self, event):
        self.stock_selected_index = self.stock_list.curselection()[0]
        print(f"selected text of index {self.stock_selected_index}: {self.stock_list.get(self.stock_selected_index)}")
        self.info_field.delete('1.0', END)
        self.info_field.insert(INSERT, self.get_stock_info_text())
        line = self.stock_list.get(self.stock_selected_index)
        self.selected_stock_name = line.split('[')[1]
        self.selected_stock_name = self.selected_stock_name.strip()[:-1]
        print("Selected stock name: ", self.selected_stock_name)

    def _on_selected(self):
        StockPredictionGUI.event_handler(StockEvent.selected_stock, self.selected_stock_name)
        self.on_exit()

    def update_stock_listbox(self, index):
        """更新Listbox显示"""
        # 清空Listbox
        self.stock_list.delete(0, END)
        stock_list = self.stock_info_data[StockInfo.get_product_type_by_index(index)]
        # 添加所有股票
        for i, key in enumerate(stock_list.keys()):
            self.stock_list.insert(END, f"{i}.[{key}]")
        # 更新状态显示
        self.stock_list.update()

    def add_new_stock(self):
        new_stock = self.new_stock_entry.get().strip()
        if not new_stock:
            messagebox.showwarning("Input Error", "Please enter a stock code.")
            return
        if self.product_index < 0:
            messagebox.showwarning("Selection Error", "Please select a product type first.")
            return
        prod_type = StockInfo.get_product_type_by_index(self.product_index)
        stock_dict = self.stock_info_data.get(prod_type, None)
        if stock_dict is None:
            messagebox.showerror("Error", "Invalid product type selected.")
            return
        if new_stock in stock_dict:
            messagebox.showwarning("Duplicate Error", "This stock code already exists.")
            return
        # Add new stock with default name same as code
        stock_dict[new_stock] = new_stock
        self.update_stock_listbox(self.product_index)
        self.new_stock_entry.delete(0, END)

    def on_exit(self):
        CloseChildWindow(self.root)

def open_stock_info(parent):
    StockInfo(parent)

if __name__ == '__main__':
    print(f'Text of Product Type ger_stock: {StockInfo.get_product_type_text(StockInfo.PRODUCT_TYPE.ger_stock)}')
    print(f'Product Type name of index 1: {StockInfo.get_product_type_by_index(1).name}')
    from tkinter import Tk
    root = Tk()
    root.geometry('100x50')
    # frm = Frame(root)
    # frm.pack(fill='both')
    btn = Button(root, text='Click me', command=lambda p=root: open_stock_info(p))
    btn.pack(side='top', fill='x', pady=10)
    root.mainloop()