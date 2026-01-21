# debug_yfinance.py
import yfinance as yf
import pandas as pd
import sys
from datetime import datetime

def test_yfinance_direct():
    """直接测试yfinance下载功能"""
    print("="*60)
    print("测试yfinance下载功能")
    print("="*60)
    
    test_tickers = ['AAPL', 'GOOGL', 'INVALIDTICKER']
    
    for ticker in test_tickers:
        print(f"\n测试股票代码: {ticker}")
        
        try:
            # 方法1: 使用Ticker对象
            print("方法1: 使用Ticker对象")
            stock = yf.Ticker(ticker)
            info = stock.info
            print(f"  公司名称: {info.get('longName', 'N/A')}")
            print(f"  当前价格: {info.get('regularMarketPrice', 'N/A')}")
            
            # 获取历史数据
            hist = stock.history(period="1mo")
            print(f"  历史数据行数: {len(hist)}")
            
        except Exception as e:
            print(f"  方法1失败: {type(e).__name__}: {e}")
        
        try:
            # 方法2: 直接使用download
            print("方法2: 使用download函数")
            data = yf.download(ticker, period="1mo", progress=False)
            print(f"  下载数据形状: {data.shape}")
            if not data.empty:
                print(f"  列名: {list(data.columns)}")
                
        except Exception as e:
            print(f"  方法2失败: {type(e).__name__}: {e}")

def test_yfinance_with_years():
    """测试按年份下载数据"""
    print("\n" + "="*60)
    print("测试按年份下载数据")
    print("="*60)
    
    ticker = 'AAPL'
    test_cases = [
        ("2020-01-01", "2020-12-31", "完整一年"),
        ("2020", "2023", "年份字符串"),
        (2020, 2023, "整数年份"),
    ]
    
    for start, end, description in test_cases:
        print(f"\n测试: {description} (start={start}, end={end})")
        
        try:
            # 转换日期格式
            if isinstance(start, int):
                start_str = f"{start}-01-01"
                end_str = f"{end}-12-31"
            elif isinstance(start, str) and len(start) == 4:
                start_str = f"{start}-01-01"
                end_str = f"{end}-12-31"
            else:
                start_str = str(start)
                end_str = str(end)
            
            print(f"  转换后: start={start_str}, end={end_str}")
            
            data = yf.download(ticker, start=start_str, end=end_str, progress=False)
            print(f"  数据形状: {data.shape}")
            
        except Exception as e:
            print(f"  失败: {type(e).__name__}: {e}")

def check_network_connectivity():
    """检查网络连接"""
    print("\n" + "="*60)
    print("检查网络连接")
    print("="*60)
    
    try:
        import urllib.request
        import urllib.error
        
        # 测试连接到yahoo finance
        test_urls = [
            "https://finance.yahoo.com",
            "https://query1.finance.yahoo.com"
        ]
        
        for url in test_urls:
            try:
                response = urllib.request.urlopen(url, timeout=10)
                print(f"✓ 可以连接到: {url}")
                print(f"  状态码: {response.status}")
            except urllib.error.URLError as e:
                print(f"✗ 无法连接到 {url}: {e.reason}")
            except Exception as e:
                print(f"✗ 连接 {url} 时出错: {e}")
                
    except Exception as e:
        print(f"网络检查失败: {e}")

def check_proxy_settings():
    """检查代理设置"""
    print("\n" + "="*60)
    print("检查代理设置")
    print("="*60)
    
    import os
    
    proxy_vars = ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy']
    
    for var in proxy_vars:
        if var in os.environ:
            print(f"发现代理设置: {var} = {os.environ[var]}")
        else:
            print(f"未设置代理: {var}")

if __name__ == "__main__":
    print("Python版本:", sys.version)
    print("yfinance版本:", yf.__version__)
    print("pandas版本:", pd.__version__)
    
    test_yfinance_direct()
    test_yfinance_with_years()
    check_network_connectivity()
    check_proxy_settings()