from unittest import TestCase, TestSuite, defaultTestLoader, TextTestRunner
from stock_model import Stock_Model


class StockModelTest(TestCase):
    def setUp(self):
        pass
    
    @classmethod
    def setUpClass(cls):
        SYMBOL = "IFX.DE"
        HISTORY = "10y"
        cls.sm = Stock_Model(SYMBOL, HISTORY, path='./resource')
    
    @classmethod    
    def tearDownClass(self):
        pass
    
    def testInit(self):
        self.assertEqual(self.sm.stock_symbol, 'IFX.DE')
        self.assertEqual(self.sm.model_name, 'IFX.DE_10y_1d_60')
        self.assertEqual(self.sm.base_features.shape[1], 7)
        self.assertEqual(self.sm.window_size, 60)
        self.assertEqual(self.sm.path, './resource')

    def testSetWorkingData(self):
        pass
    
    def testScaleData(self):
        # self.assertEqual(self.sm._get_config_submodule(), '1')
        pass
    
    def testCreateTrainTestData(self):
        pass
    
    def testLoadKerasModel(self):
        self.sm.load_keras_model()
    
    def testLoadTrainData(self):
        pass
    
    def testLoadPredictData(self):
        pass
    
    def testLoadScaler(self):
        pass

from reference_methods.predict_by_model import Model_Prediction
class ModelPredictionTest(TestCase):
    @classmethod
    def setUpClass(cls):
        SYMBOL = "IFX.DE"
        cls.sm = Model_Prediction("./resource/IFX.DE_10y_1d_60.keras")
    
    @classmethod    
    def tearDownClass(self):
        pass
    
    def testMPPropert(self):
        stock = self.sm.stock
        self.assertEqual(stock.stock_symbol, 'IFX.DE')
        self.assertEqual(stock.model_name, 'IFX.DE_3mo_1d_60')
        self.assertEqual(stock.period, '3mo')
        self.assertEqual(stock.path, './resource/')

# def suite(codes):
#     TEST_CASES = {'ST':SubmoduleToolTest, 'EM':EmailTest, 'TB':TableTest, 'EV':EventHandlerTest, 'LG':LanguageTest}
#     s = TestSuite()
#     for k in codes:
#         s.addTests(defaultTestLoader.loadTestsFromTestCase(TEST_CASES[k]))
#     return s

import argparse    
if __name__ == '__main__':
    # parser = argparse.ArgumentParser(prog='Test.py', usage='%(prog)s [options]', description='Test Common package')
    # parser.add_argument('-s', '--submod', dest='case', action='append_const', const='ST',help='test submodule tool')
    # parser.add_argument('-e', '--email', dest='case', action='append_const', const='EM', help='test Email module')
    # parser.add_argument('-t', '--table', dest='case', action='append_const', const='TB', help='test Table module')
    # parser.add_argument('-v', '--event', dest='case', action='append_const', const='EV', help='test EventHandler module')
    # parser.add_argument('-l', '--language', dest='case', action='append_const', const='LG', help='test Language module')
    # args = parser.parse_args()
    s = TestSuite()
    s.addTest(defaultTestLoader.loadTestsFromTestCase(StockModelTest))
    s.addTest(defaultTestLoader.loadTestsFromTestCase(ModelPredictionTest))
    TextTestRunner().run(s)