from unittest import *
from stock_model import Stock_Model


class StockModelTest(TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        SYMBOL = "IFX.DE"
        HISTORY = "10y"
        cls.sm = Stock_Model(SYMBOL, HISTORY, path='./resource')
    
    @classmethod    
    def tearDownClass(self) -> None:
        pass
    
    def testSetWorkingData(self):
        pass
    
    def testScaleData(self):
        # self.assertEqual(self.sm._get_config_submodule(), '1')
        pass
    
    def testCreateTrainTestData(self):
        pass
    
    def testLoadModel(self):
        pass
    
    def testLoadTrainData(self):
        pass
    
    def testLoadPredictData(self):
        pass
    
    def testLoadScaler(self):
        pass
    
def usage():
    print ("Usage: stock_model.py fin|data|cm DAYS|pred")
    print ("fin: get data directly from finance markt")
    print ("fin: get data from saved data file")
    print ("cm DAYS: get confusion matrix for delay days DAYS")
    
if __name__ == "__main__":
    # import sys
    # if len(sys.argv) == 1:
    #     usage()
    # sm = Stock_Model(SYMBOL, HISTORY, path='./resource')
    # if sys.argv[1] == "fin":
    #     sm.train_model_with_actual_data()
    # elif sys.argv[1] == "data":
    #     sm.train_model_with_loaded_data()
    # elif sys.argv[1] == "cm": # confusion matrix
    #     nday = sys.argv[2]
    #     sm.load_predict_data()
    #     sm.get_confusion_matrix(nday)
    # elif sys.argv[1] == "pred":
    #     sm.load_predict_data()
    #     sm.evaluate_result()
    #     sm.save_evaluation_data()
    # else:
    #     usage()        
    pass