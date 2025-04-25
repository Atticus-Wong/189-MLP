'''
Concrete SettingModule class for a specific experimental SettingModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.setting import setting
from sklearn.model_selection import train_test_split
import numpy as np

class Setting_Train_Test_Split(setting):
    fold = 3
    
    def load_run_save_evaluate(self):
        
        # load dataset
        loaded_data = self.dataset.load()

        X, y = loaded_data['X'], loaded_data['y']

        # run MethodModule
        self.method.data = {'X': X, 'y': y}
        learned_result = self.method.run()
            
        # save raw ResultModule
        self.result.data = learned_result
        self.result.save()
            
        self.evaluate.data = learned_result
        
        return self.evaluate.evaluate(), None

        