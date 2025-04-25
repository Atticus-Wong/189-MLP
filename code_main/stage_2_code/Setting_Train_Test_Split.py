'''
Concrete SettingModule class for a specific experimental SettingModule
Handles loading pre-defined train and test datasets.
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code_main.base_class.setting import setting
# train_test_split is not needed as we have separate train/test files
# from sklearn.model_selection import train_test_split
import numpy as np

class Setting_Train_Test_Split(setting):
    
    # test_dataset object needs to be provided during initialization
    test_dataset = None

    def __init__(self, sName=None, sDescription=None, test_dataset_obj=None):
        super().__init__(sName, sDescription)
        if test_dataset_obj is None:
            raise ValueError("Test dataset object must be provided for Setting_Train_Test_Split in Stage 2.")
        self.test_dataset = test_dataset_obj

    def load_run_save_evaluate(self):
        
        # load training dataset
        print('Loading training data...')
        train_loaded_data = self.dataset.load()
        if train_loaded_data is None or 'X' not in train_loaded_data or 'y' not in train_loaded_data:
             raise ValueError("Training data not loaded correctly by the dataset loader.")
        X_train, y_train = train_loaded_data['X'], train_loaded_data['y']
        print(f'Training data loaded: {len(X_train)} samples.')

        # load testing dataset
        print('Loading testing data...')
        test_loaded_data = self.test_dataset.load()
        if test_loaded_data is None or 'X' not in test_loaded_data or 'y' not in test_loaded_data:
             raise ValueError("Testing data not loaded correctly by the dataset loader.")
        X_test, y_test = test_loaded_data['X'], test_loaded_data['y']
        print(f'Testing data loaded: {len(X_test)} samples.')

        # No need to split, we already have train and test sets
        # X_train, X_test, y_train, y_test = train_test_split(loaded_data['X'], loaded_data['y'], test_size = 0.33)

        # run MethodModule
        self.method.data = {'train': {'X': X_train, 'y': y_train}, 'test': {'X': X_test, 'y': y_test}}
        learned_result = self.method.run()
            
        # save raw ResultModule
        self.result.data = learned_result
        self.result.save()
            
        self.evaluate.data = learned_result
        
        # Evaluate and return score, std deviation is None as it's a single train/test run
        return self.evaluate.evaluate(), None

