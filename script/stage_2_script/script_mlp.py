import os
import sys
import torch
import numpy as np

# Add the project root directory to the Python path
# This allows importing modules from the 'code' directory
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, root_path)

# Import necessary classes from stage_2_code
from code_main.stage_2_code.Dataset_Loader import Dataset_Loader
from code_main.stage_2_code.Method_MLP import Method_MLP
from code_main.stage_2_code.Result_Saver import Result_Saver
from code_main.stage_2_code.Setting_Train_Test_Split import Setting_Train_Test_Split
from code_main.stage_2_code.Evaluate_Accuracy import Evaluate_Accuracy

#---- Multi-Layer Perceptron script ----
if __name__ == '__main__':
    #---- parameter section -------------------------------
    # Set seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    #------------------------------------------------------

    # ---- objection initialization section ---------------
    # Initialize Dataset_Loader for training data
    data_obj_train = Dataset_Loader('MNIST Train', 'Handwritten digit recognition training data')
    data_obj_train.dataset_source_folder_path = '../../data/stage_2_data/' 
    data_obj_train.dataset_source_file_name = 'train.csv' 

    # Initialize Dataset_Loader for testing data
    data_obj_test = Dataset_Loader('MNIST Test', 'Handwritten digit recognition testing data')
    data_obj_test.dataset_source_folder_path = '../../data/stage_2_data/'
    data_obj_test.dataset_source_file_name = 'test.csv'

    # Initialize Method_MLP
    # Parameters like hidden layers, learning rate, epochs etc., might be set within the class or passed here
    method_obj = Method_MLP('multi-layer perceptron', '')

    # Initialize Result_Saver
    result_obj = Result_Saver('saver', '')
    result_obj.result_destination_folder_path = '../../result/stage_2_result/MLP_'
    result_obj.result_destination_file_name = 'prediction_result'

    # Initialize Setting_Train_Test_Split
    # Pass both train and test data objects to the setting
    setting_obj = Setting_Train_Test_Split('train test split', '', data_obj_test) # Assuming the setting takes test_data_obj

    # Initialize Evaluate_Accuracy
    evaluate_obj = Evaluate_Accuracy('accuracy', '')
    # ------------------------------------------------------

    # ---- running section ---------------------------------
    print('************ Start ************')
    # Prepare the setup using the training data object
    setting_obj.prepare(data_obj_train, method_obj, result_obj, evaluate_obj)
    setting_obj.print_setup_summary()
    # Load data, run the model, save results, and evaluate
    mean_score, std_score = setting_obj.load_run_save_evaluate()
    print('************ Overall Performance ************')
    # Accuracy score is printed; std_score might be 0 or NaN if only one run is performed
    print('MLP Accuracy: ' + str(mean_score) + ' +/- ' + str(std_score))
    print('************ Finish ************')
    # ------------------------------------------------------