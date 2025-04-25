'''
Concrete IO class for a specific dataset
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code_main.base_class.dataset import dataset
import csv # Import csv module for easier CSV handling
import numpy as np # Import numpy for potential array operations if needed later

class Dataset_Loader(dataset):
    data = None
    dataset_source_folder_path = None
    dataset_source_file_name = None

    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)

    def load(self):
        print('loading data...')
        X = []
        y = []
        # Use csv reader for robust handling of CSV format
        file_path = self.dataset_source_folder_path + self.dataset_source_file_name
        try:
            with open(file_path, 'r') as f:
                reader = csv.reader(f)
                # Skip header if there is one (assuming the first line might be headers)
                # next(reader, None) # Uncomment this line if your CSV file has a header row
                for row in reader:
                    # Ensure row is not empty
                    if row:
                        # Convert all elements to integers
                        # MNIST CSV usually has label in the first column
                        elements = [int(i) for i in row]
                        # The first element is the label (y)
                        y.append(elements[0])
                        # The rest are features (X) - pixel values
                        X.append(elements[1:])
        except FileNotFoundError:
            print(f"Error: File not found at {file_path}")
            return None
        except Exception as e:
            print(f"An error occurred while loading the data: {e}")
            return None

        # Consider converting lists to numpy arrays for efficiency if doing numerical operations
        # X = np.array(X)
        # y = np.array(y)
        return {'X': X, 'y': y}