from local_code.stage_2_code.Result_Loader import Result_Loader
import matplotlib.pyplot as plt

if 1:
    result_obj = Result_Loader('saver', '')
    result_obj.result_destination_folder_path = '../../result/stage_2_result/MLP_'
    result_obj.result_destination_file_name = 'prediction_result'

    training_metrics = []

    for fold_count in [1, 2, 3]:
        result_obj.fold_count = fold_count
        result_obj.load()
        print('Fold:', fold_count, ', Result:', result_obj.data)

        for metric in result_obj.data['training_metrics']:
            metric['epoch'] = metric['epoch'] + ((fold_count - 1) * 200)
            training_metrics.append(metric)

    epochs = [entry['epoch'] for entry in training_metrics]
    accuracy = [entry['accuracy'] for entry in training_metrics]
    f1_score = [entry['f1_score'] for entry in training_metrics]
    precision = [entry['precision'] for entry in training_metrics]
    recall = [entry['recall'] for entry in training_metrics]

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, accuracy, label='Accuracy')
    plt.plot(epochs, f1_score, label='F1 Score')
    plt.plot(epochs, precision, label='Precision')
    plt.plot(epochs, recall, label='Recall')

    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.title('Model Metrics Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.show()