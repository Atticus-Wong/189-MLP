from code_main.base_class.evaluate import evaluate
from code_main.stage_2_code.Result_Loader import Result_Loader
import matplotlib.pyplot as plt
from code_main.stage_2_code.Evaluate_Accuracy import Evaluate_Accuracy
from code_main.stage_2_code.Evaluate_F1 import Evaluate_F1
from code_main.stage_2_code.Evaluate_Precision import Evaluate_Precision
from code_main.stage_2_code.Evaluate_Recall import Evaluate_Recall

if 1:
    result_obj = Result_Loader('saver', '')
    result_obj.result_destination_folder_path = '../../result/stage_2_result/MLP_'
    result_obj.result_destination_file_name = 'prediction_result'

    result_obj.load()

    print("got fields", result_obj.data.keys())

    training_metrics = []
    for metric in result_obj.data['training_metrics']:
        training_metrics.append(metric)

    accuracy_obj = Evaluate_Accuracy('accuracy', '')
    accuracy_obj.data = {'true_y': result_obj.data["true_y"], 'pred_y': result_obj.data["pred_y"]}
    test_accuracy_score = accuracy_obj.evaluate()

    f1_score_obj = Evaluate_F1('f1_score', '')
    f1_score_obj.data = {'true_y': result_obj.data["true_y"], 'pred_y': result_obj.data["pred_y"]}
    test_f1_score = f1_score_obj.evaluate()

    precision_obj = Evaluate_Precision('precision', '')
    precision_obj.data = {'true_y': result_obj.data["true_y"], 'pred_y': result_obj.data["pred_y"]}
    test_precision_score = precision_obj.evaluate()

    recall_obj = Evaluate_Recall('recall', '')
    recall_obj.data = {'true_y': result_obj.data["true_y"], 'pred_y': result_obj.data["pred_y"]}
    test_recall_score = recall_obj.evaluate()
    print("Testing set - Accuracy: {:.4f}, F1 Score: {:.4f}, Precision: {:.4f}, Recall: {:.4f}".format(
        test_accuracy_score, test_f1_score, test_precision_score, test_recall_score
    ))

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
    plt.title('Model Metrics Over Epoch')
    plt.legend()
    plt.grid(True)
    plt.show()

    loss = [entry['loss'] for entry in training_metrics]

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, loss, label='Loss')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Over Epoch')
    plt.legend()
    plt.grid(True)
    plt.show()
