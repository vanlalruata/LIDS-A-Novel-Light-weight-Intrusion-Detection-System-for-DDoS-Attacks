from sklearn import metrics
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix, precision_score
import torch
from tqdm import tqdm
import warnings
import sys
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = "cpu"


def evaluate_model(pred, labels):
    accuracy = metrics.accuracy_score(pred, labels)
    precision = metrics.precision_score(pred, labels)
    recall = metrics.recall_score(pred, labels)
    f1 = metrics.f1_score(pred, labels)
    roc_auc = metrics.roc_auc_score(pred, labels)

    print("Accuracy {}, Precision {}, Recall {}, f1 {}, roc_auc {}".format(accuracy, precision, recall, f1, roc_auc))


def classification_report(outputs, labels):
    """
    report is in batches inorder to save memory usage.
    The function gives various matrics performance of the model
    Args:
        outputs: predictions by the model
        labels: original labels
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        predictions = torch.max(outputs, 1)[1].to(device)

        predictions = predictions.cpu().numpy()
        labels = labels.cpu().numpy()
        accuracy = accuracy_score(predictions, labels)
        recall = recall_score(predictions, labels)
        f1 = f1_score(predictions, labels)
        cm = confusion_matrix(predictions, labels)
        precision = precision_score(predictions, labels)
        print("Accuracy : {}, Recall : {}, F1 Score : {} Precision {}".format(accuracy, recall, f1, precision))
        print("************************* Confusion Matrix ***********************")
        print(cm)


def classification_report_multi(outputs, labels):
    """
    report is in batches inorder to save memory usage.
    The function gives various matrics performance of the model
    Args:
        outputs: predictions by the model
        labels: original labels
    """

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        predictions = torch.max(outputs, 1)[1].to(device)

        predictions = predictions.cpu().numpy()
        labels = labels.cpu().numpy()
        accuracy = accuracy_score(predictions, labels)
        recall = recall_score(predictions, labels, average='weighted')
        f1 = f1_score(predictions, labels, average='weighted')
        cm = confusion_matrix(predictions, labels)
        precision = precision_score(predictions, labels, average='weighted')
        # precision = precision_score(predictions, labels)
        print("Accuracy : {}, Recall : {}, F1 Score : {} Precision {}".format(accuracy, recall, f1, precision))
        print("************************* Confusion Matrix ***********************")
        print(cm)


def evaluate_proposed_model(model, test_loader, mode):
    """
    Args:
        model: trained model
        test_loader: test dataset loader
        mode: binary or multiclass
    """

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        count = 1
        with torch.no_grad():
            correct = 0
            total_samples = 0

            test_bar = tqdm(test_loader, file=sys.stdout)

            for step, (samples, labels) in enumerate(test_bar):

                samples, labels = samples.to(device), labels.to(device)

                output = model(samples)

                print("Report on BATCH: ", count)
                if mode == 'binary':
                    classification_report(output, labels)
                else:
                    classification_report_multi(output, labels)
                count += 1

                start_time = time.time()
                output = model(samples)
                execution_time = time.time() - start_time
                print('*****************************************')
                print("Execution time: {}, Per sample: {}".format(execution_time, execution_time / len(samples)))


def accuracy(outputs, labels):
    """
    Args:
        outputs: predictions from the model
        labels: original labels
    return:
        correct: total number of correct classifications
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        predictions = torch.max(outputs, 1)[1].to(device)
        correct = (predictions == labels).sum().to(device)

        acc = correct / len(predictions)
        acc = acc.cpu().numpy()
        return acc
