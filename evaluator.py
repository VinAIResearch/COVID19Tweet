import sys
import os

label_map = {"UNINFORMATIVE": 0, "INFORMATIVE": 1}


def get_labels(file_in, gold_indices=None):
    """
    Read the labels from file
    :param file_in: path to the file
    :param gold_indices: set of gold indices
    :return: list of labels
    """
    labels = []
    count = 0
    with open(file_in) as reader:
        for line in reader:
            line = line.strip()
            if line == "Id\tText\tLabel":
                continue
            count += 1
            if gold_indices is not None and count not in gold_indices:
                continue

            label = line.split()[-1].upper()
            if len(label) == 0:
                continue

            if label in label_map:
                labels.append(label_map[label])
            else:
                print("Error occurs at line {}. "
                      "{} is not a label (only support UNINFORMATIVE and INFORMATIVE). "
                      "Process terminated.".format(count + 1, label))
                sys.exit()


    return labels


def calculate_scores(pred_labels, true_labels, pos_label=label_map["INFORMATIVE"]):
    """
    Calculate the precision, recall, f1 and accuracy scores for the predictions
    :param pred_labels: prediction labels
    :param true_labels: ground truth labels
    :param pos_label: INFORMATIVE label
    :return: precision, recall, f1 and accuracy scores
    """
    assert len(pred_labels) == len(true_labels)

    tp = 0  # true positive
    fn = 0  # false negative
    fp = 0  # false positive
    n_correct = 0
    for i in range(len(pred_labels)):
        if true_labels[i] == pred_labels[i]:
            n_correct += 1
            if pred_labels[i] == pos_label:
                tp += 1
        else:
            if pred_labels[i] == pos_label:
                fp += 1
            else:
                fn += 1
    # Precision score
    precision = 0.0
    if tp + fp > 0:
        precision = tp * 1.0 / (tp + fp)

    # Recall score
    recall = 0.0
    if tp + fn > 0:
        recall = tp * 1.0 / (tp + fn)

    # F1 score
    f1 = 0.0
    if recall + precision > 0:
        f1 = 2 * precision * recall / (precision + recall)

    # Accuracy score
    accuracy = 0.0
    if len(true_labels) > 0:
        accuracy = n_correct * 1.0 / len(true_labels)
    return precision, recall, f1, accuracy


def evaluate(pred_label_file, true_label_file, gold_indices_file=None):
    gold_indices = get_gold_indices(gold_indices_file)
    pred_labels = get_labels(pred_label_file, gold_indices)
    true_labels = get_labels(true_label_file)
    return calculate_scores(pred_labels, true_labels)


def get_gold_indices(gold_indices_file):
    if gold_indices_file is None:
        return None

    gold_indices = set()
    with open(gold_indices_file, "r") as reader:
        for line in reader:
            line = line.strip()
            if len(line) > 0:
                gold_indices.add(int(line))
    return gold_indices


def score(input_dir, output_dir):
    # unzipped submission data is always in the 'res' subdirectory
    submission_dir = os.path.join(input_dir, 'res')
    submission_file = []
    for el in os.listdir(submission_dir):
        if el.startswith('predictions'):
            submission_file.append(el)
    if not len(submission_file) == 1:
        print(
            "Warning: the submission folder should only contain 1 file (i.e., 'predictions.txt'). Process terminated.")
        sys.exit()
    submission_file_name = submission_file[0]
    submission_path = os.path.join(submission_dir, submission_file_name)
    ground_truth_file = os.path.join(input_dir, 'ref', 'groundtruth_data.txt')
    gold_indices_file = os.path.join(input_dir, 'ref', 'gold_indices.txt')
    precision, recall, f1, accuracy = evaluate(submission_path, ground_truth_file, gold_indices_file)
    with open(os.path.join(output_dir, 'scores.txt'), 'w') as output_file:
        output_file.write("F1-score:{}\nPrecision:{}\nRecall:{}\nAccuracy:{}\n".format(f1, precision, recall, accuracy))


def main():
    [_, input_dir, output_dir] = sys.argv
    score(input_dir, output_dir)


if __name__ == "__main__":
    #main()
    [_, pred_label_file, true_label_file] = sys.argv
    precision, recall, f1, accuracy = evaluate(pred_label_file, true_label_file)
    print("F1-score: {}\nPrecision: {}\nRecall: {}\nAccuracy: {}\n".format(f1, precision, recall, accuracy))
