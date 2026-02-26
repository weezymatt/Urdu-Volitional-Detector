"""
Scripts for F1-Score, Accuracy, and Coverage.
"""
import pdb

def f1_score(y_true, y_pred, labels) -> dict, float, float:
	assert len(y_true) == len(y_pred)
	f1_dict = {}
	tp, fp, fn, tn = 0, 0, 0, 0
	tp2, fp2, fn2, tn2 = 0, 0, 0, 0

	for label in labels:
		if label == 'OTHER':
			# Explicitly ignore the negative class 'OTHER.'
			continue

		for gold, pred in zip(y_true, y_pred):
			if gold == label and pred == label:
				tp += 1
				tp2 += 1
			if gold == label and pred != label:
				fn += 1
				fn2 += 1
			if gold != label and pred == label:
				fp += 1
				fp2 += 1
			if gold != label and pred != label:
				tn += 1
				tn2 += 1

			precision = tp / (tp + fp)
			recall = tp / (tp + fn)
			f1 = 2 * precision * recall / (precision + recall)
			f1_dict[label] = {"precision": precision, "recall": recall, "f1": f1}

	precision2 = tp2 / (tp2 + fp2)
	recall2 = tp2 / (tp2 + fn2)
	micro_f1 = 2 * precision2 * recall2 / (precision2 + recall2)
	macro_f1 = sum(f1_dict[label]['f1'] for label in labels[:-1]) / (len(labels) - 1)

	return f1_dict, macro_f1, micro_f1

def accuracy(y_true,y_pred) -> tuple[float, list[int]]:
	predictions_correct = []

	for gold,pred in zip(y_true, y_pred):
		if gold == pred:
			predictions_correct.append(1)
		else:
			predictions_correct.append(0)

	accuracy = sum(predictions_correct) / len(predictions_correct)

	return accuracy, predictions_correct

def coverage(y_true,y_preds):
	pass