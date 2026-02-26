import argparse
import csv
import stanza

from ergative_detector import ErgativeDetector
from metrics import f1_score, accuracy

def read_data(infile):
	"""
	Read function moses style into a generator.
	"""
	for line in infile:
		yield line.strip()

def csvread_data(infile, header=True):
	with open(infile, "r", newline='', encoding='utf-8') as csvfile:
		read_data = csv.reader(csvfile)
		input_rows = list(read_data)
	return input_rows

def ablation():
	vol0 = "Accuracy: 0.59 [645 / 1096]" # base
	vol1 = "Accuracy: 0.68 [745 / 1096]" # with inf
	vol2 = "Accuracy: 0.82 [901 / 1096]" # with obj present rule
	nvol = "Accuracy: 0.87 [903 / 1040]" # wide net: ambiguous rules

def error_analysis():
	pass

def main():
	parser = argparse.ArgumentParser(description="Detect the case of a document.")
	parser.add_argument('filename')
	args = parser.parse_args()

	nlp = stanza.Pipeline('ur')
	clf = ErgativeDetector()

	dataset = csvread_data(args.filename)
	labels = ["VOL", "NVOL", "OTHER"]
	y_pred = []
	y_true = []

	for i, row in enumerate(dataset[1:]):
		sentence = row[1]
		gold = row[-1]

		doc = nlp(sentence)

		pred = clf.detect_case(doc)

		y_pred.append(pred)
		y_true.append(gold)
		print(f"{i} PRED: {pred} | GOLD: {gold} | {sentence} | {row[0]}")

	acc, pred_correct = accuracy(y_true, y_pred)
	_, macrof1, microf1 = f1_score(y_true, y_pred, labels)
	print("-" * 40)
	print(f"  Accuracy: {acc:.3f}",
		  f"[{sum(pred_correct)} / {len(pred_correct)}]")

	print(f"  Macro-f1: {macrof1:.3f}\n",
		  f" Micro-f1: {microf1:.3f}",
		  )

if __name__ == '__main__':
	main()