"""
Code to extract ergative and non-volitional (i.e., accidental) instances in Urdu.
"""
import argparse
import csv
from typing import Optional, Dict
from collections import defaultdict
from pythonjsonlogger import jsonlogger


class ErgativeDetector:
	"""
	Urdu rule-based Ergative detector with morph-based rule selection.
	"""
	def __init__(self):
		self.vol_marker = "نے"
		self.nvol_marker = "سے"

	def detect_case(self, sentence) -> Optional[str]:
		"""
		Detect ergative (ERG) or absolutive (ABS) case of a document.

		Args:
			sentence: a tokenized sentence. The Token object provides a set of linguistic annotations. 

		Returns:
			String indicating the case of a sentence
				VOL if the document contains ERG
				NVOL if the document contains ABS
				OTHER if the document contains neither
		"""
		case = None

		possible_markers = []
		for word in sentence.iter_words():
			marker = self.check_marker(word)
			if marker:
				possible_markers.append(marker)

		if possible_markers:
			# do full pass & verify verb pattern
			case = self.check_pattern(sentence, possible_markers)

		assert case in ["VOL", "NVOL", "OTHER"]
		return case

	def check_marker(self, word) -> Optional[str]:
		"""
		First pass to check if a marker is present.
		"""
		features = word.feats

		if features and "AdpType=Post" in features and "ADP" in word.upos:
			if word.text.endswith(self.vol_marker):
				return "VOL"
			if word.text.endswith(self.nvol_marker):
				return "NVOL"

		return None


	def check_pattern(self, sentence, markers) -> str:
		"""
		Use Urdu's split ergativity to verify markers via minimal pairs.
		"""
		rules = set(markers)

		if "VOL" in rules:
			# priority given to VOL
			return self.ergative_pattern(sentence)
		if "NVOL" in rules:
			return self.absolutive_pattern(sentence)

		return "OTHER"

	@staticmethod
	def get_feats(word) -> Dict[str, str]:
		"""
		Helper function to create dict for features.
		"""
		if not word.feats:
			return {}

		pairs = (ft.split("=") for ft in word.feats.split("|"))

		return {k:v for k,v in pairs}

	def ergative_pattern(self, sentence) -> str:
		"""
		Function to verify the ergative pattern. Propoosed that ergative marker has the 
		semantic feature of volition.

		Pattern: 
			- Linguistic: The ergative case is marked in the perfective aspect for 
			transitive and ditransitive verbs.
			- Linguistic: Gender and Number agreement between perfective verb and object
			- Nsubj using Accusative case
		"""
		children = defaultdict(list)
		words = list(sentence.iter_words())

		# Create dict of head -> child
		for word in words:
			children[word.head].append(word)

		for word in words:
			feats = word.feats
			# Find transitive / ditransitive verbs
			if "VERB" in word.upos and feats and ("Aspect=Perf" in feats or "VerbForm=Inf" in feats):
				verb_feats = self.get_feats(word)

				# Handle compound verbs
				verb_id = word.head if "compound" in word.deprel else word.id
				subj_found = False
				obj_found = False
				obj_agreement = False

				# Find obj marker
				for child in children.get(verb_id, []):
					child_feats = self.get_feats(child)

					if (
						"Inf" in verb_feats.get("VerbForm", "") # MODALS
						and "AUX" in child.upos
						): 
						return "VOL"

					if (
						"nsubj" in child.deprel
						and "Acc" in child_feats.get("Case", "") # RELATIVE CLAUSES
						or "Nom" not in child_feats.get("Case", "")
						): 
							subj_found = True
					if (
						"obj" in child.deprel
						and verb_feats.get("Gender") == child_feats.get("Gender")
						and verb_feats.get("Number") == child_feats.get("Number")
						):
							obj_agreement = True
					if (
						"obj" in child.deprel # BODILYTAKE
						):
							obj_found = True

				# if (subj_found and obj_agreement) or (subj_found and not obj_found):
				if subj_found and (obj_agreement or not obj_found):
					return "VOL"

		return "OTHER"

	def absolutive_pattern(self, sentence) -> str:
		"""
		Function to verify the absolutive pattern.

		Candidate Pattern:
			- Minimal pairs: Nsubj uses nominative case and ergative case cannot
		"""
		children = defaultdict(list)
		words = list(sentence.iter_words())

		for word in words:
			children[word.head].append(word)

		for word in words:
			# Find candidate verbs
			if "VERB" in word.upos and word.feats and "Aspect=Perf" not in word.feats:
				verb_feats = self.get_feats(word)

				# Handle compound verbs
				verb_id = word.head if "compound" in word.deprel else word.id

				# Find obj markers
				for child in children.get(verb_id, []):
					child_feats = self.get_feats(child)

					if "nsubj" in child.deprel:
						# Reliable marker
						if child_feats and "Nom" in child_feats.get("Case"):
							# Candidate found!
							return "NVOL"
					if "obl" in child.deprel:
						# Verify nvol marker is attached
						for nvol in words:
							if nvol.head == child.id:
								return "NVOL"

		return "OTHER"
