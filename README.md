# Urdu Volitional Detector

Urdu volitional detector with morph-based rule selection using Stanza's neural pipeline. The detector uses the Universal Dependencies (UD) features as cues to classify a sentence as volitional (VOL), non-volition (NVOL), or neither (None).

# Getting Started

```python
pip install requirements.txt
```

You must also download the Stanza's neural pipeline for Urdu.

```python
>>> stanza.download('ur')
```

# Augmented Data
> Credit to Asma for generating the dataset available [here](https://github.com/amyasma/nllb-ur-en-volitional-control/)!

The detector was evaluated on a VOL/NVOL dataset created using a context-free grammar (CFG).

# Evaluation

```
  Accuracy: 0.845 [1804 / 2136]
  Macro-f1: 0.909
  Micro-f1: 0.916

  Volitional Accuracy: 0.82 [901 / 1096]
  Non-volitional Accuracy: 0.87 [903 / 1040]
```

TODO Error analysis, coverage (?)

# Citations & Papers
```
@software{asma2026nllb,
    author = Asma, title = {NLLB EN–UR Volitional Control},
    title = {NLLB UR–EN Volitional Control}
    year = {2026},
    url = {https://github.com/amyasma/nllb-en-ur-volitional-control}
}
```

## Paper Links

- [Spatian, Temporal, and Structural Usages of Urdo ko](https://web.stanford.edu/group/cslipublications/cslipublications/LFG/11/pdfs/lfg06ahmed.pdf)
- [Agentivity: The view from semantic typology](https://www.acsu.buffalo.edu/~jb77/Bohnemeyer_2019_agentivity_Singapore.pdf)
- [System for Grammatical relations in Urdu](https://alt.qcri.org/~ndurrani/pubs/system_grammatical_relations.pdf)