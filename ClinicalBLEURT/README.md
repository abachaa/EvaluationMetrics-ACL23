# ClinicalBLEURT

This folder contains information on to how download and run ClinicalBLEURT.

ClinicalBEURT uses the original BLEURT evaluation model (https://arxiv.org/abs/2004.04696) as it's base.
The BLEURT-20 model is fine-tuned on a propriety clinical note dataset for one epoch.


To run the ClinicalBLEURT model, please first install the bleurt code-base found here:
```
https://github.com/google-research/bleurt
```

The fine-tuned model can be downloaded from here:
```
https://drive.google.com/file/d/1pjd8TxqGeYVdCWZHdXow8pDXpwRebwFY/view?usp=sharing
```

To run the model use the command:
```
python -m bleurt.score_files \
  -candidate_file=<candidate-file> \
  -reference_file=<reference-file> \
  -bleurt_checkpoint=ClinicalBLEURT/ \
  -scores_file=<output-file>
```
