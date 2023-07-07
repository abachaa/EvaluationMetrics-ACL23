# MedBARTScore


This folder contains information on to how download and run MedBARTScore.

MedBARTScore uses the original BARTScore evaluation pipeline (https://arxiv.org/abs/2106.11520).

The MedBARTScore model uses a  novel weighted policy which provides a higher weight to words with medical meaning

## Setup
- To run the MedBARTScore model,  please first follow the setup instruction found here:
```
https://github.com/neulab/BARTScore
```
- Please also install the MedCat tool found here:
```
https://github.com/CogStack/MedCAT
```

## MedBARTScore
-  To run the MedBARTScore firstly we need to create the weights for each word/token in the document: 
```
python bartscore_create_medical_weight.py \
  --candidate=<candidate-file> \
  --refecrence=<reference-file> \
  --output=output-path>  \
  --file=<medicalcat-file-path>
```

-  We need to create the final input file that the MedBARTScore model wil use:
```
python create_pkl.py \
  --candidate=<candidate-file> \
  --refecrence=<reference-file> \
  --output=output-path>  \
  --source=<source-file>
```

-  We can run the MedBARTScore model:
```
python score.py \
  --file=<dataset-file> \
  --bart_path=<bart-model-path> \
  --output=output-path> \
  --weight \
  --bart_score  
```

