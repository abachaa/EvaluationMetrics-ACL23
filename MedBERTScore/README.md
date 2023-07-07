# MedBERTScore


This folder contains information on to how download and run MedBERTScore.

MedBERTScore uses the original BERTScore evaluation pipeline (https://arxiv.org/abs/1904.09675).

The MedBERTScore model uses a  novel weighted policy which provides a higher weight to words with medical meaning

## Setup
- To run the MedBERTScore model,  please first follow the setup instruction found here:
```
https://github.com/Tiiiger/bert_score
```
- Please also install the MedCat tool found here:
```
https://github.com/CogStack/MedCAT
```

## MedBERTScore
-  To run the MedBERTScore firstly we need to create the weights for each word/token in the documents: 
```
python deberta_create_medical_weight.py \
  --candidate=<candidate-file> \
  --refecrence=<reference-file> \
  --output=output-path>  \
  --file=<medicalcat-file-path>
```

-  We can run the MedBERTScore model:
```
python score_cli.py -s \
  -c=<candidate-file> \
  -r=<reference-file> \
  -wc=<candidate-weight-file> \
  -wr=<reference-weight-file> \
  --output=output-path> \
  --lang en
```
