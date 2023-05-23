# EvaluationMetrics-ACL23

This repository will contain the source code and annotations for the paper: 
- An Investigation of Evaluation Metrics for Automated Medical Note Generation. Asma Ben Abacha, Wen-wai Yim, George Michalopoulos and Thomas Lin. ACL Findings 2023. 

```
@inproceedings{eval-2023,
  author = {Asma {Ben Abacha} and 
            Wen{-}wai Yim and
            George Michalopoulos and
            Thomas Lin},
  title = {An Investigation of Evaluation Metrics for Automated Medical Note Generation},
   booktitle = "Findings of the Association for Computational Linguistics: ACL 2023",
   publisher = "Association for Computational Linguistics",
  year = {2023},
  abstract = {Recent studies on automatic note generation have shown that doctors can save significant amounts of time when using automatic clinical note generation (Knoll et al., 2022). Summarization models have been used for this task to generate clinical notes as summaries of doctor-patient conversations (Krishna et al., 2021; Cai et al., 2022). However, assessing which model would best serve clinicians in their daily practice is still a challenging task due to the large set of possible correct summaries, and the potential limitations of automatic evaluation metrics. In this paper we study evaluation methods and metrics for the automatic generation of clinical notes from medical conversation. In particular, we propose new task-specific metrics and we compare them to SOTA evaluation metrics in text summarization and generation, including: (i) knowledge-graph embedding-based metrics, (ii) customized model-based metrics, (iii) domain-adapted/fine-tuned metrics, and (iv) ensemble metrics. To study the correlation between the automatic metrics and manual judgments, we evaluate automatic notes/summaries by comparing the system and reference facts and computing the factual correctness, and the hallucination and omission rates for critical medical facts. This study relied on seven datasets manually annotated by domain experts. Our experiments show that automatic evaluation metrics can have substantially different behaviors on different types of clinical notes datasets. However, the results highlight one stable subset of metrics as the most correlated with human judgments with a relevant aggregation of different evaluation criteria.}
}
```


## Contact

    -  Asma Ben abacha (abenabacha at microsoft dot com)
     - Wen-wai Yim (yimwenwai at microsoft dot com)
     - Goerge Michalopoulos (georgemi at microsoft dot com)

----

Release Date: June 30, 2023. 

----
