# Sentiment Analysis Champion–Challenger Benchmark with Hugging Face and MLflow

## 1. Overview: Task and Models

**Task:** Sentiment Analysis  
**Dataset:** Stanford Sentiment Treebank (SST-2) validation split  
**Evaluation sample:** 100 instances — 50 Positive and 50 Negative  
**Random seed:** 42

### Model A — Initial Champion
- **Hugging Face ID:** `distilbert-base-uncased-finetuned-sst-2-english`
- **Role:** Initial Champion
- **Architecture:** DistilBERT fine-tuned on SST-2 for sentiment classification

### Model B — Challenger
- **Hugging Face ID:** `siebert/sentiment-roberta-large-english`
- **Role:** Challenger
- **Architecture:** RoBERTa-large fine-tuned for English sentiment classification

To guarantee a fair comparison, both models were tested on the **exact same 100 inputs in the identical sequence**.

---

## 2. Running the Workflow

First, launch the local MLflow tracking server:

```bash
mlflow server --host 127.0.0.1 --port 5000
```

Next, execute the evaluation script from the project root directory:

```bash
python evaluate.py
```

Register both evaluated models in MLflow and set their Champion and Challenger aliases:

```bash
python register_champion.py
```

Finally, generate predictions on unseen data by loading the model assigned to the Champion alias:

```bash
python predict_from_mlflow.py
```

The resulting predictions are written to:

```text
outputs/new_data_predictions.csv
```

The active Champion model is fetched from the MLflow Model Registry via:

```text
models:/HF_Sentiment_Champion_Challenger@champion
```

---

## 3. Champion Selection Rule

The promotion criteria were established **prior to executing the final benchmark**.

- **Primary metric:** Macro-F1
- **Optimisation goal:** Higher is better
- **Minimum improvement threshold:** 0.01
- **Tie-breaker:** Lower average inference latency
- **Selection logic:** The Challenger replaces the current Champion only if it outperforms the Champion's Macro-F1 by at least 0.01; otherwise, the existing Champion holds its title. If the scores are tied within the decision margin, the model with lower latency wins.

At the start of the experiment, Model A served as the Champion and Model B was tested as the Challenger.

---

## 4. Benchmark Results

| Metric | Model A — DistilBERT | Model B — RoBERTa | Difference (B − A) |
|---|---:|---:|---:|
| Accuracy | 0.9200 | 0.9400 | +0.0200 |
| Macro-F1 | 0.9200 | 0.9399 | +0.0199 |
| Avg. Latency (ms/sample) | 113.42 | 691.70 | +578.28 |
| Total Inference Time (s) | 11.3417 | 69.1695 | +57.8278 |

### Outcome

**Selected Champion:** Model B — `siebert/sentiment-roberta-large-english`

Model B reached a Macro-F1 score of **0.9399**, outperforming Model A's **0.9200**. Because this **0.0199** gain exceeds the required **0.01** minimum improvement threshold, the Challenger meets the selection rule and takes over from Model A.

---

## 5. Decision Rationale

**Model B (`siebert/sentiment-roberta-large-english`)** has been selected as the final Champion. This outcome rests primarily on Macro-F1, the metric agreed upon before running the benchmark. Model B recorded a Macro-F1 of **0.9399** against **0.9200** for Model A (`distilbert-base-uncased-finetuned-sst-2-english`), delivering an improvement of **0.0199**—comfortably above the mandatory **0.01** margin. Model B also led in overall accuracy, scoring **0.9400** versus **0.9200** for Model A.

This gain in predictive performance comes with a steep latency cost. Model B averaged roughly **691.70 ms per sample**, compared with approximately **113.42 ms per sample** for Model A, making Model B around 6 times slower on the CPU setup used for testing. Even with this overhead, the Macro-F1 margin was wide enough to satisfy the pre-established Champion rule.

Accordingly, Model B is promoted to **Champion** and Model A shifts to **Challenger**. Both models are logged in the MLflow Model Registry, with the `champion` alias assigned to Model B and the `challenger` alias assigned to Model A. Following registration, the Champion was retrieved from the registry to run inference on a separate batch of unseen data.

---

## 6. Limitations and Declaration

### Limitations

Because this benchmark relies on a modest sample of **100 SST-2 validation examples**, the observed scores may not fully generalise to all real-world sentiment inputs. Running the evaluation on a CPU also means latency figures are heavily influenced by the specific hardware used. While Model B delivers stronger classification metrics, it requires noticeably more compute time and has a larger footprint.

Strict separation was maintained between benchmark data and inference inputs—none of the 5 records in `data/new_data.csv` were included in computing the evaluation metrics.

### AI Assistant Declaration

AI assistants supported the development process through coding guidance, troubleshooting, documentation drafting, and workflow advice. The student independently reviewed and verified the final codebase, MLflow experiment tracking, model evaluation, Champion–Challenger decision, registry configuration, and inference pipeline.

This project relies on MLflow for experiment tracking and model lifecycle management, alongside Hugging Face Transformers for sentiment classification.
