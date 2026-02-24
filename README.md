# Text Classification: TF-IDF vs BiLSTM vs Transformer (AG News)

## Project Overview
比較三種文本分類方法在 **AG News** 資料集上的表現：

- **TF-IDF + Logistic Regression**
- **BiLSTM**
- **Transformer (DistilBERT)**

目標是分析不同模型的效能差異、錯誤型態與模型行為

---

## Models

### TF-IDF + Logistic Regression
- Linear model
- Bag-of-Words representation
- Fast and strong baseline
- No contextual understanding

### BiLSTM
- Sequential modeling via RNN
- Captures word order and local context
- No pretraining

### DistilBERT (Transformer)
- Pretrained contextual embeddings
- Self-attention based
- Strong semantic understanding

---

## Dataset
**AG News — 4-class news classification**

**Classes**
- World
- Sports
- Business
- Sci/Tech

**Samples**
- Train: 120,000
- Test: 7,600

---

## Results

| Model | Accuracy | Macro F1 | Weighted F1 |
|------|----------:|---------:|------------:|
| TF-IDF | 0.9260 | 0.9259 | 0.9259 |
| BiLSTM | 0.9148 | 0.9147 | 0.9147 |
| DistilBERT | **0.9478** | **0.9476** | **0.9476** |

---

## Error Analysis
(From normalized confusion matrices)

- Transformer 顯著優於 TF-IDF 與 BiLSTM  
- 最大改善出現在 **Business** 類別，該類別與 **Sci/Tech** 存在 semantic overlap  
- contextual + pretraining 使模型能更精準區分細微領域差異

---

## Project Structure

```text
src/
  data.py            # load/split/label handling
  metrics.py         # metrics + confusion matrix
  utils.py           # run_dir

  tfidf/
    model.py
    train.py

  bilstm/
    data.py
    model.py
    train.py

  transformer/
    model.py
    train.py
```

---

## Key Takeaways

- **Pretrained Transformers** significantly outperform baseline 與 RNN 模型

- **TF-IDF 仍是強而高效的 baseline**，特別是在 short text classification

- 最困難的分類為 **Business vs Sci/Tech**，主要因為兩者存在 **semantic overlap**

- **Contextual representation** 對於區分跨領域語意差異具有關鍵作用