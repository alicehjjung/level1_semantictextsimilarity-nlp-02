# level1_semantictextsimilarity-nlp-02
[ENG](#ENG)   
[한국어](#한국어)

## 🌱Members


|<img src='https://avatars.githubusercontent.com/u/85860941?v=4' height=100 width=100px></img>|<img src='https://avatars.githubusercontent.com/u/28584259?v=4' height=100 width=100px></img>|<img src='https://avatars.githubusercontent.com/u/75467530?v=4' height=100 width=100px></img>|<img src='https://avatars.githubusercontent.com/u/60664644?v=4' height=100 width=100px></img>|<img src='https://avatars.githubusercontent.com/u/101383752?v=4' height=100 width=100px></img>|
| --- | --- | --- | --- | --- |
| [서가은](https://github.com/gaeun0112) | [서보성](https://github.com/Seoboseong) | [오원택](https://github.com/dnjdsxor21) | [이승우](https://github.com/OLAOOT) | [정효정](https://github.com/HYOJUNG08) |

## ENG
# Semantic Text Similarity(STS) project
Semantic Textual Similarity (STS) is a task that measures the semantic similarity between pairs of texts. It is commonly used to evaluate how well a model captures the intimacy between two sentences or implements the semantic representation of a sentence. Through this project, we build an AI model to predict the similarity between two sentences from 0 to 5.

## Getting Started

To use this project, follow these steps:

### Train and Test the Model
```python
# TRAIN
python3 code/train.py

# INFERENCE
python3 code/inference.py
```

## 한국어
## 📎STS (Semantic Textual Similarity)

> 부스트 캠프 AI-Tech 5기 NLP 트랙 Level1 경진대회 프로젝트입니다. Semantic Textual Similarity(STS)는 텍스트 쌍의 의미적 유사도를 측정하는 Task로 모델이 의미상 두 문장의 친밀도를 얼마나 잘 잡아내는지 또는 문장의 의미적 표현을 얼마나 잘 구현하는지 평가하는데 일반적으로 사용됩니다.
> 

### Data (Private)

- 총 데이터 개수 : 10,974 문장 쌍
    - Train(학습) 데이터 개수: 9,324 (85%)
    - Dev(검증) 데이터 개수: 550 (5%)
    - Test(평가) 데이터 개수: 1,100 (10%)
    - Label 점수: 0 ~ 5사이의 실수

### Metric

Test 데이터의 입력에 대한 결과 값들의 피어슨 상관 계수(Pearson Correlation Coefficient)

![Pearson](https://www.simplilearn.com/ice9/free_resources_article_thumb/Pearson_Correlation_2.png)

### 📂Structure

```python
root/
|
|-- code/
|   |-- train.py
|   |-- inference.py
|
|-- module/
|   |-- preprocessing.py
|   |-- augmentation.py
|   |-- model_params.py
|   |-- rdrop.py
|   |-- seed.py
|   |-- translation.py
|   |-- visualization.py
|   |-- memo.py
|   |-- sweep.yaml

```

## ✔️Project


### Preprocessing

- Sentence Refinement
    - 특수문자 제거(RegExp)
    - [한글 맞춤법검사](https://github.com/ssut/py-hanspell)
- Data Augmentation
    - Back Translation
    - [Easy Data Augmentation](https://github.com/toriving/KoEDA)
    - Token masking & Changing sequence order

### Modeling

- [R-Drop](https://github.com/dropreg/R-Drop)
- K-fold
- Model Parameter Freezing
- Label Smoothing
- Add modules to Pre-Trained Model
- [Sentence Bert](https://www.sbert.net)
- Dropout & L2 Regularization
- Model Ensemble

### Wrap-Up Report

- [살펴보기](https://github.com/boostcampaitech5/level1_semantictextsimilarity-nlp-02/blob/main/wrap_up_report/%EB%AC%B8%EC%9E%A5%20%EA%B0%84%20%EC%9C%A0%EC%82%AC%EB%8F%84%20%EC%B8%A1%EC%A0%95_NLP_%ED%8C%80%20%EB%A6%AC%ED%8F%AC%ED%8A%B8(02).pdf)

## 🐞Usage

```python
# TRAIN
python3 code/train.py

# INFERENCE
python3 code/inference.py
```
