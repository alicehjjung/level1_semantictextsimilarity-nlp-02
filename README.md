# level1_semantictextsimilarity-nlp-02

## 🌱Members


| image1 | image2 | image3 | image4 |  |
| --- | --- | --- | --- | --- |
| [서가은](https://github.com/gaeun0112) | [서보성](https://github.com/Seoboseong) | [오원택](https://github.com/dnjdsxor21) | [이승우](https://github.com/OLAOOT) | [정효정](https://github.com/HYOJUNG08) |

## 📎STS (Semantic Textual Similarity)

> 부스트 캠프 AI-Tech 5기 NLP 트랙 Level1 경진대회 프로젝트입니다. Semanic Textual Similarity(STS)는 텍스트 쌍의 의미적 유사도를 측정하는 Task로 모델이 의미상 두 문장의 친밀도를 얼마나 잘 잡아내는지 또는 문장의 의미적 표현을 얼마나 잘 구현하는지 평가하는데 일반적으로 사용됩니다.
> 

### Data (Private)

- 총 데이터 개수 : 10,974 문장 쌍
    - Train(학습) 데이터 개수: 9,324 (85%)
    - Dev(검증) 데이터 개수: 550 (5%)
    - Test(평가) 데이터 개수: 1,100 (10%)
    - Label 점수: 0 ~ 5사이의 실수

### Metric

Test 데이터의 입력에 대한 결과 값들의 피어슨 상관 계수(Pearson Correlation Coefficient)

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/c02b4c09-bd52-4ead-8e91-9b9073fe5cf1/Untitled.png)

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

## 🐞Usage

```python
# TRAIN
python3 code/train.py

# INFERENCE
python3 code/inference.py
```
