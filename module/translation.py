import googletrans
from tqdm.auto import tqdm
#from transformers import MarianMTModel, MarianTokenizer
import torch
#from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
from dooly import Dooly
import pandas as pd
from preprocessing import *

'''
translator = googletrans.Translator()

def kor_eng(df):
    """
        Google Trans 이용한 한영 번역 함수
        Args:
        df (DataFrame) : 한국어에서 영어로 번역할 data

        Returns:
        DataFrame: 번역이 완료된 data
    """ 
    for idx, item in tqdm(df.iterrows(), desc='Korean to English', total=len(df)):
            tmp1, tmp2="",""
            tmp1 = translator.translate(item['sentence_1'],dest='en')
            tmp2 = translator.translate(item['sentence_2'],dest='en')
            #df.loc[idx,'sentence_1'] = tmp1
            #df.loc[idx,'sentence_2'] = tmp2
            df.at[idx, 'sentence_1'] = tmp1.text
            df.at[idx, 'sentence_2'] = tmp2.text
    return df

def eng_kor(df):
    """
        Google Trans 이용한 영한 번역 함수
        Args:
        df (DataFrame) : 영어에서 한국어로 번역할 data

        Returns:
        DataFrame: 번역이 완료된 data
    """
    for idx, item in tqdm(df.iterrows(), desc='English to Korean', total=len(df)):
            tmp1, tmp2="",""
            tmp1 = translator.translate(item['sentence_1'],dest='kor')
            tmp2 = translator.translate(item['sentence_2'],dest='kor')
            #df.loc[idx,'sentence_1'] = tmp1
            #df.loc[idx,'sentence_2'] = tmp2
            df.at[idx, 'sentence_1'] = tmp1.text
            df.at[idx, 'sentence_2'] = tmp2.text
    return df
'''
'''
def kor_eng(df):
    """
        Helsinki-NLP/opus-mt-ko-en 모델을 이용한 한영 번역 함수
        Args:
        df (DataFrame) : 한국어에서 영어로 번역할 data

        Returns:
        DataFrame: 번역이 완료된 data
    """
    #model_name = 'Helsinki-NLP/opus-mt-ko-en'

    # 모델
    model_name = 'Helsinki-NLP/opus-mt-tc-big-ko-en'
    model = MarianMTModel.from_pretrained(model_name)
    tokenizer = MarianTokenizer.from_pretrained(model_name) 

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # 배차 사이즈
    batch_size = 32

    for i in tqdm(range(0, len(df), batch_size), desc='Korean to English', total=len(df)//batch_size):
        batch = df[i:i+batch_size]

        sentence_1_batch = batch['sentence_1'].tolist()
        sentence_2_batch = batch['sentence_2'].tolist()

        # Tokenize
        input_texts = [f'{s} </s>' for s in sentence_1_batch + sentence_2_batch]
        input_ids = tokenizer.batch_encode_plus(input_texts, return_tensors='pt', padding=True, truncation=True)

        input_ids = {k: v.to(device) for k, v in input_ids.items()}

        with torch.no_grad():
            outputs = model.generate(input_ids['input_ids'], max_length=128, num_beams=4, no_repeat_ngram_size=3, early_stopping=True)

        translated_sentences = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        # Dataframe을 번역된 문장으로 업데이트 한다.
        num_sentences = len(sentence_1_batch)
        df.loc[i:i+num_sentences-1, 'sentence_1'] = translated_sentences[:num_sentences]
        df.loc[i:i+num_sentences-1, 'sentence_2'] = translated_sentences[num_sentences:]

    return df

def eng_kor(df):
    """
        Helsinki-NLP/opus-mt-ko-en 모델을 이용한 한영 번역 함수
        Args:
        df (DataFrame) : 영어에서 한국어로 번역할 data

        Returns:
        DataFrame: 번역이 완료된 data
    """
    # 모델
    model_name = 'Helsinki-NLP/opus-mt-tc-big-en-ko'
    model = MarianMTModel.from_pretrained(model_name)
    tokenizer = MarianTokenizer.from_pretrained(model_name)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # 배치 사이즈
    batch_size = 32

    for i in tqdm(range(0, len(df), batch_size), desc='English to Korean', total=len(df)//batch_size):
        batch = df[i:i+batch_size]

        sentence_1_batch = batch['sentence_1'].tolist()
        sentence_2_batch = batch['sentence_2'].tolist()

        # Tokenize
        input_texts = [f'{s} </s>' for s in sentence_1_batch + sentence_2_batch]
        input_ids = tokenizer.batch_encode_plus(input_texts, return_tensors='pt', padding=True, truncation=True)

        input_ids = {k: v.to(device) for k, v in input_ids.items()}

        with torch.no_grad():
            outputs = model.generate(input_ids['input_ids'], max_length=128, num_beams=4, no_repeat_ngram_size=3, early_stopping=True)

        translated_sentences = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        # Dataframe을 번역된 문장으로 업데이트 한다.
        num_sentences = len(sentence_1_batch)
        df.loc[i:i+num_sentences-1, 'sentence_1'] = translated_sentences[:num_sentences]
        df.loc[i:i+num_sentences-1, 'sentence_2'] = translated_sentences[num_sentences:]
    return df
'''
'''
# facebook/mbart-large-50-many-to-many-mmt
model_name = "facebook/mbart-large-50-many-to-many-mmt"
model = MBartForConditionalGeneration.from_pretrained(model_name)
tokenizer = MBart50TokenizerFast.from_pretrained(model_name)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
    

def kor_eng(df):
    """
        facebook/mbart-large-50-many-to-many-mmt 모델을 이용한 한영 번역 함수
        Args:
        df (DataFrame) : 한국어에서 영어로 번역할 data

        Returns:
        DataFrame: 번역이 완료된 data
    """
    # 배치사이즈
    batch_size = 32

    # translated sentences를 저장할 리스트
    translated_1 = []
    translated_2 = []

    # source language
    tokenizer.src_lang = "ko_KR"

    # input DataFrame을 작은 배치로 나눈다.
    num_batches = len(df) // batch_size + 1
    for i in tqdm(range(num_batches), desc='Korean to English', total=num_batches):
        batch_df = df[i * batch_size:(i + 1) * batch_size]

        #특정 source만 번역한다.
        batch_df = batch_df[(batch_df['source']=='petition-rtt') | (batch_df['source']=='petition-sampled')]
        if not batch_df.empty:                          
            encoded_hi_1 = tokenizer(batch_df['sentence_1'].tolist(), return_tensors="pt", padding=True, truncation=True).to(device)
            encoded_hi_2 = tokenizer(batch_df['sentence_2'].tolist(), return_tensors="pt", padding=True, truncation=True).to(device)

            with torch.no_grad():
                generated_tokens_1 = model.generate(
                    **encoded_hi_1,
                    forced_bos_token_id=tokenizer.lang_code_to_id["en_XX"],
                    num_return_sequences=1  # Set to 1 for single translation
                )
                generated_tokens_2 = model.generate(
                    **encoded_hi_2,
                    forced_bos_token_id=tokenizer.lang_code_to_id["en_XX"],
                    num_return_sequences=1  # Set to 1 for single translation
                )

            # 디코드하고 번역된 문장을 저장한다.
            translated_1.extend(tokenizer.batch_decode(generated_tokens_1, skip_special_tokens=True))
            translated_2.extend(tokenizer.batch_decode(generated_tokens_2, skip_special_tokens=True))
        else:
            continue

    # DataFrame을 번역된 문장으로 업데이트한다.
    df.loc[(df['source'] == 'petition-rtt') | (df['source'] == 'petition-sampled'), 'sentence_1'] = translated_1
    df.loc[(df['source'] == 'petition-rtt') | (df['source'] == 'petition-sampled'), 'sentence_2'] = translated_2

    return df

def eng_kor(df):
    """
        facebook/mbart-large-50-many-to-many-mmt 모델을 이용한 영한 번역 함수
        Args:
        df (DataFrame) : 영어에서 한국어로 번역할 data

        Returns:
        DataFrame: 번역이 완료된 data
    """
    # 배치 사이즈
    batch_size = 32

    # translated sentences를 저장할 리스트
    translated_1 = []
    translated_2 = []

    # source language
    tokenizer.src_lang = "en_XX"

    # input DataFrame을 작은 배치로 나눈다.
    num_batches = len(df) // batch_size + 1
    for i in tqdm(range(num_batches), desc='English to Korean', total=num_batches):
        batch_df = df[i * batch_size:(i + 1) * batch_size]

        #특정 source만 번역한다.
        batch_df = batch_df[(batch_df['source']=='petition-rtt') | (batch_df['source']=='petition-sampled')]
        if not batch_df.empty: 
            encoded_hi_1 = tokenizer(batch_df['sentence_1'].tolist(), return_tensors="pt", padding=True, truncation=True).to(device)
            encoded_hi_2 = tokenizer(batch_df['sentence_2'].tolist(), return_tensors="pt", padding=True, truncation=True).to(device)

            with torch.no_grad():
                generated_tokens_1 = model.generate(
                    **encoded_hi_1,
                    forced_bos_token_id=tokenizer.lang_code_to_id["ko_KR"],
                    num_return_sequences=1  # Set to 1 for single translation
                )
                generated_tokens_2 = model.generate(
                    **encoded_hi_2,
                    forced_bos_token_id=tokenizer.lang_code_to_id["ko_KR"],
                    num_return_sequences=1  # Set to 1 for single translation
                )

            # 디코드하고 번역된 문장을 저장한다.
            translated_1.extend(tokenizer.batch_decode(generated_tokens_1, skip_special_tokens=True))
            translated_2.extend(tokenizer.batch_decode(generated_tokens_2, skip_special_tokens=True))
        else:
            continue

    # DataFrame을 번역된 문장으로 업데이트한다.
    df.loc[(df['source'] == 'petition-rtt') | (df['source'] == 'petition-sampled'), 'sentence_1'] = translated_1
    df.loc[(df['source'] == 'petition-rtt') | (df['source'] == 'petition-sampled'), 'sentence_2'] = translated_2

    return df
'''

def kor_eng(df):
    """
        Dooly 라이브러리를 이용한 한영 번역 함수
        Args:
        df (DataFrame) : 한국어에서 영어로 번역할 data

        Returns:
        DataFrame: 번역이 완료된 data
    """
    # 배치사이즈 설정
    batch_size = 512
    num_batches = len(df)//batch_size + 1

    for i in tqdm(range(0, num_batches ), desc='Korean to English', total=num_batches ):
        start_idx = i * batch_size
        end_idx = min(start_idx+batch_size,len(df))
 
        sentence_1 = df.iloc[start_idx:end_idx]['sentence_1'].tolist()
        sentence_2 = df.iloc[start_idx:end_idx]['sentence_2'].tolist()
        
        # 각 배치를 번역한다.
        translations_1 = mt(sentence_1,src_langs="ko",tgt_langs="en")
        translations_2 = mt(sentence_2,src_langs="ko",tgt_langs="en")

        # Dataframe을 번역된 문장으로 업데이트한다.
        df.iloc[start_idx:end_idx, df.columns.get_loc('sentence_1')] = translations_1
        df.iloc[start_idx:end_idx, df.columns.get_loc('sentence_2')] = translations_2

    return df

def eng_kor(df):
    """
        Dooly 라이브러리를 이용한 영한 번역 함수
        Args:
        df (DataFrame) : 영어에서 한국어로 번역할 data

        Returns:
        DataFrame: 번역이 완료된 data
    """
    # 배치사이즈 설정
    batch_size = 512
    num_batches = len(df)//batch_size + 1

    for i in tqdm(range(0,num_batches), desc='English to Korean', total=num_batches ):
        start_idx = i * batch_size
        end_idx = min(start_idx+batch_size,len(df))

        sentence_1 = df.iloc[start_idx:end_idx]['sentence_1'].tolist()
        sentence_2 = df.iloc[start_idx:end_idx]['sentence_2'].tolist()
       
        # 각 배치를 번역한다.
        translations_1 = mt(sentence_1,src_langs="en",tgt_langs="ko")
        translations_2 = mt(sentence_2,src_langs="en",tgt_langs="ko")

        # Dataframe을 번역된 문장으로 업데이트한다.
        df.iloc[start_idx:end_idx, df.columns.get_loc('sentence_1')] = translations_1
        df.iloc[start_idx:end_idx, df.columns.get_loc('sentence_2')] = translations_2

    return df

def pre_df(df):
    """
        한영 번역, 영한 번역 함수 실행
        Args:
        df (DataFrame) : 번역할 data

        Returns:
        DataFrame: 역번역이 완료된 data
    """
    df = kor_eng(df)
    df = eng_kor(df)

    return df

if __name__ == '__main__':
    mt = Dooly(task="translation",lang="multi")
    train_data = pd.read_csv('./data/train.csv')
    train_translate = pre_df(train_data)
    train_translate.to_csv("./data/train_translate.csv")