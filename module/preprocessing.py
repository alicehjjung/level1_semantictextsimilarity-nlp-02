import pandas as pd
# import numpy as np
import re
from hanspell import spell_checker
# from soynlp.normalizer import repeat_normalize
from tqdm.auto import tqdm
# from emoji import *
# from soynlp.normalizer import *

def drop_marks(data, clean_drop=False):
    """특수문자를 제거하고, 반복되는 초성의 개수를 줄입니다.

    Args:
        data (DataFrame): spell_check와 같이 사용할 시 먼저 처리해야 합니다.
        clean_drop (bool, optional):
            True(default): 한글(초성 포함), 영어, 숫자, <PERSON>만 남깁니다.
            False: 괄호나 -, =, / 등의 연결자는 남깁니다. 실험 중입니다.
        
    Returns:
        DataFrame: 변경된 data
    """
    assert 'sentence_1' in data.columns
    assert 'sentence_2' in data.columns
    
    patterns = [
        # ('ㅜ', 'ㅠ'),
        # (r'\^\^', 'ㅎㅎ'),  
        # (r'([ㄱ-ㅎㅏ-ㅣ!\?])\1{2,}', r'\1\1'),   # 중복 문자열을 두번 반복으로 변환(ㅋㅋㅋㅋ-> ㅋㅋ)
        # (r'[^a-zA-Z가-힣ㄱ-ㅎㅏ-ㅣ\d·\-+=%/()\[\]\:\?\!]+', ' '),
        # (r'\s+', ' '), # 이중 공백 제거
        (r'([ㄱ-ㅎㅏ-ㅣ])\1{2,}', r'\1\1'),
        (r'[^a-zA-Z가-힣ㄱ-ㅎㅏ-ㅣ\d]', ' '),
        (r'\s+', ' ') # 이중 공백 제거
    ]
    
    # if clean_drop:
    #     patterns.append((r'[^\w가-힣ㄱ-ㅎㅏ-ㅣ]', ' '))
    # patterns.append(('PERSON', '<PERSON>'))

    for old, new in patterns:
        data['sentence_1'] = data['sentence_1'].apply(lambda s: re.sub(old, new, s).strip())
        data['sentence_2'] = data['sentence_2'].apply(lambda s: re.sub(old, new, s).strip())
    print('drop_mark Done')
    return data

def check_spell(data):
    """맞춤법을 교정합니다.

    Args:
        data (DataFrame): drop_marks와 같이 사용할 시 나중에 처리해야 합니다.

    Returns:
        DataFrame: 변경된 data
    """
    assert 'sentence_1' in data.columns
    assert 'sentence_2' in data.columns
    
    for i in tqdm(range(len(data)), desc='check_spell', leave=True):
        for col in ['sentence_1', 'sentence_2']:
            # print(data.loc[i, col])
            data.loc[i, col] = spell_checker.check(data.loc[i, col]).as_dict()['checked']
    print('check spell Done')
    return data

def remove_punct(dataframe):
    """
        특수문자를 제거하는 함수
        Args:
        data (DataFrame): drop 된 data

        Returns:
        DataFrame: 특수문자가 제거된 data
    """
    pun = re.compile("\W+")
    for idx, item in tqdm(dataframe.iterrows(), desc='removing punctuation', total=len(dataframe)):
        # 특수문자를 지워서 전처리합니다.
        tmp1, tmp2="",""
        tmp1 = pun.sub(" ",item['sentence_1'])
        tmp2 = pun.sub(" ",item['sentence_2'])
        dataframe.loc[idx,'sentence_1'] = tmp1
        dataframe.loc[idx,'sentence_2'] = tmp2
    return dataframe
    
def remove_emoji(dataframe):
    """
        이모지를 제거하는 함수

        Args:
        data (DataFrame): 특수문자가 제거된 data

        Returns:
        DataFrame: 이모지가 제거된 data
    """
    for idx, item in tqdm(dataframe.iterrows(), desc='removing emoji', total=len(dataframe)):
        # emoji를 지워서 전처리합니다.
        tmp1, tmp2="",""
        tmp1 = emoji.replace_emoji(item['sentence_1'], replace="")
        tmp2 = emoji.replace_emoji(item['sentence_2'], replace="")
        dataframe.loc[idx,'sentence_1'] = tmp1
        dataframe.loc[idx,'sentence_2'] = tmp2
        
    return dataframe

def normalize_repeats(dataframe):
    """
        반복되는 문자 정제 (ex.ㅎㅎㅎ,ㅋㅋㅋ,ㅜㅜㅜ)
        이유는 모르겠으나 작동되지않는 중임 ->3이상 안됨..
        Args:
        data (DataFrame): 특수문자, 이모지가 제거된 data

        Returns:
        DataFrame: 반복되는 문자가 제거된 data
    """
    for idx, item in tqdm(dataframe.iterrows(), desc='removing repeating words', total=len(dataframe)):
        tmp1, tmp2="",""
        print()
        tmp1 = repeat_normalize(item['sentence_1'], num_repeats=1)
        tmp2 = repeat_normalize(item['sentence_2'], num_repeats=1)
        dataframe.loc[idx,'sentence_1'] = tmp1
        dataframe.loc[idx,'sentence_2'] = tmp2
    return  dataframe

def remove_letters(dataframe):
    pattern = '([ㄱ-ㅎㅏ-ㅣ])+' #자음 모음 제거
    for idx, item in tqdm(dataframe.iterrows(), desc='removing vowels and consonants', total=len(dataframe)):
        tmp1, tmp2="",""
        tmp1 = re.sub(pattern=pattern, repl='', string=item['sentence_1'])
        tmp2 = re.sub(pattern=pattern, repl='', string=item['sentence_2'])
        dataframe.loc[idx,'sentence_1'] = tmp1
        dataframe.loc[idx,'sentence_2'] = tmp2
    return  dataframe

def remove_spaces(dataframe):
    pattern = re.compile(r' +') #이중 공백제거
    for idx, item in tqdm(dataframe.iterrows(), desc='removing spaces', total=len(dataframe)):
        tmp1, tmp2="",""
        tmp1 = re.sub(pattern=pattern, repl=' ', string=item['sentence_1'])
        tmp2 = re.sub(pattern=pattern, repl=' ', string=item['sentence_2'])
        dataframe.loc[idx,'sentence_1'] = tmp1
        dataframe.loc[idx,'sentence_2'] = tmp2
    return  dataframe

def only_hangul(dataframe):
    pattern = re.compile(r"[^가-힣\s]") #한글,공백만 남기기(자모 제외)
    for idx, item in tqdm(dataframe.iterrows(), desc='removing except hangul and space', total=len(dataframe)):
        tmp1, tmp2="",""
        tmp1 = re.sub(pattern=pattern, repl='', string=item['sentence_1'])
        tmp2 = re.sub(pattern=pattern, repl='', string=item['sentence_2'])
        dataframe.loc[idx,'sentence_1'] = tmp1
        dataframe.loc[idx,'sentence_2'] = tmp2
    return  dataframe

#def remove_stopwords(dataframe):
 #   for idx, item in tqdm(dataframe.iterrows(), desc='removing except hangul and space', total=len(dataframe)):
  #      tmp1, tmp2="",""
   #     tmp1 = re.sub(pattern=pattern, repl='', string=item['sentence_1'])
    #    tmp2 = re.sub(pattern=pattern, repl='', string=item['sentence_2'])
     #   dataframe.loc[idx,'sentence_1'] = tmp1
      #  dataframe.loc[idx,'sentence_2'] = tmp2
  #  return  dataframe

if __name__ == '__main__':
    ## 맞춤법 csv파일 생성
    train_df = pd.read_csv('./data/train.csv')
    dev_df = pd.read_csv('./data/dev.csv')
    test_df = pd.read_csv('./data/test.csv')

    ## 전처리 한 데이터 생성
    check_spell(drop_marks(train_df)).to_csv('./data/pre_train.csv')
    check_spell(drop_marks(dev_df)).to_csv('./data/pre_dev.csv')
    check_spell(drop_marks(test_df)).to_csv('./data/pre_test.csv')
