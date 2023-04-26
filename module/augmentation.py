from tqdm.auto import tqdm
import random

def augment(df, tokenizer, targets, max_length):
    """
    기존 데이터에 증강된 데이터를 더해주어 tokenize하는 함수

    Args:
        df (DataFrame): 입력 데이터
        tokenizer (BertTokenizerFast): pretrained tokenizer
        max_length (int): tokenizer의 최대 token 길이
    
    Returns:
        증강된 데이터를 tokenize한 결과값 inputs, attention masks, token type ids
    """
    text_columns = ['sentence_1', 'sentence_2']
    data = []
    new_targets = []
    token_type_ids = []
    attention_mask = []
    SEP_num = tokenizer.convert_tokens_to_ids("[SEP]")
    MASK_num = tokenizer.convert_tokens_to_ids("[MASK]")
    PAD_num = tokenizer.convert_tokens_to_ids("[PAD]")

    for idx, item in tqdm(df.iterrows(), desc='tokenizing', total=len(df)):
        # text1 : 일반, outputs1 : text1 출력
        text1_1, text1_2 = [item[text_column] for text_column in text_columns]
        outputs1 = tokenizer(text1_1, text1_2, add_special_tokens=True, padding='max_length', truncation=True, return_token_type_ids=True, max_length=max_length)

        sentence1_end_idx = outputs1['input_ids'].index(SEP_num)
        if 1 in outputs1['input_ids']:
            start_pad_idx = outputs1['input_ids'].index(PAD_num)
        else:
            start_pad_idx = len(outputs1['input_ids'])

        # text2 : 입력 순서 변경, outputs2 : text2 출력
        text2_1, text2_2 = [item[text_column] for text_column in text_columns[::-1]]
        outputs2 = tokenizer(text2_1, text2_2, add_special_tokens=True, padding='max_length', truncation=True, return_token_type_ids=True, max_length=max_length)

        # text3 : masking, outputs3 : text3 출력
        text3_1, text3_2 = [item[text_column] for text_column in text_columns]
        outputs3 = tokenizer(text3_1, text3_2, add_special_tokens=True, padding='max_length', truncation=True, return_token_type_ids=True, max_length=max_length)

        if sentence1_end_idx != 2 and start_pad_idx - sentence1_end_idx != 3:
            masking_idx = random.randrange(1, start_pad_idx-2)

            if sentence1_end_idx <= masking_idx:
                masking_idx += 1

            outputs3['input_ids'][masking_idx] = MASK_num

            data.append(outputs3['input_ids'])
            token_type_ids.append(outputs3['token_type_ids'])
            attention_mask.append(outputs3['attention_mask'])
            new_targets.append(targets[idx])

        # text4 : Document Rotation, outputs4 : text4 출력
        text4_1, text4_2 = [item[text_column] for text_column in text_columns]
        outputs4 = tokenizer(text4_1, text4_2, add_special_tokens=True, padding='max_length', truncation=True, return_token_type_ids=True, max_length=max_length)

        if sentence1_end_idx != 2 and start_pad_idx - sentence1_end_idx != 3:
            sent_choice = random.randrange(1, 3)

            if sent_choice == 1:
                rotate1_idx = random.randrange(1, sentence1_end_idx)
                rotate2_idx = random.randrange(1, sentence1_end_idx-1)
            else:
                rotate1_idx = random.randrange(sentence1_end_idx+1, start_pad_idx-1)
                rotate2_idx = random.randrange(sentence1_end_idx+1, start_pad_idx-2)

            if rotate1_idx == rotate2_idx:
                rotate2_idx += 1

            outputs4['input_ids'][rotate1_idx], outputs4['input_ids'][rotate2_idx] = outputs4['input_ids'][rotate2_idx], outputs4['input_ids'][rotate1_idx]

            data.append(outputs4['input_ids'])
            token_type_ids.append(outputs4['token_type_ids'])
            attention_mask.append(outputs4['attention_mask'])
            new_targets.append(targets[idx])

        data.extend([outputs1['input_ids'], outputs2['input_ids']])
        token_type_ids.extend([outputs1['token_type_ids'], outputs2['token_type_ids']])
        attention_mask.extend([outputs1['attention_mask'], outputs2['attention_mask']])
        new_targets.extend([targets[idx], targets[idx]])

    
    return data, attention_mask, token_type_ids, new_targets