from tqdm.auto import tqdm
import random

def r_augment(df, tokenizer, targets, max_length, do_augmentation):
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
        ##################################################
        # text1 : 일반, outputs1 : text1 출력
        text1_1, text1_2 = [item[text_column] for text_column in text_columns]
        normal_outputs1 = tokenizer(text1_1, text1_2, add_special_tokens=True, padding='max_length', truncation=True, return_token_type_ids=True, max_length=max_length)
        reversed_outputs1 = tokenizer(text1_2, text1_1, add_special_tokens=True, padding='max_length', truncation=True, return_token_type_ids=True, max_length=max_length)
        outputs1_input_ids = (normal_outputs1['input_ids'], reversed_outputs1['input_ids'])
        outputs1_token_type_ids = (normal_outputs1['token_type_ids'], reversed_outputs1['token_type_ids'])
        outputs1_attention_mask = (normal_outputs1['attention_mask'], reversed_outputs1['attention_mask'])
        ##################################################

        if do_augmentation:
            ##################################################
            # condition
            if PAD_num in normal_outputs1['input_ids']:
                start_pad_idx = normal_outputs1['input_ids'].index(PAD_num)
            else:
                start_pad_idx = len(normal_outputs1['input_ids'])
            ##################################################

            ##################################################
            # text3 : masking, outputs3 : text3 출력
            text3_1, text3_2 = [item[text_column] for text_column in text_columns]
            normal_outputs3 = tokenizer(text3_1, text3_2, add_special_tokens=True, padding='max_length', truncation=True, return_token_type_ids=True, max_length=max_length)
            reversed_outputs3 = tokenizer(text3_2, text3_1, add_special_tokens=True, padding='max_length', truncation=True, return_token_type_ids=True, max_length=max_length)

            # masking normal_outputs3
            normal_rint = random.randrange(1, start_pad_idx-2)
            normal3_sep_idx = normal_outputs3['input_ids'].index(SEP_num)

            if normal3_sep_idx == normal_rint: normal_rint += 1

            normal_outputs3['input_ids'][normal_rint] = MASK_num

            # masking reversed_outputs3
            reversed_rint = random.randrange(1, start_pad_idx-2)
            reversed3_sep_idx = reversed_outputs3['input_ids'].index(SEP_num)

            if reversed3_sep_idx == reversed_rint: reversed_rint += 1

            reversed_outputs3['input_ids'][reversed_rint] = MASK_num

            outputs3_input_ids = (normal_outputs3['input_ids'], reversed_outputs3['input_ids'])
            outputs3_token_type_ids = (normal_outputs3['token_type_ids'], reversed_outputs3['token_type_ids'])
            outputs3_attention_mask = (normal_outputs3['attention_mask'], reversed_outputs3['attention_mask'])

            data.append(outputs3_input_ids)
            token_type_ids.append(outputs3_token_type_ids)
            attention_mask.append(outputs3_attention_mask)
            new_targets.append(targets[idx])
            ##################################################

            ##################################################
            # text4 : Document Rotation, outputs4 : text4 출력
            text4_1, text4_2 = [item[text_column] for text_column in text_columns]
            normal_outputs4 = tokenizer(text4_1, text4_2, add_special_tokens=True, padding='max_length', truncation=True, return_token_type_ids=True, max_length=max_length)
            reversed_outputs4 = tokenizer(text4_2, text4_1, add_special_tokens=True, padding='max_length', truncation=True, return_token_type_ids=True, max_length=max_length)

            if 6 <= start_pad_idx:
                normal4_sep_idx = normal_outputs4['input_ids'].index(SEP_num)

                if normal4_sep_idx == 2:
                    rotate1_idx = random.randrange(normal4_sep_idx+1, start_pad_idx-1)
                    rotate2_idx = random.randrange(normal4_sep_idx+1, start_pad_idx-2)

                    if rotate1_idx == rotate2_idx:
                        rotate2_idx += 1

                elif normal4_sep_idx == start_pad_idx - 3:
                    rotate1_idx = random.randrange(1, normal4_sep_idx)
                    rotate2_idx = random.randrange(1, normal4_sep_idx-1)

                    if rotate1_idx == rotate2_idx:
                        rotate2_idx += 1

                else:
                    sent_choice = random.randrange(1, 3)

                    if sent_choice == 1:
                        rotate1_idx = random.randrange(1, normal4_sep_idx)
                        rotate2_idx = random.randrange(1, normal4_sep_idx-1)
                    else:
                        rotate1_idx = random.randrange(normal4_sep_idx+1, start_pad_idx-1)
                        rotate2_idx = random.randrange(normal4_sep_idx+1, start_pad_idx-2)

                    if rotate1_idx == rotate2_idx:
                        rotate2_idx += 1

                normal_outputs4['input_ids'][rotate1_idx], normal_outputs4['input_ids'][rotate2_idx] = normal_outputs4['input_ids'][rotate2_idx], normal_outputs4['input_ids'][rotate1_idx]

                reversed4_sep_idx = reversed_outputs4['input_ids'].index(SEP_num)

                if reversed4_sep_idx == 2:
                    rotate1_idx = random.randrange(reversed4_sep_idx+1, start_pad_idx-1)
                    rotate2_idx = random.randrange(reversed4_sep_idx+1, start_pad_idx-2)

                    if rotate1_idx == rotate2_idx:
                        rotate2_idx += 1

                elif reversed4_sep_idx == start_pad_idx - 3:
                    rotate1_idx = random.randrange(1, reversed4_sep_idx)
                    rotate2_idx = random.randrange(1, reversed4_sep_idx-1)

                    if rotate1_idx == rotate2_idx:
                        rotate2_idx += 1

                else:
                    sent_choice = random.randrange(1, 3)

                    if sent_choice == 1:
                        rotate1_idx = random.randrange(1, reversed4_sep_idx)
                        rotate2_idx = random.randrange(1, reversed4_sep_idx-1)
                    else:
                        rotate1_idx = random.randrange(reversed4_sep_idx+1, start_pad_idx-1)
                        rotate2_idx = random.randrange(reversed4_sep_idx+1, start_pad_idx-2)

                    if rotate1_idx == rotate2_idx:
                        rotate2_idx += 1

                reversed_outputs4['input_ids'][rotate1_idx], reversed_outputs4['input_ids'][rotate2_idx] = reversed_outputs4['input_ids'][rotate2_idx], reversed_outputs4['input_ids'][rotate1_idx]
                
                outputs4_input_ids = (normal_outputs4['input_ids'], reversed_outputs4['input_ids'])
                outputs4_token_type_ids = (normal_outputs4['token_type_ids'], reversed_outputs4['token_type_ids'])
                outputs4_attention_mask = (normal_outputs4['attention_mask'], reversed_outputs4['attention_mask'])

                data.append(outputs4_input_ids)
                token_type_ids.append(outputs4_token_type_ids)
                attention_mask.append(outputs4_attention_mask)
                new_targets.append(targets[idx])
            ##################################################

        data.append(outputs1_input_ids)
        token_type_ids.append(outputs1_token_type_ids)
        attention_mask.append(outputs1_attention_mask)
        new_targets.append(targets[idx])

    return data, attention_mask, token_type_ids, new_targets