import argparse
import pandas as pd
from tqdm.auto import tqdm
import numpy as np
import transformers
import torch
import torchmetrics
import pytorch_lightning as pl
from torch.optim.lr_scheduler import StepLR
from pytorch_lightning.loggers import WandbLogger
import wandb
from datetime import datetime
import yaml
import random
from datetime import datetime, timezone, timedelta
from module.preprocessing import drop_marks, check_spell
from module.seed import seed_everything
from module.augmentation import augment
from module.rdrop import r_augment
from module.model_params import freezing, dropout_change
from module.memo import memo
from torch import nn
from sklearn.model_selection import KFold

class Dataset(torch.utils.data.Dataset):
    """
    Dataset 생성 Class

    Attributes:
        inputs (list): tokenize된 입력 문장
        attention_masks (list): inputs에 대한 attention masks 
        token_type_ids (list): inputs에 대한 token type ids 
        targets (list): inputs에 대한 targets

    Methods:
        __init__: Dataset Class 초기화 함수
        __getitem__: Dataset Class의 객체에 인덱스에 접근할 때 호출되는 함수
        __len__: Dataset Class에 len() 함수를 사용했을 때 호출되는 함수
    """
    def __init__(self, inputs, attention_masks, token_type_ids, targets=[]):
        """
        Dataset Class 초기화 함수
        
        Args:
            inputs (list): tokenize된 입력 문장 
            attention_masks (list): inputs에 대한 attention masks 
            token_type_ids (list): inputs에 대한 token type ids 
            targets (list, default=[]): inputs에 대한 targets
        """
        self.inputs = inputs
        self.attention_masks = attention_masks
        self.token_type_ids = token_type_ids
        self.targets = targets
    
    def __getitem__(self, idx):
        """
        Dataset Class의 객체에 인덱스에 접근할 때 호출되는 함수
        
        Args:
            idx (int): 접근하고자 하는 데이터의 index 
        
        Return:
            target이 empty list인 경우:
                해당 index의 inputs, attention mask, token type id에 대한 정보
            target이 empty list가 아닌 경우:
                해당 index의 inputs, attention mask, token type id, target에 대한 정보
        """
        if len(self.targets) == 0:
            return torch.tensor(self.inputs[idx]), torch.tensor(self.attention_masks[idx]), torch.tensor(self.token_type_ids[idx])
        else:
            return torch.tensor(self.inputs[idx]), torch.tensor(self.attention_masks[idx]), torch.tensor(self.token_type_ids[idx]), torch.tensor(self.targets[idx])

    def __len__(self):
        """
        Dataset Class에 len() 함수를 사용했을 때 호출되는 함수

        Return:
            inputs 데이터의 개수
        """
        return len(self.inputs)

class Dataloader(pl.LightningDataModule):
    """
    DataLoader 생성 Class

    Attributes:
        model_name (str): 불러오고자 하는 model의 이름 
        batch_size (int): Dataset에 적용하려는 batch 크기 
        shuffle (bool): Dataset의 shuffle 여부 
        train_path (str): Train Dataset의 path 
        dev_path (str): Dev Dataset의 path 
        test_path (str): Test Dataset의 path 
        predict_path (str): Predict Dataset의 path 
        train_dataset (Dataset): Train Dataset
        val_dataset (Dataset): Validation Dataset
        test_dataset (Dataset): Test Dataset
        predict_dataset (Dataset): Predict Dataset
        do_drop_marks (int): 특수문자를 제거해주는 함수의 사용 여부
        do_check_spell (int): 맞춤법 검사를 해주는 함수의 사용 여부
        do_sampler (int): Weighted sampler를 적용해주는 함수의 사용 여부
        do_rdrop (int): R-drop을 사용하는 여부
        do_sbert (int): SBert를 사용하는지 여부
        do_kfold (int): K-Fold를 사용하는지 여부
        seed (int): 고정된 seed값
        k (int): K-Fold의 k값
        max_length (int): tokenizer의 최대 token 길이
        tokenizer (BertTokenizerFast): pretrained tokenizer
        target_columns (list): Target에 해당하는 Column 이름
        delete_columns (list): model에서 사용하지 않는 Column 이름
        text_columns (list): Inputs에 해당하는 Column 이름
        train_sampler (WeightedRandomSampler): Train에 적용하는 sampler 변수
    
    Methods:
        __init__: DataLoader Class 초기화 함수
        tokenizing: 문장을 tokenize하는 함수
        preprocessing: 문장 전처리 함수
        get_sampler: Weighted sampler를 사용하기 위한 함수
        setup: 데이터를 준비하기 위한 함수
        train_dataloader: 전처리된 Train Data를 DataLoader로 변환해주는 함수
        val_dataloader: 전처리된 Validation Data를 DataLoader로 변환해주는 함수
        test_dataloader: 전처리된 Test Data를 DataLoader로 변환해주는 함수
        predict_dataloader: 전처리된 Predict Data를 DataLoader로 변환해주는 함수
    """
    def __init__(self, model_name, batch_size, shuffle, train_path, dev_path, test_path, predict_path, 
                 do_drop_marks, do_check_spell, do_sampler, do_augmentation, do_rdrop, do_sbert, do_kfold, seed_value, k=0, max_length=160):
        """
        DataLoader Class 초기화 함수

        Args:
            model_name (str): 불러오고자 하는 model의 이름 
            batch_size (int): Dataset에 적용하려는 batch 크기 
            shuffle (bool): Dataset의 shuffle 여부 
            train_path (str): Train Dataset의 path 
            dev_path (str): Dev Dataset의 path 
            test_path (str): Test Dataset의 path 
            predict_path (str): Predict Dataset의 path 
            do_drop_marks (int): 특수문자를 제거해주는 함수의 사용 여부
            do_check_spell (int): 맞춤법 검사를 해주는 함수의 사용 여부
            do_sampler (int): Weighted sampler를 적용해주는 함수의 사용 여부
            do_augmentation (int): Train Data를 증강하는지 여부
            do_rdrop (int): R-drop을 사용하는지 여부
            do_sbert (int): SBert를 사용하는지 여부
            do_kfold (int): K-Fold를 사용하는지 여부
            seed (int): 고정된 seed값
            k (int, default=0): K-Fold의 k값
            max_length (int, default=160): tokenizer의 최대 token 길이
        """
        super().__init__()
        self.model_name = model_name
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.train_path = train_path
        self.dev_path = dev_path
        self.test_path = test_path
        self.predict_path = predict_path
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.predict_dataset = None
        self.do_drop_marks = do_drop_marks
        self.do_check_spell = do_check_spell
        self.do_sampler = do_sampler
        self.do_augmentation = do_augmentation
        self.do_rdrop = do_rdrop
        self.do_sbert = do_sbert
        self.do_kfold = do_kfold
        self.seed = seed_value
        self.k = k
        self.max_length = max_length
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, max_length=self.max_length)
        self.target_columns = ['label']
        self.delete_columns = ['id']
        self.text_columns = ['sentence_1', 'sentence_2']
        self.train_sampler = None

    def tokenizing(self, dataframe):
        """
        문장을 tokenize하는 함수

        Args:
            dataframe (DataFrame): tokenize 하고자하는 Dataframe

        Returns:
            tokenize된 inputs, attention masks, token type ids
        """
        data = []
        attention_masks = []
        token_type_ids = []

        if self.do_sbert:
            for _, item in tqdm(dataframe.iterrows(), desc='tokenizing', total=len(dataframe)):
                text1, text2 = [item[text_column] for text_column in self.text_columns]
                outputs1 = self.tokenizer(text1, add_special_tokens=True, padding='max_length', truncation=True, max_length=self.max_length)
                outputs2 = self.tokenizer(text2, add_special_tokens=True, padding='max_length', truncation=True, max_length=self.max_length)

                data.append((outputs1['input_ids'], outputs2['input_ids']))
                attention_masks.append((outputs1['attention_mask'], outputs2['attention_mask']))
                token_type_ids.append((outputs1['token_type_ids'], outputs2['token_type_ids']))
        else:
            for _, item in tqdm(dataframe.iterrows(), desc='tokenizing', total=len(dataframe)):
                text1, text2 = [item[text_column] for text_column in self.text_columns]
                outputs = self.tokenizer(text1, text2, add_special_tokens=True, padding='max_length', truncation=True, max_length=self.max_length)

                data.append(outputs['input_ids'])
                attention_masks.append(outputs['attention_mask'])
                token_type_ids.append(outputs['token_type_ids'])
            
        if 1 not in token_type_ids[0]:
            print('token_type_ids가 적용되지 않습니다.')

        return data, attention_masks, token_type_ids

    def preprocessing(self, data, is_train=False):
        """
        문장 전처리 함수

        Args:
            data (DataFrame): 전처리를 진행하고자 하는 DataFrame
            is_train (bool, default=False): 입력 데이터가 Train인지 여부 확인
        
        Returns:
            전처리가 완료된 inputs에 대해서 tokenize가 완료된 inputs, attention masks, token type ids, targets
        """
        data = data.drop(columns=self.delete_columns)

        try:
            targets = data[self.target_columns].values.tolist()
        except:
            targets = []
        
        if self.do_drop_marks:
            data = drop_marks(data, clean_drop=True)

        if self.do_check_spell:
            data = check_spell(data)

        if self.do_rdrop:
            inputs, attention_masks, token_type_ids, new_targets = r_augment(data, self.tokenizer, targets, self.max_length, self.do_augmentation)
            return inputs, attention_masks, token_type_ids, new_targets
        
        if is_train and self.do_augmentation:
            inputs, attention_masks, token_type_ids, new_targets = augment(data, self.tokenizer, targets, self.max_length)
            return inputs, attention_masks, token_type_ids, new_targets

        inputs, attention_masks, token_type_ids = self.tokenizing(data)
        print('Preprocessing is Done')
        
        return inputs, attention_masks, token_type_ids, targets
    
    def get_sampler(self, labels):
        """
        Weighted sampler를 사용하기 위한 함수

        Args:
            labels (list): sampler를 사용하고자 하는 Targets
        
        Returns:
            Targets 개수에 따라 설정된 WeightedRandomSampler
        """
        num_per_label = [0, 0, 0, 0, 0]
        for label in labels:
            if 0.0<=label[0]<1.0:
                num_per_label[0]+=1
            elif 1.0<=label[0]<2.0:
                num_per_label[1]+=1
            elif 2.0<label[0]<=3.0:
                num_per_label[2]+=1
            elif 3.0<=label[0]<4.0:
                num_per_label[3]+=1
            else:
                num_per_label[4]+=1

        class_count = np.array(num_per_label)
        weights = 1. / class_count

        samples_weight = []
        for idx, t in enumerate(labels):
            if 0.0<=t[0]<1.0:
                samples_weight.append(weights[0])
            elif 1.0<=t[0]<2.0:
                samples_weight.append(weights[1])
            elif 2.0<=t[0]<3.0:
                samples_weight.append(weights[2])
            elif 3.0<=t[0]<4.0:
                samples_weight.append(weights[3])
            elif 4.0<=t[0]<=5.0:
                samples_weight.append(weights[4])

        samples_weight = np.array(samples_weight)
        samples_weight = torch.from_numpy(samples_weight)
        samples_weight = samples_weight.double()
        sampler = torch.utils.data.WeightedRandomSampler(samples_weight, len(samples_weight))

        return sampler

    def setup(self, stage='fit'):
        """
        데이터를 준비하기 위한 함수
        
        Args:
            stage (str, default='fit'): Train을 위한 Data인지 Test를 위한 데이터인지 확인
        """
        if stage == 'fit':
            train_data = pd.read_csv(self.train_path, index_col=0)
            val_data = pd.read_csv(self.dev_path)

            if self.do_kfold:
                data = pd.concat([train_data, val_data], ignore_index=True)

                kf = KFold(n_splits=self.num_splits, shuffle=self.shuffle, random_state=self.seed)
                
                train_datas = []
                val_datas = []
            
                for t, v in kf.split(data):
                    train_datas.append(data.iloc[t].reset_index())
                    val_datas.append(data.iloc[v].reset_index())

                train_inputs, train_attention_masks, train_token_type_ids, train_targets = self.preprocessing(train_datas[self.k], is_train=True)
                val_inputs, val_attention_masks, val_token_type_ids, val_targets = self.preprocessing(val_datas[self.k])
            else:
                train_inputs, train_attention_masks, train_token_type_ids, train_targets = self.preprocessing(train_data, is_train=True)
                val_inputs, val_attention_masks, val_token_type_ids, val_targets = self.preprocessing(val_data)

            self.train_dataset = Dataset(train_inputs, train_attention_masks, train_token_type_ids, train_targets)
            self.val_dataset = Dataset(val_inputs, val_attention_masks, val_token_type_ids, val_targets)

            self.train_sampler = self.get_sampler(train_targets)
        else:
            if self.do_kfold:
                self.test_dataset = self.val_dataset
            else:
                test_data = pd.read_csv(self.test_path)
                test_inputs, test_attention_masks, test_token_type_ids, test_targets = self.preprocessing(test_data)
                self.test_dataset = Dataset(test_inputs, test_attention_masks, test_token_type_ids, test_targets)

            predict_data = pd.read_csv(self.predict_path)
            predict_inputs, predict_attention_masks, predict_token_type_ids, predict_targets = self.preprocessing(predict_data)
            self.predict_dataset = Dataset(predict_inputs, predict_attention_masks, predict_token_type_ids, [])


    def train_dataloader(self):
        """
        전처리된 Train Data를 DataLoader로 변환해주는 함수

        Returns:
            전처리된 Train Data에 대한 DataLoader
        """
        if self.do_sampler:
            return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=4,sampler = self.train_sampler, drop_last=True)
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=4,shuffle=args.shuffle, drop_last=True)

    def val_dataloader(self):
        """
        전처리된 Validation Data를 DataLoader로 변환해주는 함수

        Returns:
            전처리된 Validation Data에 대한 DataLoader
        """
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size,num_workers=4)

    def test_dataloader(self):
        """
        전처리된 Test Data를 DataLoader로 변환해주는 함수

        Returns:
            전처리된 Test Data에 대한 DataLoader
        """
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size,num_workers=4)

    def predict_dataloader(self):
        """
        전처리된 Predict Data를 DataLoader로 변환해주는 함수

        Returns:
            전처리된 Predict Data에 대한 DataLoader
        """
        return torch.utils.data.DataLoader(self.predict_dataset, batch_size=self.batch_size,num_workers=4)

class Model(pl.LightningModule):
    """
    Model 생성 Class

    Attributes:
        model_name (str): 불러오고자 하는 model의 이름 
        lr (float): model 학습시 사용할 learning rate
        max_epoch (int): 모델 학습 횟수
        do_lstm (int): Pretrained Bert 계열 model에 lstm을 붙인 model을 선택
        do_rdrop (int): Pretrained Bert 계열 model에 linear layer를 붙인 뒤 r-drop을 적용하는 model을 선택
        do_param_freeze (int): Pretrained Bert 계열 model의 parameter를 freeze하는지 여부
        do_sbert (int): SBert를 사용하는지 여부
        do_label_smoothing (int): label smoothing을 사용하는지 여부
        drop_out_prob (float, default=0.1): dropout의 probability
        plm (BertModel, RobertaModel, ElectraModel): 불러온 Pretrained model
        lstm (LSTM): LSTM 모델
        fc (Linear): Fully Connected Layer
        dropout (Dropout): Dropout layer
        net (Sequential): Dropout, Linear, ReLU가 합쳐져 있는 Layer
        projection (Linear): Fully Connected Layer
        loss_func (MSELoss): 손실 함수

    Methods:
        __init__: 
        forward: 
        training_step: 
        training_step: 
        training_step: 
        training_step: 
        lr_lambda: 
        configure_optimizer: 
    """
    def __init__(self, model_name, lr, max_epoch, do_lstm, do_rdrop, do_param_freeze, do_sbert, do_label_smoothing, drop_out_prob=0.1, remain_params=[], hidden_size=256):
        """
        Model Class 초기화 함수

        Args:
            model_name (str): 불러오고자 하는 model의 이름 
            lr (float): model 학습시 사용할 learning rate
            max_epoch (int): 모델의 총 학습 횟수
            do_lstm (int): Pretrained Bert 계열 model에 lstm을 붙인 model을 선택
            do_rdrop (int): Pretrained Bert 계열 model에 linear layer를 붙인 뒤 r-drop을 적용하는 model을 선택
            do_param_freeze (int): Pretrained Bert 계열 model의 parameter를 freeze하는지 여부
            do_sbert (int): SBert를 사용하는지 여부
            do_label_smoothing (int): label smoothing을 사용하는지 여부
            drop_out_prob (float, default=0.1): dropout의 probability
            remain_params (list, default=[]): model의 parameter들 중 학습을 진행할 parameter를 설정
        """
        super().__init__()
        self.save_hyperparameters()

        self.model_name = model_name
        self.lr = lr
        self.max_epoch = max_epoch
        self.do_lstm = do_lstm
        self.do_rdrop = do_rdrop
        self.do_param_freeze = do_param_freeze
        self.do_sbert = do_sbert
        self.do_label_smoothing = do_label_smoothing
        self.drop_out_prob = drop_out_prob

        if self.do_lstm:
            self.plm = transformers.AutoModel.from_pretrained(model_name)
            self.lstm = nn.LSTM(input_size=768, hidden_size=hidden_size, num_layers=2, bidirectional=True, batch_first=True)
            self.fc = nn.Linear(hidden_size * 2, 1)
            self.dropout = nn.Dropout(self.drop_out_prob)
        elif self.do_rdrop or self.do_sbert:
            self.plm = transformers.AutoModel.from_pretrained(model_name)
            if do_rdrop:
                self.net = nn.Sequential(
                    nn.Dropout(self.drop_out_prob),
                    nn.Linear(self.plm.config.hidden_size, self.plm.config.hidden_size),
                    nn.ReLU()
                )
                self.projection = nn.Linear(self.plm.config.hidden_size*3, 1)
        else:
            self.plm = transformers.AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=model_name, num_labels=1)

        if self.do_param_freeze:
            self.plm = freezing(self.plm, keys_to_remain=remain_params)
            self.plm = dropout_change(self.plm, new_p = drop_out_prob)

        self.loss_func = torch.nn.MSELoss()

    def forward(self, x, attention_mask=[], token_type_ids=[]):
        """
        Model의 forward 함수

        Args:
            x (list): DataLoader를 통해 batch size만큼 들어온 inputs
            attention_mask (list, dafault=[]): DataLoader를 통해 batch size만큼 들어온 attention masks
            token_type_ids (list, dafault=[]): DataLoader를 통해 batch size만큼 들어온 token type ids
        
        Returns:
            모델의 출력 값
        """
        outputs = self.plm(x, attention_mask=attention_mask, token_type_ids=token_type_ids)

        if self.do_lstm:
            outputs, _ = self.lstm(outputs['logits'])
            pooled_output = outputs.mean(dim=1)
            pooled_output = self.dropout(pooled_output)
            logits = self.fc(pooled_output)
        elif self.do_rdrop:
            hidden_states = outputs.last_hidden_state
            batch_size, seq_length, hidden_size = hidden_states.shape
        
            cls_cond = (x == self.tokenizer.convert_tokens_to_ids('[CLS]')).bool()
            sep_cond = (x == self.tokenizer.convert_tokens_to_ids('[SEP]')).bool()
        
            special_states = hidden_states[cls_cond + sep_cond].view(batch_size, -1, hidden_size)

            special_hs = self.net(special_states)
        
            pooled_output = special_hs.view(batch_size, -1)
            logits = self.projection(pooled_output)
        elif self.do_sbert:
            token_embeddings = outputs[0]
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()

            logits = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        else:
            logits = outputs['logits']

        return logits

    def training_step(self, batch, batch_idx):
        """
        Model의 각 Training 단계에서 수행되는 주요 계산에 대한 함수

        Args:
            batch (Tensor): Training Data의 mini batch
            batch_idx (int): 각 mini batch의 index

        Returns:
            Training 단계에서 발생한 loss
        """
        x, attention_mask, token_type_ids, y = batch

        if self.do_rdrop:
            x1, x2 = x[:, 0], x[:, 1]
            attention_mask1, attention_mask2 = attention_mask[:, 0], attention_mask[:, 1]
            token_type_ids1, token_type_ids2 = token_type_ids[:, 0], token_type_ids[:, 1]

            logits1 = self(x1, attention_mask1, token_type_ids1)
            logits2 = self(x2, attention_mask2, token_type_ids2)

            loss_rmse1 = torch.sqrt(self.loss_func(logits1, y.float()))
            loss_rmse2 = torch.sqrt(self.loss_func(logits2, y.float()))
            loss_kl = nn.functional.kl_div(nn.functional.softmax(logits1, dim=1), nn.functional.softmax(logits2, dim=1), reduction="batchmean")
        
            loss = loss_kl + 0.5 * loss_rmse1 + 0.5 * loss_rmse2
            
        elif self.do_sbert:
            x1, x2 = x[:, 0], x[:, 1]
            attention_mask1, attention_mask2 = attention_mask[:, 0], attention_mask[:, 1]
            token_type_ids1, token_type_ids2 = token_type_ids[:, 0], token_type_ids[:, 1]

            logits1 = self(x1, attention_mask1, token_type_ids1)
            logits2 = self(x2, attention_mask2, token_type_ids2)

            cos_score_transformation = torch.nn.Identity()
            logits = cos_score_transformation(torch.cosine_similarity(logits1, logits2)) * 2.5 + 2.5

            loss = self.loss_func(logits.view(-1, 1), y)
            
        elif self.do_label_smoothing:
            logits = self(x, attention_mask, token_type_ids)
            labels = y.unsqueeze(1).to(torch.device("cuda:0"))
            
            smoothed_labels = ((1 - self.label_smoothing) * labels) + (self.label_smoothing / labels.shape[1])
            std_dev = torch.sqrt(torch.tensor(0.1, device=torch.device('cuda')))
            dist = torch.distributions.normal.Normal(logits, std_dev)
            log_probs = dist.log_prob(smoothed_labels).sum(dim=1)
            loss = -log_probs.mean()
            
        else:
            logits = self(x, attention_mask, token_type_ids)
            loss = self.loss_func(logits, y.float())

        self.log("train_loss", loss, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        """
        Model의 각 Validation 단계에서 수행되는 주요 계산에 대한 함수

        Args:
            batch (Tensor): Validation Data의 mini batch
            batch_idx (int): 각 mini batch의 index

        Returns:
            Validation 단계에서 발생한 loss
        """
        x, attention_mask, token_type_ids, y = batch

        if self.do_sbert:
            x1, x2 = x[:, 0], x[:, 1]
            attention_mask1, attention_mask2 = attention_mask[:, 0], attention_mask[:, 1]
            token_type_ids1, token_type_ids2 = token_type_ids[:, 0], token_type_ids[:, 1]

            logits1 = self(x1, attention_mask1, token_type_ids1)
            logits2 = self(x2, attention_mask2, token_type_ids2)

            cos_score_transformation = torch.nn.Identity()
            logits = cos_score_transformation(torch.cosine_similarity(logits1, logits2)) * 2.5 + 2.5

            loss = self.loss_func(logits.view(-1,1), y)
            
        elif self.do_label_smoothing:
            logits = self(x, attention_mask, token_type_ids)
            labels = y.unsqueeze(1).to(torch.device("cuda:0"))
            
            smoothed_labels = ((1 - self.label_smoothing) * labels) + (self.label_smoothing / labels.shape[1])
            std_dev = torch.sqrt(torch.tensor(0.1, device=torch.device('cuda')))
            dist = torch.distributions.normal.Normal(logits, std_dev)
            log_probs = dist.log_prob(smoothed_labels).sum(dim=1)
            loss = -log_probs.mean()
            
        else:
            logits = self(x, attention_mask, token_type_ids)
            loss = self.loss_func(logits, y.float())

            self.log("val_loss", loss, prog_bar=True)
            self.log("val_pearson", torchmetrics.functional.pearson_corrcoef(logits.squeeze(), y.squeeze()),\
                     prog_bar=True, on_epoch=True)

        return loss

    def test_step(self, batch, batch_idx):
        """
        Model의 각 Test 단계에서 수행되는 주요 계산에 대한 함수

        Args:
            batch (Tensor): Test Data의 mini batch
            batch_idx (int): 각 mini batch의 index

        Returns:
            Test 단계에서 계산된 pearson_corrcoef
        """
        x, attention_mask, token_type_ids, y = batch

        if self.do_sbert:
            x1, x2 = x[:, 0], x[:, 1]
            attention_mask1, attention_mask2 = attention_mask[:, 0], attention_mask[:, 1]
            token_type_ids1, token_type_ids2 = token_type_ids[:, 0], token_type_ids[:, 1]

            logits1 = self(x1, attention_mask1, token_type_ids1)
            logits2 = self(x2, attention_mask2, token_type_ids2)

            cos_score_transformation = torch.nn.Identity()
            logits = cos_score_transformation(torch.cosine_similarity(logits1, logits2)) * 2.5 + 2.5
        else:
            logits = self(x, attention_mask, token_type_ids)

        pearson = torchmetrics.functional.pearson_corrcoef(logits.squeeze(), y.squeeze())
        self.log("test_pearson", pearson ,prog_bar=True)

        return pearson

    def predict_step(self, batch, batch_idx):
        """
        Model의 각 Predict 단계에서 수행되는 주요 계산에 대한 함수

        Args:
            batch (Tensor): Predict Data의 mini batch
            batch_idx (int): 각 mini batch의 index

        Returns:
            Predict 단계에서 나온 결과 값
        """
        x, attention_mask, token_type_ids = batch

        if self.do_sbert:
            x1, x2 = x[:, 0], x[:, 1]
            attention_mask1, attention_mask2 = attention_mask[:, 0], attention_mask[:, 1]
            token_type_ids1, token_type_ids2 = token_type_ids[:, 0], token_type_ids[:, 1]

            logits1 = self(x1, attention_mask1, token_type_ids1)
            logits2 = self(x2, attention_mask2, token_type_ids2)

            cos_score_transformation = torch.nn.Identity()
            logits = cos_score_transformation(torch.cosine_similarity(logits1, logits2)) * 2.5 + 2.5
        else:
            logits = self(x, attention_mask, token_type_ids)

        return logits.squeeze()
    

    def lr_lambda(self, epoch):
        """
        learning rate를 조절하기 위한 함수

        Args:
            epoch (int): 모델의 현재 epoch

        Returns:
            learning rate의 변동값
        """
        max_factor = 3
        
        if epoch < self.max_epoch / 2:
            return 1 + epoch * (max_factor - 1) / (self.max_epoch / 2)
        elif epoch == self.max_epoch / 2:
            return max_factor
        else:
            return max_factor - (epoch - self.max_epoch / 2) * (max_factor - self.lr) / (self.max_epoch / 2 - 1)
        
    def configure_optimizers(self):
        """
        optimizer와 scheduler를 정의하기 위한 함수

        Returns:
            optimizer와 scheduler
        """
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.8)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=8e-6)
        # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-4, total_steps=32*8, epochs=8, anneal_strategy='linear', div_factor=10)
        # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=self.lr_lambda)
        return [optimizer], [scheduler]


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='beomi/KcELECTRA-base-v2022', type=str)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--max_epoch', default=1, type=int)
    parser.add_argument('--shuffle', default=True)
    parser.add_argument('--learning_rate', default=3e-5, type=float)
    parser.add_argument('--train_path', default='./data/pre_train.csv')
    parser.add_argument('--dev_path', default='./data/pre_dev.csv')
    parser.add_argument('--test_path', default='./data/pre_dev.csv')
    parser.add_argument('--predict_path', default='./data/pre_test.csv')
    parser.add_argument('--drop_marks', default=0, type=int)
    parser.add_argument('--check_spell', default=0, type=int)
    parser.add_argument('--sampler', default=0, type=int)
    parser.add_argument('--augmentation', default=0, type=int)
    parser.add_argument('--lstm', default=0, type=int)
    parser.add_argument('--rdrop', default=0, type=int)
    parser.add_argument('--param_freeze', default=0, type=int)
    parser.add_argument('--sbert', default=0, type=int)
    parser.add_argument('--label_smoothing', default=0, type=int)
    parser.add_argument('--kfold', default=0, type=int)
    parser.add_argument('--nums_fold', default=5, type=int)
    parser.add_argument('--dropout_prob', default=0.1, type=int)
    parser.add_argument('--max_length', default=160, type=int)
    parser.add_argument('--seed', default=1234, type=int)
    parser.add_argument('--remain_params', default=['1', '2', '3', '4', '5', '6', '7' ,'8' ,'9' ,'10' ,'11', 'classifier'], type=list)
    args = parser.parse_args()

    # seed
    seed_everything(args.seed)

    if args.kfold:
        for k in range(args.nums_fold):
            dataloader = Dataloader(args.model_name, args.batch_size, args.shuffle, args.train_path, args.dev_path,
                            args.test_path, args.predict_path, args.drop_marks, args.check_spell, args.sampler, args.augmentation, args.rdrop, args.sbert, args.kfold, args.seed, args.max_length)
            # TRAIN 
            model = Model(args.model_name, args.learning_rate, args.max_epoch, args.lstm, args.rdrop, args.param_freeze, args.sbert, args.label_smoothing, args.dropout_prob, args.remain_params)
    
            today = datetime.now(tz=timezone(timedelta(hours=9))).strftime('%Y%m%d_%H:%M')
            wandb_logger = WandbLogger(project="Final", name=f'{today}')

            # callback
            lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='epoch')
            early_stop = pl.callbacks.EarlyStopping(monitor='val_pearson', mode='max', patience=5)

            trainer = pl.Trainer(accelerator='gpu', max_epochs=args.max_epoch,
                          log_every_n_steps=1, logger=wandb_logger,
                          callbacks=[lr_monitor, early_stop])
            trainer.fit(model=model, datamodule=dataloader)
            test_pearson = trainer.test(model=model, datamodule=dataloader)[0]['test_pearson']

            torch.save(model, f'./model_{today[-5:]}_{test_pearson}.pt')
    else:
        dataloader = Dataloader(args.model_name, args.batch_size, args.shuffle, args.train_path, args.dev_path,
                            args.test_path, args.predict_path, args.drop_marks, args.check_spell, args.sampler, args.augmentation, args.rdrop, args.sbert, args.kfold, args.seed, args.max_length)
        # TRAIN 
        model = Model(args.model_name, args.learning_rate, args.max_epoch, args.lstm, args.rdrop, args.param_freeze, args.sbert, args.label_smoothing, args.dropout_prob, args.remain_params)
    
        today = datetime.now(tz=timezone(timedelta(hours=9))).strftime('%Y%m%d_%H:%M')
        wandb_logger = WandbLogger(project="Final", name=f'{today}')

        # callback
        lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='epoch')
        early_stop = pl.callbacks.EarlyStopping(monitor='val_pearson', mode='max', patience=5)

        trainer = pl.Trainer(accelerator='gpu', max_epochs=args.max_epoch,
                          log_every_n_steps=1, logger=wandb_logger,
                          callbacks=[lr_monitor, early_stop])
        trainer.fit(model=model, datamodule=dataloader)
        test_pearson = trainer.test(model=model, datamodule=dataloader)[0]['test_pearson']

        torch.save(model, f'./model_{today[-5:]}_{test_pearson}.pt')

        memo('./memo.txt',
            model_name=args.model_name,
            batch_size=args.batch_size,
            max_epoch=args.max_epoch,
            lr=args.learning_rate,
            seed=args.seed,
            pearson=test_pearson)