import transformers
import torch

def freezing(model, keys_to_remain:list = ['pooler', 'classifier']):
    """
    모델의 일부 parameter를 freezing하는 함수

    Args:
        model (BertModel, RobertaModel, ElectraModel): Pretrained Bert model
        keys_to_remain (list): freeze 하지 않는 parameters
    Returns:
        freezing된 model
    """
    for name, param in model.named_parameters():
        param.requires_grad = False
        for key in keys_to_remain:
            if key in name:
                param.requires_grad =True
    return model

def dropout_change(model, new_p=0.3):
    """
    Dropout의 비율 변경 함수

    Args:
        model (BertModel, RobertaModel, ElectraModel): Pretrained Bert model
        new_p (float): Dropout Probability
    Returns:
        Dropout이 변경된 model
    """
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = new_p
    return model