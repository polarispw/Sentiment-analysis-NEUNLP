import torch
import torch.nn as nn
from transformers import BertModel


class BertSCModel(nn.Module):

    def __init__(self, pretrained_name='bert-base-uncased', output_hidden_state=False):
        super(BertSCModel, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_name, return_dict=True, output_hidden_states=output_hidden_state)

    def forward(self, inputs):
        input_ids, input_tyi, input_attn_mask = inputs['input_ids'], inputs['token_type_ids'], inputs['attention_mask']
        output = self.bert(input_ids, input_tyi, input_attn_mask)
        return output


class Bert_FFN(BertSCModel):

    def __init__(self, class_size, pretrained_name='bert-base-uncased'):
        super(Bert_FFN, self).__init__(pretrained_name)
        self.classifier = nn.Linear(768, class_size)

    def forward(self, inputs):
        input_ids, input_tyi, input_attn_mask = inputs['input_ids'], inputs['token_type_ids'], inputs['attention_mask']
        output = self.bert(input_ids, input_tyi, input_attn_mask)
        categories_numberic = self.classifier(output.pooler_output)
        return categories_numberic


class Bert_LSTM(BertSCModel):

    def __init__(self, class_size, pretrained_name='bert-base-uncased'):
        super(Bert_LSTM, self).__init__(pretrained_name, output_hidden_state=True)
        self.classifier = nn.LSTM(768, class_size)

    def forward(self, inputs):
        input_ids, input_tyi, input_attn_mask = inputs['input_ids'], inputs['token_type_ids'], inputs['attention_mask']
        output = self.bert(input_ids, input_tyi, input_attn_mask)

        states = output.hidden_states[1:]
        cls_s = [state[:, 0, :] for state in states]
        cls_states = torch.tensor([item.cpu().detach().numpy() for item in cls_s]).cuda()
        cls_output, (_, _) = self.classifier(cls_states)
        categories_numberic = cls_output[-1]
        return categories_numberic
