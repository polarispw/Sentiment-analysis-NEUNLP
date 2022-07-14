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


class BertBase(BertSCModel):

    def __init__(self, class_size, pretrained_name='bert-base-uncased', layers=None, pooling=None):
        super(BertBase, self).__init__(pretrained_name, output_hidden_state=True)
        self.dropout = nn.Dropout(0.1)
        self.pooling = pooling
        if layers is None or layers[0] == -2:
            self.layers = []
        elif layers[0] == -1:
            self.layers = [i for i in range(1, 12)]
        else:
            self.layers = layers
        self.classifier = nn.Linear(768 * max(1, len(self.layers)), class_size)

    def forward(self, inputs):
        input_ids, input_tyi, input_attn_mask = inputs['input_ids'], inputs['token_type_ids'], inputs['attention_mask']
        output = self.bert(input_ids, input_tyi, input_attn_mask)

        if len(self.layers) > 0:
            features = []
            states = output.hidden_states[1:]
            for l in self.layers:
                features.append(states[l][:, 0].unsqueeze(1))
            features = torch.cat(features, dim=1)

            if self.pooling == 'max':
                features, _ = torch.max(features, dim=1)
            elif self.pooling == 'mean':
                features = torch.mean(features, dim=1)
            else:
                features = features.view(features.size(0), -1)

            features = self.dropout(features)
            categories_numberic = self.classifier(features)

        else:
            features = self.dropout(output.pooler_output)
            categories_numberic = self.classifier(features)
        return categories_numberic


class BertFFN(BertSCModel):

    def __init__(self, class_size, pretrained_name='bert-base-uncased'):
        super(BertFFN, self).__init__(pretrained_name)
        self.classifier = nn.Sequential(
            torch.nn.Linear(768, 512),
            torch.nn.Dropout(0.3),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 64),
            torch.nn.Dropout(0.2),
            torch.nn.ReLU(),
            torch.nn.Linear(64, class_size)
        )

    def forward(self, inputs):
        input_ids, input_tyi, input_attn_mask = inputs['input_ids'], inputs['token_type_ids'], inputs['attention_mask']
        output = self.bert(input_ids, input_tyi, input_attn_mask)
        categories_numberic = self.classifier(output.pooler_output)
        return categories_numberic


class BertLSTM(BertSCModel):

    def __init__(self, class_size, pretrained_name='bert-base-uncased'):
        super(BertLSTM, self).__init__(pretrained_name, output_hidden_state=True)
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


class BertSimCSE(BertSCModel):
    def __init__(self, class_size, pretrained_name='bert-base-uncased'):
        super(BertSimCSE, self).__init__(pretrained_name)

    def forward(self, inputs):
        input_ids, input_tyi, input_attn_mask = inputs['input_ids'], inputs['token_type_ids'], inputs['attention_mask']
        output = self.bert(input_ids, input_tyi, input_attn_mask)
        output = output.last_hidden_state[:, 0]
        return output
