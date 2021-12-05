from transformers import (ElectraModel, ElectraPreTrainedModel, BertModel, BertPreTrainedModel, RobertaModel,RobertaPreTrainedModel)
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss, BCEWithLogitsLoss, Conv1d
import torch.nn.functional as F
import math
import torch.nn.utils.rnn as rnn_utils
import numpy as np

from transformers.modeling_outputs import QuestionAnsweringModelOutput

BertLayerNorm = torch.nn.LayerNorm

ACT2FN = {"gelu": F.gelu, "relu": F.relu}

class MHA(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3) # (batch_size, seq_len, all_head_size) -> (batch_size, num_attention_heads, seq_len, attention_head_size)

    def forward(self, input_ids_a, input_ids_b, attention_mask=None, head_mask=None, output_attentions=False):
        mixed_query_layer = self.query(input_ids_a)
        mixed_key_layer = self.key(input_ids_b)
        mixed_value_layer = self.value(input_ids_b)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        # (batch_size * num_choice, num_attention_heads, seq_len, attention_head_size) -> (batch_size * num_choice, seq_len, num_attention_heads, attention_head_size)

        # Should find a better way to do this
        w = (
            self.dense.weight.t()
            .view(self.num_attention_heads, self.attention_head_size, self.hidden_size)
            .to(context_layer.dtype)
        )
        b = self.dense.bias.to(context_layer.dtype)

        
        projected_context_layer = torch.einsum("bfnd,ndh->bfh", context_layer, w) + b
        projected_context_layer_dropout = self.dropout(projected_context_layer)
        layernormed_context_layer = self.LayerNorm(input_ids_a + projected_context_layer_dropout)
        return (layernormed_context_layer, attention_probs) if output_attentions else (layernormed_context_layer,)


class GRUWithPadding(nn.Module):
    def __init__(self, config, num_rnn = 1):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_layers = num_rnn
        self.biGRU = nn.GRU(config.hidden_size, config.hidden_size, self.num_layers, batch_first = True, bidirectional = True)

    def forward(self, inputs):
        batch_size = len(inputs)
        sorted_inputs = sorted(enumerate(inputs), key=lambda x: x[1].size(0), reverse = True)
        idx_inputs = [i[0] for i in sorted_inputs]
        inputs = [i[1] for i in sorted_inputs]
        inputs_lengths = [len(i[1]) for i in sorted_inputs]
        # print('idx_inputs:',idx_inputs)
        # print('inputs:',inputs)
        # print('inputs_lengths:',inputs_lengths)

        inputs = rnn_utils.pad_sequence(inputs, batch_first = True)
        inputs = rnn_utils.pack_padded_sequence(inputs, inputs_lengths, batch_first = True) #(batch_size, seq_len, hidden_size)
        # print('inputs:',inputs)

        h0 = torch.rand(2 * self.num_layers, batch_size, self.hidden_size).to(inputs.data.device) # (2, batch_size, hidden_size)
        self.biGRU.flatten_parameters()
        out, _ = self.biGRU(inputs, h0) # (batch_size, 2, hidden_size )
        out_pad, out_len = rnn_utils.pad_packed_sequence(out, batch_first = True) # (batch_size, seq_len, 2 * hidden_size)

        _, idx2 = torch.sort(torch.tensor(idx_inputs))
        idx2 = idx2.to(out_pad.device)
        output = torch.index_select(out_pad, 0, idx2)
        out_len = out_len.to(out_pad.device)
        out_len = torch.index_select(out_len, 0, idx2)

        out_idx = (out_len - 1).unsqueeze(1).unsqueeze(2).repeat([1,1,self.hidden_size * 2])
        output = torch.gather(output, 1, out_idx).squeeze(1)

        return output 


class FuseLayer(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.linear1 = nn.Linear(4 * config.hidden_size, config.hidden_size)
        self.linear2 = nn.Linear(4 * config.hidden_size, config.hidden_size)
        self.linear3 = nn.Linear(2 * config.hidden_size, config.hidden_size)
        self.activation = nn.ReLU()
        self.gate = nn.Sigmoid()

    def forward(self, orig, input1, input2):
        out1 = self.activation(self.linear1(torch.cat([orig, input1, orig - input1, orig * input1], dim = -1)))
        out2 = self.activation(self.linear2(torch.cat([orig, input2, orig - input2, orig * input2], dim = -1)))
        fuse_prob = self.gate(self.linear3(torch.cat([out1, out2], dim = -1)))

        return fuse_prob * input1 + (1 - fuse_prob) * input2

class ElectraForMolweni(ElectraPreTrainedModel):
    def __init__(self, config, num_rnn = 1, num_decoupling = 1):
        super().__init__(config)
        print('this is ElectraForMolweni model.')
        self.electra = ElectraModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)
        self.relation_outputs = nn.Linear(config.max_utterance_num * 4 * config.hidden_size, config.max_utterance_num + 1)
        self.relation_sigmoid = nn.Sigmoid()
        self.type_outputs = nn.Linear(4*config.hidden_size, 16)
        self.init_weights()
        self.dropout = nn.Dropout(p=config.dropout)

        # print('config.max_utterance_num:',self.config.max_utterance_num)
        # print('config.num_labels:',self.config.num_labels)

    def forward(
        self,
        input_ids=None,
        token_type_ids=None,
        attention_mask=None,
        sep_position = None,
        position_ids=None,
        turn_ids = None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        start_positions=None,
        end_positions=None,
        adjacent_matrix=None,
        speaker_ids=None,
        cls_index=None
    ):
        # print(self.relation_outputs.weight)
        # print(self.qa_outputs.weight)
        input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None 
        attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        # (batch_size * choice, seq_len)
        token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        # print(adjacent_matrix.shape)
        #print("sep_pos:", sep_pos)
        
        position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        inputs_embeds = (
            inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1))
            if inputs_embeds is not None
            else None
        )
        
        #print("size of sequence_output:", sequence_output.size())
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(2) # (batch_size * num_choice, 1, 1, seq_len)
        attention_mask = attention_mask.to(dtype=self.dtype)  # fp16 compatibility
        attention_mask = (1.0 - attention_mask) * -10000.0
        # (batch_size * num_choice, 1, 1, seq_len)

        outputs = self.electra(
            input_ids,
            attention_mask=attention_mask.squeeze(1),
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        sequence_output = outputs[0] # (batch_size * num_choice, seq_len, hidden_size) #


        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        # hyc:
        # utter_batch = None
        utter_batch = []
        utter_num = []

        for i in range(sequence_output.shape[0]):
            utter_num_i = 1
            utter_embedding = torch.unsqueeze(sequence_output[i][sep_position[i][0]][:], 0)
            for j in range(1, len(sep_position[i])):
                if sep_position[i][j]!=0 :
                    utter_num_i = utter_num_i+1
                    utter_embedding = torch.cat((utter_embedding,torch.unsqueeze(sequence_output[i][sep_position[i][j]][:], 0)),0)
            # utter_feature = torch.zeros([self.config.max_utterance_num,self.config.max_utterance_num, self.config.hidden_size*4], dtype=torch.float32)
            utter_feature = torch.zeros(
                [utter_num_i, utter_num_i, self.config.hidden_size * 4],
                dtype=torch.float32)
            if sequence_output.is_cuda:
                utter_feature = utter_feature.cuda()

            for k1 in range(utter_num_i):
                for k2 in range(utter_num_i):
                    a = utter_embedding[k1][:]-utter_embedding[k2][:]
                    b = torch.mul(utter_embedding[k1][:],utter_embedding[k2][:])
                    utter_feature[k1][k2][:] = torch.cat((utter_embedding[k1][:], utter_embedding[k2][:], a, b),0)
            # print("utter_feature",utter_feature.shape)

            utter_batch.append(utter_feature)
            utter_num.append(utter_num_i)
        # print(utter_batch.shape)  # batch_size*max_utterance_num*max_utterance_num*(2*hidden_size)

        relation_pred = []  #batch_size*utter_num*(max_utterance_num+1)
        relation_golden = []
        type_pred = []
        type_golden = []
        link_type = []
        for i in range(sequence_output.shape[0]):
            # link prediction
            # print(utter_batch[i].shape)
            link_feature_withpad = torch.zeros([utter_num[i], self.config.max_utterance_num*4*self.config.hidden_size])
            if sequence_output.is_cuda:
                link_feature_withpad = link_feature_withpad.cuda()

            link_feature = utter_batch[i].view(utter_num[i], utter_num[i]*4*self.config.hidden_size)
            pad = torch.zeros((self.config.max_utterance_num-utter_num[i])*4*self.config.hidden_size,dtype=torch.float32)
            if sequence_output.is_cuda:
                pad = pad.cuda()
            for line in range(link_feature.shape[0]):
                link_feature_withpad[line] = torch.cat((link_feature[line],pad), 0)
            # print('link_feacture.shape:',link_feature_withpad.shape)
            relation_pred_i = self.relation_outputs(link_feature_withpad)
            relation_pred_i = self.dropout(relation_pred_i)
            # print('relation_pred_i.shape', relation_pred_i.shape)
            relation_pred.append(relation_pred_i)

            relation_prob_i = F.softmax(torch.Tensor(relation_pred_i.cpu()), dim=1)
            relation_pred_label_i = torch.argmax(relation_prob_i, dim=1)
            link_type_i = []
            for y in range(utter_num[i]):
                if relation_pred_label_i[y]<utter_num[i]:
                    link_type_i.append(utter_batch[i][y][relation_pred_label_i[y]])
            if len(link_type_i):
                link_type_i = torch.stack(link_type_i, 0)
                link_type_i = self.type_outputs(link_type_i)
                link_type_i = self.dropout(link_type_i)
            # print(link_type_i.shape)
            link_type.append(link_type_i)
            # print(relation_pred_i.shape)
            # print(link_type_i.shape)


            relation_golden_i = []
            type_golden_i = []
            type_pred_i = []

            if adjacent_matrix is not None:
                for y in range(utter_num[i]):
                    no_relation = True
                    for x in range(utter_num[i]):
                        if adjacent_matrix[i][y][x]!=16:
                            no_relation = False
                            relation_golden_i.append(x)
                            type_golden_i.append(adjacent_matrix[i][y][x])
                            type_pred_i.append(utter_batch[i][y][x])
                            break
                    if no_relation:
                        relation_golden_i.append(self.config.max_utterance_num)
                relation_golden_i = torch.Tensor(relation_golden_i)
                type_golden_i = torch.Tensor(type_golden_i)
                if sequence_output.is_cuda:
                    relation_golden_i = relation_golden_i.cuda()
                    type_golden_i = type_golden_i.cuda()
                relation_golden.append(relation_golden_i)
                type_golden.append(type_golden_i)

                type_pred_i = torch.stack(type_pred_i,0)
                # print(type_pred_i.shape)
                type_pred_i = self.type_outputs(type_pred_i)
                type_pred_i = self.dropout(type_pred_i)
                type_pred.append(type_pred_i)
        # end

        outputs = (start_logits, end_logits, relation_pred, link_type) + outputs[2:]
        total_loss = 0
        # print('start/end_position is not None?')
        if start_positions is not None and end_positions is not None:
            if adjacent_matrix is not None:
                # dp_loss_relation = nn.BCEWithLogitsLoss(pos_weight=pos_weight, size_average=True)
                dp_loss_relation = CrossEntropyLoss(size_average=True)
                dp_loss_type = CrossEntropyLoss(size_average=True)
                relation_loss = 0
                type_loss = 0
                for i in range(sequence_output.shape[0]):
                    relation_loss = relation_loss + dp_loss_relation(relation_pred[i],relation_golden[i].long())
                    type_loss = type_loss + dp_loss_type(type_pred[i], type_golden[i].long())
                relation_loss = relation_loss / (sequence_output.shape[0])
                type_loss = type_loss / (sequence_output.shape[0])
                if self.config.task!=1:
                    total_loss = relation_loss + type_loss
                    print('type_loss',type_loss)
                    print('relation_loss',relation_loss)

            # If we are on multi-GPU, split add a dimension
            # print('start/end_position is not None:',start_positions, end_positions)
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)
            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            mrc_loss = (start_loss + end_loss) / 2
            if self.config.task != 2:
                total_loss = total_loss + (start_loss + end_loss) / 2
                print('total_loss:',total_loss)
            
            outputs = (total_loss, mrc_loss, relation_loss, type_loss) + outputs
        return outputs #(loss), reshaped_logits, (hidden_states), (attentions)

class RobertaForMolweni(RobertaPreTrainedModel):
    def __init__(self, config, num_rnn = 1, num_decoupling = 1):
        super().__init__(config)
        print('this is RobertaForMolweni model.')
        self.roberta = RobertaModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        token_type_ids=None,
        attention_mask=None,
        sep_position = None,
        position_ids=None,
        turn_ids = None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        start_positions=None,
        end_positions=None,
        speaker_ids=None,
        cls_index=None
    ):

        input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None 
        attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        # (batch_size * choice, seq_len)
        token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        
        position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        inputs_embeds = (
            inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1))
            if inputs_embeds is not None
            else None
        )
        
        #print("size of sequence_output:", sequence_output.size())
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(2) # (batch_size * num_choice, 1, 1, seq_len)
        attention_mask = attention_mask.to(dtype=self.dtype)  # fp16 compatibility
        attention_mask = (1.0 - attention_mask) * -10000.0

        # print(inputs_embeds)
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask.squeeze(1),
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        sequence_output = outputs[0] # (batch_size * num_choice, seq_len, hidden_size)

        # print('word_level:',word_level.size())

        logits = self.qa_outputs(sequence_output)
        # print('logits:',logits.size())
        start_logits, end_logits = logits.split(1, dim=-1)
        # print('start_logits:',start_logits)
        # print('end_logits:',end_logits)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        # print('start_logits:',start_logits)
        # print('end_logits:',end_logits)
        outputs = (start_logits, end_logits,) + outputs[2:]

        total_loss = None
        # print('start/end_position is not None?')
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            # print('start/end_position is not None:',start_positions, end_positions)
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
            # print('total_loss:',total_loss)
            
            outputs = (total_loss,) + outputs
        return outputs


#20201117 mdfn for molweni
class BertForMolweni(BertPreTrainedModel):
    def __init__(self, config, num_rnn = 1, num_decoupling = 1):
        super().__init__(config)

        self.bert = BertModel(config, add_pooling_layer=False)
        self.num_labels = config.num_labels#20201118
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        self.relation_outputs = nn.Linear(config.max_utterance_num * 4 * config.hidden_size, config.max_utterance_num + 1)
        self.relation_sigmoid = nn.Sigmoid()
        self.type_outputs = nn.Linear(4*config.hidden_size, 16)
        self.dropout = nn.Dropout(p=config.dropout)
        # self.has_ans = nn.Linear(config.hidden_size, 2)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        token_type_ids=None,
        attention_mask=None,
        sep_position = None,
        position_ids=None,
        turn_ids = None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        start_positions=None,
        end_positions=None,
        adjacent_matrix=None,
        speaker_ids=None,
        cls_index=None,
        is_impossibles=None
    ):
        # print('qas_id:',qas_id)
        # print('turn_ids_justin:', turn_ids)
        # print('speaker_ids_justin:', speaker_ids)
        input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None

        position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        inputs_embeds = (
            inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1))
            if inputs_embeds is not None
            else None
        )
        # print('position_ids:',position_ids if position_ids is None else position_ids.size()) #None

        #print("size of sequence_output:", sequence_output.size())
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(2) # (batch_size * num_choice, 1, 1, seq_len)
        attention_mask = attention_mask.to(dtype=self.dtype)  # fp16 compatibility

        attention_mask = (1.0 - attention_mask) * -10000.0

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask.squeeze(1),
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        sequence_output = outputs[0] # (batch_size * num_choice, seq_len, hidden_size)
        # print(sequence_output.shape)
        logits = self.qa_outputs(sequence_output)
        # print('logits:',logits.size())
        start_logits, end_logits = logits.split(1, dim=-1)
        # print('start_logits:',start_logits)
        # print('end_logits:',end_logits)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        # print('start_logits:',start_logits)
        # print('end_logits:',end_logits)
        # first_word = sequence_output[:, 0, :]
        # has_log = self.has_ans(first_word)
        utter_batch = []
        utter_num = []

        for i in range(sequence_output.shape[0]):
            utter_num_i = 1
            utter_embedding = torch.unsqueeze(sequence_output[i][sep_position[i][0]][:], 0)
            for j in range(1, len(sep_position[i])):
                if sep_position[i][j] != 0:
                    utter_num_i = utter_num_i + 1
                    utter_embedding = torch.cat(
                        (utter_embedding, torch.unsqueeze(sequence_output[i][sep_position[i][j]][:], 0)), 0)
            # utter_feature = torch.zeros([self.config.max_utterance_num,self.config.max_utterance_num, self.config.hidden_size*4], dtype=torch.float32)
            utter_feature = torch.zeros(
                [utter_num_i, utter_num_i, self.config.hidden_size * 4],
                dtype=torch.float32)
            if sequence_output.is_cuda:
                utter_feature = utter_feature.cuda()

            for k1 in range(utter_num_i):
                for k2 in range(utter_num_i):
                    a = utter_embedding[k1][:] - utter_embedding[k2][:]
                    b = torch.mul(utter_embedding[k1][:], utter_embedding[k2][:])
                    utter_feature[k1][k2][:] = torch.cat((utter_embedding[k1][:], utter_embedding[k2][:], a, b), 0)
            # print("utter_feature",utter_feature.shape)

            utter_batch.append(utter_feature)
            utter_num.append(utter_num_i)
        # print(utter_batch.shape)  # batch_size*max_utterance_num*max_utterance_num*(2*hidden_size)

        relation_pred = []  # batch_size*utter_num*(max_utterance_num+1)
        relation_golden = []
        type_pred = []
        type_golden = []
        link_type = []
        for i in range(sequence_output.shape[0]):
            # link prediction
            # print(utter_batch[i].shape)
            link_feature_withpad = torch.zeros(
                [utter_num[i], self.config.max_utterance_num * 4 * self.config.hidden_size])
            if sequence_output.is_cuda:
                link_feature_withpad = link_feature_withpad.cuda()

            link_feature = utter_batch[i].view(utter_num[i], utter_num[i] * 4 * self.config.hidden_size)
            pad = torch.zeros((self.config.max_utterance_num - utter_num[i]) * 4 * self.config.hidden_size,
                              dtype=torch.float32)
            if sequence_output.is_cuda:
                pad = pad.cuda()
            for line in range(link_feature.shape[0]):
                link_feature_withpad[line] = torch.cat((link_feature[line], pad), 0)
            # print('link_feacture.shape:',link_feature_withpad.shape)
            relation_pred_i = self.relation_outputs(link_feature_withpad)
            relation_pred_i = self.dropout(relation_pred_i)
            # print('relation_pred_i.shape', relation_pred_i.shape)
            relation_pred.append(relation_pred_i)

            relation_prob_i = F.softmax(torch.Tensor(relation_pred_i.cpu()), dim=1)
            relation_pred_label_i = torch.argmax(relation_prob_i, dim=1)
            link_type_i = []
            for y in range(utter_num[i]):
                if relation_pred_label_i[y] < utter_num[i]:
                    link_type_i.append(utter_batch[i][y][relation_pred_label_i[y]])
            if len(link_type_i):
                link_type_i = torch.stack(link_type_i, 0)
                link_type_i = self.type_outputs(link_type_i)
                link_type_i = self.dropout(link_type_i)
            # print(link_type_i.shape)
            link_type.append(link_type_i)
            # print(relation_pred_i.shape)
            # print(link_type_i.shape)

            relation_golden_i = []
            type_golden_i = []
            type_pred_i = []

            if adjacent_matrix is not None:
                for y in range(utter_num[i]):
                    no_relation = True
                    for x in range(utter_num[i]):
                        if adjacent_matrix[i][y][x] != 16:
                            no_relation = False
                            relation_golden_i.append(x)
                            type_golden_i.append(adjacent_matrix[i][y][x])
                            type_pred_i.append(utter_batch[i][y][x])
                            break
                    if no_relation:
                        relation_golden_i.append(self.config.max_utterance_num)
                relation_golden_i = torch.Tensor(relation_golden_i)
                type_golden_i = torch.Tensor(type_golden_i)
                if sequence_output.is_cuda:
                    relation_golden_i = relation_golden_i.cuda()
                    type_golden_i = type_golden_i.cuda()
                relation_golden.append(relation_golden_i)
                type_golden.append(type_golden_i)

                type_pred_i = torch.stack(type_pred_i, 0)
                # print(type_pred_i.shape)
                type_pred_i = self.type_outputs(type_pred_i)
                type_pred_i = self.dropout(type_pred_i)
                type_pred.append(type_pred_i)
        # end

        outputs = (start_logits, end_logits, relation_pred, link_type) + outputs[2:]
        total_loss = 0
        # outputs = (start_logits, end_logits,) + outputs[2:]

        # total_loss = None
        # print('start/end_position is not None?')
        if start_positions is not None and end_positions is not None:
            if adjacent_matrix is not None:
                # dp_loss_relation = nn.BCEWithLogitsLoss(pos_weight=pos_weight, size_average=True)
                dp_loss_relation = CrossEntropyLoss(size_average=True)
                dp_loss_type = CrossEntropyLoss(size_average=True)
                relation_loss = 0
                type_loss = 0
                for i in range(sequence_output.shape[0]):
                    relation_loss = relation_loss + dp_loss_relation(relation_pred[i],relation_golden[i].long())
                    type_loss = type_loss + dp_loss_type(type_pred[i], type_golden[i].long())
                relation_loss = relation_loss / (sequence_output.shape[0])
                type_loss = type_loss / (sequence_output.shape[0])
                if self.config.task!=1:
                    total_loss = total_loss + (relation_loss + type_loss)
                # print('relation_loss',relation_loss)
                # print('type_loss',type_loss)

            # If we are on multi-GPU, split add a dimension
            # print('start/end_position is not None:',start_positions, end_positions)
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # if len(is_impossibles.size()) > 1:
               #  is_impossibles = is_impossibles.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)
            # is_impossibles.clamp_(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            # choice_loss = loss_fct(has_log, is_impossibles.long())
            mrc_loss = (start_loss + end_loss) / 2
            if self.config.task!=2:
                total_loss = total_loss + (start_loss + end_loss ) / 2
            print('total_loss:',total_loss)

            outputs = (total_loss, mrc_loss, relation_loss, type_loss) + outputs

        return outputs #(loss), start_logits, end_logits, (hidden_states), (attentions)
