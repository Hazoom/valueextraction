import os
import json
import argparse
import torch
from torch import nn
from transformers import BertPreTrainedModel, BertModel, BertTokenizer


def _load_data(data_file: str):
    with open(data_file, "r") as in_fp:
        return [json.loads(j_line.strip()) for j_line in list(in_fp)]


def _encode_json(tokenizer, train_json):
    utterance_encoding = tokenizer.encode_plus(train_json["utterance"])
    parameter_encoding = tokenizer.encode_plus(train_json["parameter"])

    value = train_json["value"]
    if not value:
        print("problem")
        return dict()

    value_token_id = tokenizer.encode_plus(value, add_special_tokens=False)["input_ids"][0]

    if value_token_id not in utterance_encoding["input_ids"]:
        print("problem")
        return dict()

    value_token_position = utterance_encoding["input_ids"].index(value_token_id)
    pair_input_ids = utterance_encoding["input_ids"] + parameter_encoding["input_ids"][1:]
    token_type_ids = [0] * len(utterance_encoding["token_type_ids"]) + [1] * (
                len(parameter_encoding["token_type_ids"]) - 1)
    attention_mask = utterance_encoding["attention_mask"] + parameter_encoding["attention_mask"][1:]

    # labels is a vector of zeros except in the position of the value
    labels = [0] * value_token_position + [1] + [0] * (len(pair_input_ids) - value_token_position - 1)

    assert len(pair_input_ids) == len(token_type_ids)
    assert len(token_type_ids) == len(attention_mask)
    assert len(attention_mask) == len(labels)

    return dict(
        input_ids=pair_input_ids,
        token_type_ids=token_type_ids,
        attention_mask=attention_mask,
        labels=attention_mask
    )


class BertForValueExtraction(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Labels for computing the token classification loss.
            Indices should be in ``[0, ..., config.num_labels - 1]``.

    Returns:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.BertConfig`) and inputs:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when ``labels`` is provided) :
            Classification loss.
        scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, config.num_labels)`)
            Classification scores (before SoftMax).
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.

    Examples::

        from transformers import BertTokenizer, BertForTokenClassification
        import torch

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForTokenClassification.from_pretrained('bert-base-uncased')

        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        labels = torch.tensor([1] * input_ids.size(1)).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)

        loss, scores = outputs[:2]

        """

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        outputs = (logits,)
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                )
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), scores


def train(train_file: str, val_file: str, output_dir: str):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    train_jsons = _load_data(train_file)
    val_jsons = _load_data(val_file)

    train_encoded = [_encode_json(tokenizer, json_line) for json_line in train_jsons]
    val_encoded = [_encode_json(tokenizer, json_line) for json_line in train_jsons]


def main():
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--train-file", type=str, help="Input jsonl file for training", required=True)
    argument_parser.add_argument("--val-file", type=str, help="Input jsonl file for validation", required=True)
    argument_parser.add_argument("--output-dir", type=str, help="Output directory to save model", required=True)
    args = argument_parser.parse_args()
    train(args.train_file, args.val_file, args.output_dir)


if __name__ == "__main__":
    main()
