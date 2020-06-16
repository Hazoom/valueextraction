import os
import json
import argparse
import logging
from tqdm import trange
import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.optim import AdamW
from transformers import BertPreTrainedModel, BertModel, BertTokenizer, get_linear_schedule_with_warmup
from seqeval.metrics import accuracy_score


logging.basicConfig(format='%(asctime)s %(message)s', level=logging.DEBUG)


# hyper parameters. Note: they are fixed and not fine-tuned on the validation set for the purpose of the exercise
MAX_SEQUENCE_LENGTH = 32
BATCH_SIZE = 32
EPOCHS = 3
MAX_GRAD_NORM = 1.0
LEARNING_RATE = 3e-5
EPSILON = 1e-8


def _load_data(data_file: str):
    results = []
    with open(data_file, "r") as in_fp:
        for json_line in in_fp:
            try:
                utterance_json = json.loads(json_line.strip())
                value = utterance_json["value"]
                parameter = utterance_json["parameter"]
                utterance = utterance_json["utterance"]
                if not value:
                    logging.warning(f"value is empty in: {utterance_json}")
                    continue
                if not parameter:
                    logging.warning(f"parameter is empty in: {utterance_json}")
                    continue
                if value not in utterance:
                    logging.warning(f"value: '{value}' not in utterance: {utterance}")
                    continue
                results.append(utterance_json)
            except json.decoder.JSONDecodeError:
                logging.warning(f"line: {json_line.strip()} has a broken json schema")
    return results


def _encode_json(tokenizer, sample_json: dict, max_sequence_length: int):
    pair_encoded = tokenizer.encode_plus(
        sample_json["utterance"],
        sample_json["parameter"],
        add_special_tokens=True,
        pad_to_max_length=True,
        max_length=max_sequence_length,
    )

    value = sample_json["value"]
    value_token_id = tokenizer.encode_plus(value, add_special_tokens=False)["input_ids"][0]

    pair_input_ids = pair_encoded["input_ids"]

    if value_token_id not in pair_input_ids:
        logging.warning(f"value {value} not in utterance as a word: {sample_json['utterance']}")
        return None

    value_token_position = pair_input_ids.index(value_token_id)
    token_type_ids = pair_encoded["token_type_ids"]
    attention_mask = pair_encoded["attention_mask"]

    # labels is a vector of zeros except in the position of the value
    labels = [0] * value_token_position + [1] + [0] * (len(pair_input_ids) - value_token_position - 1)

    assert len(pair_input_ids) == len(token_type_ids)
    assert len(token_type_ids) == len(attention_mask)
    assert len(attention_mask) == len(labels)

    assert pair_input_ids[value_token_position] == value_token_id

    return dict(
        input_ids=pair_input_ids,
        token_type_ids=token_type_ids,
        attention_mask=attention_mask,
        labels=attention_mask
    )


class BertForValueExtraction(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = 2
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.ff = nn.Sequential(nn.Linear(config.hidden_size, 100), nn.ReLU(), nn.Dropout(config.hidden_dropout_prob),
                                nn.Linear(100, 50), nn.ReLU(), nn.Dropout(config.hidden_dropout_prob),
                                nn.Linear(50, 1),
                                nn.Linear(1, 1))

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
        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.ff(sequence_output)

        outputs = (logits,)
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()

            # only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                )
                predicted_probs = torch.softmax(active_logits, 1)
                loss = loss_fct(predicted_probs, active_labels)
            else:
                # predicted_probs = torch.softmax(logits, 1)
                loss = loss_fct(logits.view(-1), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), scores


def train(train_file: str, val_file: str, output_dir: str):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    logging.info("Loading train JSONs...")
    train_jsons = _load_data(train_file)
    logging.info("Loading validation JSONs...")
    val_jsons = _load_data(val_file)

    logging.info("Encoding training samples...")
    train_encoded = [_encode_json(tokenizer, json_line, MAX_SEQUENCE_LENGTH) for json_line in train_jsons]
    train_encoded = [sample for sample in train_encoded if sample]

    logging.info("Encoding validation samples...")
    val_encoded = [_encode_json(tokenizer, json_line, MAX_SEQUENCE_LENGTH) for json_line in val_jsons]
    val_encoded = [sample for sample in val_encoded if sample]

    # convert training samples to tensors
    train_inputs = torch.tensor([train_sample["input_ids"] for train_sample in train_encoded])
    train_token_type_ids = torch.tensor([train_sample["token_type_ids"] for train_sample in train_encoded])
    train_attention_mask = torch.tensor([train_sample["attention_mask"] for train_sample in train_encoded])
    train_labels = torch.tensor([train_sample["labels"] for train_sample in train_encoded])

    # convert validation samples to tensors
    val_inputs = torch.tensor([val_sample["input_ids"] for val_sample in val_encoded])
    val_token_type_ids = torch.tensor([val_sample["token_type_ids"] for val_sample in val_encoded])
    val_attention_mask = torch.tensor([val_sample["attention_mask"] for val_sample in val_encoded])
    val_labels = torch.tensor([val_sample["labels"] for val_sample in val_encoded])

    # initialize the model
    model = BertForValueExtraction.from_pretrained("bert-base-uncased")

    # create training data loader
    train_data = TensorDataset(train_inputs, train_attention_mask, train_token_type_ids, train_labels)
    train_sampler = RandomSampler(train_data)
    train_data_loader = DataLoader(train_data, sampler=train_sampler, batch_size=BATCH_SIZE)

    # create validation data loader
    val_data = TensorDataset(val_inputs, val_attention_mask, val_token_type_ids, val_labels)
    val_sampler = SequentialSampler(val_data)
    val_data_loader = DataLoader(val_data, sampler=val_sampler, batch_size=BATCH_SIZE)

    full_fine_tuning = True
    if full_fine_tuning:
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'gamma', 'beta']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.0}
        ]
    else:
        param_optimizer = list(model.classifier.named_parameters())
        optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]

    # creating optimizer for training
    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=LEARNING_RATE,
        eps=EPSILON
    )

    # total number of training steps is number of batches * number of epochs
    total_steps = len(train_data_loader) * EPOCHS

    # create the learning rate scheduler
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # store the average loss after each epoch so we can plot them
    loss_values, validation_loss_values = [], []

    for _ in trange(EPOCHS, desc="Epoch"):
        # ========================================
        #               Training
        # ========================================
        # Perform one full pass over the training set.

        # Put the model into training mode.
        model.train()
        # Reset the total loss for this epoch.
        total_loss = 0

        # Training loop
        for step, batch in enumerate(train_data_loader):
            # add batch to gpu
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_attention_mask, b_token_type_ids, b_labels = batch
            # Always clear any previously calculated gradients before performing a backward pass.
            model.zero_grad()
            # forward pass
            # This will return the loss (rather than the model output)
            # because we have provided the `labels`.
            outputs = model(
                b_input_ids,
                token_type_ids=b_token_type_ids,
                # attention_mask=b_attention_mask,
                labels=b_labels
            )
            # get the loss
            loss = outputs[0]
            # Perform a backward pass to calculate the gradients.
            loss.backward()
            # track train loss
            total_loss += loss.item()
            # Clip the norm of the gradient
            # This is to help prevent the "exploding gradients" problem.
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=MAX_GRAD_NORM)
            # update parameters
            optimizer.step()
            # Update the learning rate.
            scheduler.step()

        # Calculate the average loss over the training data.
        avg_train_loss = total_loss / len(train_data_loader)
        print("Average train loss: {}".format(avg_train_loss))

        # Store the loss value for plotting the learning curve.
        loss_values.append(avg_train_loss)

        # ========================================
        #               Validation
        # ========================================
        # After the completion of each training epoch, measure our performance on
        # our validation set.

        # Put the model into evaluation mode
        model.eval()
        # reset the validation loss for this epoch
        eval_loss, eval_accuracy = 0, 0
        predictions, true_labels = [], []
        for batch in val_data_loader:
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_attention_mask, b_token_type_ids, b_labels = batch

            # Telling the model not to compute or store gradients,
            # saving memory and speeding up validation
            with torch.no_grad():
                # Forward pass, calculate logit predictions.
                # This will return the logits rather than the loss because we have not provided labels.
                outputs = model(
                    b_input_ids,
                    token_type_ids=b_token_type_ids,
                    attention_mask=b_attention_mask,
                    labels=b_labels
                )
            # Move logits and labels to CPU
            logits = outputs[1].detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            # Calculate the accuracy for this batch of test sentences.
            eval_loss += outputs[0].mean().item()
            predictions.extend([list(p) for p in np.argmax(logits, axis=2)])
            true_labels.extend(label_ids)

        eval_loss = eval_loss / len(val_data_loader)
        validation_loss_values.append(eval_loss)
        print("Validation loss: {}".format(eval_loss))
        pred_tags = [tag_values[p_i] for p, l in zip(predictions, true_labels)
                     for p_i, l_i in zip(p, l) if tag_values[l_i] != "PAD"]
        valid_tags = [tag_values[l_i] for l in true_labels
                      for l_i in l if tag_values[l_i] != "PAD"]
        print("Validation Accuracy: {}".format(accuracy_score(pred_tags, valid_tags)))
        print()


def main():
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--train-file", type=str, help="Input jsonl file for training", required=True)
    argument_parser.add_argument("--val-file", type=str, help="Input jsonl file for validation", required=True)
    argument_parser.add_argument("--output-dir", type=str, help="Output directory to save model", required=True)
    args = argument_parser.parse_args()
    train(args.train_file, args.val_file, args.output_dir)


if __name__ == "__main__":
    main()
