import os
import json
import argparse
import logging
from tqdm import trange
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
EPOCHS = 1
MAX_GRAD_NORM = 1.0
LEARNING_RATE = 3e-5
EPSILON = 1e-8
FEED_FORWARD_DROPOUT = 0.3
FULL_FINE_TUNING = True


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

    # labels is a vector of zeros except in the position of the value
    labels = [0] * value_token_position + [1] + [0] * (len(pair_input_ids) - value_token_position - 1)

    assert len(pair_input_ids) == len(token_type_ids)
    assert len(token_type_ids) == len(labels)

    assert pair_input_ids[value_token_position] == value_token_id

    return dict(
        input_ids=pair_input_ids,
        token_type_ids=token_type_ids,
        labels=labels
    )


class BertForValueExtraction(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = 2
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.ff = nn.Sequential(nn.Linear(config.hidden_size, 100), nn.ReLU(), nn.Dropout(FEED_FORWARD_DROPOUT),
                                nn.Linear(100, 50), nn.ReLU(), nn.Dropout(FEED_FORWARD_DROPOUT),
                                nn.Linear(50, 1),
                                nn.Linear(1, 1))

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
    ):
        outputs = self.bert(
            input_ids,
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
            # get probabilities out of logits
            predicted_probs = torch.softmax(logits, 1)

            # negative log likelihood loss
            loss = torch.nn.functional.nll_loss(predicted_probs, labels.argmax(1).unsqueeze(-1))

            outputs = (loss,) + outputs

        return outputs  # (loss), scores


def _train_model(device, model, optimizer, scheduler, train_data_loader, val_data_loader):
    for epoch_number in trange(EPOCHS, desc="Epoch"):
        # put the model into training mode.
        model.train()
        # reset the total loss for this epoch.
        total_loss = 0

        # training loop
        for step, batch in enumerate(train_data_loader):
            # add batch to gpu if available
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_token_type_ids, b_labels = batch
            # always clear any previously calculated gradients before performing a backward pass.
            model.zero_grad()
            # forward pass
            # This will return the loss (together with the model output) because we have provided the `labels`.
            outputs = model(
                b_input_ids,
                token_type_ids=b_token_type_ids,
                labels=b_labels
            )
            # get the loss
            loss = outputs[0]
            # Perform a backward pass to calculate the gradients.
            loss.backward()
            # track train loss
            total_loss += loss.item()
            # clip the norm of the gradient
            # this is to help prevent the "exploding gradients" problem.
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=MAX_GRAD_NORM)
            # update parameters
            optimizer.step()
            # Update the learning rate.
            scheduler.step()

        # calculate the average loss over the training data.
        avg_train_loss = total_loss / len(train_data_loader)
        logging.info(f"Average train loss on epoch {epoch_number}: {avg_train_loss}")

        # Put the model into evaluation mode
        model.eval()
        # reset the validation loss for this epoch
        eval_loss, eval_accuracy = 0, 0
        predictions, true_labels = [], []
        for batch in val_data_loader:
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_token_type_ids, b_labels = batch

            # telling the model not to compute or store gradients, saving memory and speeding up validation
            with torch.no_grad():
                # forward pass, calculate logit predictions.
                # this will return the logits rather than the loss because we have not provided labels.
                outputs = model(
                    b_input_ids,
                    token_type_ids=b_token_type_ids,
                    labels=b_labels
                )
            # move logits and labels to CPU
            logits = outputs[1].detach().cpu().numpy()

            # calculate the accuracy for this batch of test sentences.
            eval_loss += outputs[0].mean().item()
            predictions.extend(logits.argmax(1))
            true_labels.extend(b_labels.unsqueeze(-1).argmax(1).numpy())

        eval_loss = eval_loss / len(val_data_loader)

        logging.info(f"Validation loss on epoch {epoch_number + 1}: {eval_loss}")
        logging.info(f"Validation Accuracy on epoch {epoch_number + 1}: {accuracy_score(predictions, true_labels)[0]}")
        logging.info("\n")


def train(train_file: str, val_file: str, output_dir: str):
    # create output folder if not exists
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
    train_labels = torch.tensor([train_sample["labels"] for train_sample in train_encoded])

    # convert validation samples to tensors
    val_inputs = torch.tensor([val_sample["input_ids"] for val_sample in val_encoded])
    val_token_type_ids = torch.tensor([val_sample["token_type_ids"] for val_sample in val_encoded])
    val_labels = torch.tensor([val_sample["labels"] for val_sample in val_encoded])

    # initialize the model
    model = BertForValueExtraction.from_pretrained("bert-base-uncased")

    # create training data loader
    train_data = TensorDataset(train_inputs, train_token_type_ids, train_labels)
    train_sampler = RandomSampler(train_data)
    train_data_loader = DataLoader(train_data, sampler=train_sampler, batch_size=BATCH_SIZE)

    # create validation data loader
    val_data = TensorDataset(val_inputs, val_token_type_ids, val_labels)
    val_sampler = SequentialSampler(val_data)
    val_data_loader = DataLoader(val_data, sampler=val_sampler, batch_size=BATCH_SIZE)

    # once can choose to full fine tune BERT or just freeze BERT parameters and train only the feed-forward
    if FULL_FINE_TUNING:
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'gamma', 'beta']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.0}
        ]
    else:
        param_optimizer = list(model.ff.named_parameters())
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
        num_training_steps=total_steps,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # main method to train model
    _train_model(device, model, optimizer, scheduler, train_data_loader, val_data_loader)

    logging.info("Done.")


def main():
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--train-file", type=str, help="Input jsonl file for training", required=True)
    argument_parser.add_argument("--val-file", type=str, help="Input jsonl file for validation", required=True)
    argument_parser.add_argument("--output-dir", type=str, help="Output directory to save model", required=True)
    args = argument_parser.parse_args()
    train(args.train_file, args.val_file, args.output_dir)


if __name__ == "__main__":
    main()
