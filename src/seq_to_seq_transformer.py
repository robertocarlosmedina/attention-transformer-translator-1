import math
import spacy

import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator

from nltk.tokenize.treebank import TreebankWordDetokenizer

from src.grammer_checker import Grammar_checker
from src.transformer import Transformer
from src.utils import epoch_time, translate_sentence, bleu, meteor,\
    save_checkpoint, load_checkpoint, ter, count_parameters,\
    epoch_time


class Sequence_to_Sequence_Transformer:

    spacy_cv = spacy.load("pt_core_news_sm")
    spacy_eng = spacy.load("en_core_web_sm")
    grammar = Grammar_checker()

    def __init__(self) -> None:
        self.cv_criole = Field(tokenize=self.tokenize_cv,
                               lower=True, init_token="<sos>", eos_token="<eos>")

        self.english = Field(
            tokenize=self.tokenize_eng, lower=True, init_token="<sos>", eos_token="<eos>"
        )
        self.train_data, self.valid_data, self.test_data = Multi30k.splits(
            exts=(".cv", ".en"), fields=(self.cv_criole, self.english), test="test", 
            path=".data/criolSet"
        )

        # print(self.train_data.examples[122].src, self.train_data.examples[122].trg)

        self.cv_criole.build_vocab(self.train_data, max_size=10000, min_freq=2)
        self.english.build_vocab(self.train_data, max_size=10000, min_freq=2)
        # We're ready to define everything we need for training our Seq2Seq model
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.save_model = True
        # Training hyperparameters
        self.num_epochs = 50
        self.learning_rate = 3e-4
        self.batch_size = 10
        # Model hyperparameters
        self.src_vocab_size = len(self.cv_criole.vocab)
        self.trg_vocab_size = len(self.english.vocab)
        self.embedding_size = 512
        self.num_heads = 8
        self.num_encoder_layers = 3
        self.num_decoder_layers = 3
        self.dropout = 0.05
        self.max_len = 100
        self.forward_expansion = 4
        self.src_pad_idx = self.english.vocab.stoi["<pad>"]
        # Tensorboard to get nice loss plot
        self.writer = SummaryWriter()
        self.step = 0
        # Start the model configurations
        self.starting_model_preparation()

    def tokenize_cv(self, text):
        """
            Exctract all the tokens from the CV sentences
        """
        return [tok.text for tok in self.spacy_cv.tokenizer(text)]

    def tokenize_eng(self, text):
        """
            Exctract all the tokens from the EN sentences
        """
        return [tok.text for tok in self.spacy_eng.tokenizer(text)]

    def starting_model_preparation(self):
        """
            Method to set the transformer configuration trainning hyperparameters
        """
        self.train_iterator, self.valid_iterator, self.test_iterator = BucketIterator.splits(
            (self.train_data, self.valid_data, self.test_data),
            batch_size=self.batch_size,
            sort_within_batch=True,
            sort_key=lambda x: len(x.src),
            device=self.device,
        )

        self.model = Transformer(
            self.embedding_size,
            self.src_vocab_size,
            self.trg_vocab_size,
            self.src_pad_idx,
            self.num_heads,
            self.num_encoder_layers,
            self.num_decoder_layers,
            self.forward_expansion,
            self.dropout,
            self.max_len,
            self.device,
        ).to(self.device)

        self.optimizer = optim.Adam(
            self.model.parameters(), lr=self.learning_rate)

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, factor=0.1, patience=10, verbose=True
        )

        self.pad_idx = self.english.vocab.stoi["<pad>"]
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.pad_idx)

        try:
            load_checkpoint(torch.load("checkpoints/my_checkpoint.pth.tar"),
                            self.model, self.optimizer)
        except:
            pass
    
    def evaluate(self):
        self.model.eval()

        epoch_loss = 0
        train_acc, correct_train, target_count = 0, 0, 0
        len_valid_iterator = len(self.valid_iterator)

        with torch.no_grad():

            for i, batch in enumerate(self.valid_iterator):

                src = batch.src.to(self.device)
                trg = batch.trg.to(self.device)

                output = self.model(src, trg[:-1, :])

                # output = [batch size, trg len - 1, output dim]
                # trg = [batch size, trg len]

                # output_dim = output.shape[-1]

                output = output.reshape(-1, output.shape[-1])
                trg = trg[1:].reshape(-1)

                # output = [batch size * trg len - 1, output dim]
                # trg = [batch size * trg len - 1]

                loss = self.criterion(output, trg)

                _, predicted = torch.max(output.data, 1)
                target_count += trg.size(0)
                correct_train += (trg == predicted).sum().item()
                train_acc += (correct_train) / target_count

                epoch_loss += loss.item()

        return epoch_loss / len_valid_iterator, train_acc / len_valid_iterator

    def train(self):
        self.model.eval()
        self.model.train()

        losses = []
        epoch_loss = 0
        train_acc, correct_train, target_count = 0, 0, 0
        len_iterator = len(self.train_iterator)

        for batch_index, batch in enumerate(self.train_iterator):
            print(f" Training Iteration: {batch_index+1:04}/{len_iterator}")
            # Get input and targets and get to cuda
            inp_data = batch.src.to(self.device)
            target = batch.trg.to(self.device)
            # Forward prop
            output = self.model(inp_data, target[:-1, :])
            # Output is of shape (trg_len, batch_size, output_dim) but Cross Entropy Loss
            # doesn't take input in that form. For example if we have MNIST we want to have
            # output to be: (N, 10) and targets just (N). Here we can view it in a similar
            # way that we have output_words * batch_size that we want to send in into
            # our cost function, so we need to do some reshapin.
            # Let's also remove the start token while we're at it
            output = output.reshape(-1, output.shape[2])
            target = target[1:].reshape(-1)
            self.optimizer.zero_grad()
            loss = self.criterion(output, target)
            losses.append(loss.item())
            _, predicted = torch.max(output.data, 1)
            target_count += target.size(0)
            correct_train += (target == predicted).sum().item()
            train_acc += (correct_train) / target_count

            epoch_loss += loss.item()
            # Back prop
            loss.backward()
            # Clip to avoid exploding gradient issues, makes sure grads are
            # within a healthy range
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=1)
            # Gradient descent step
            self.optimizer.step()
            self.step += 1
            
        mean_loss = sum(losses) / len(losses)
        self.scheduler.step(mean_loss)
    
        return epoch_loss / len_iterator, train_acc / len_iterator

    def show_train_metrics(self, epoch: int, epoch_time: str, train_loss: float, 
        train_accuracy: float, valid_loss: float, valid_accuracy:float) -> None:
        print(f' Epoch: {epoch+1:03}/{self.num_epochs} | Time: {epoch_time}')
        print(
            f' Train Loss: {train_loss:.3f} | Train Acc: {train_accuracy:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(
            f' Val. Loss: {valid_loss:.3f} | Val Acc: {valid_accuracy:.3f} | Val. PPL: {math.exp(valid_loss):7.3f}')
    
    def save_train_metrics(self, epoch: int, train_loss: float, 
            train_accuracy: float, valid_loss: float, valid_accuracy:float) -> None:
        """
            Save the training metrics to be ploted in the tensorboard.
        """
        # All stand alone metrics
        self.writer.add_scalar(
            "Training Loss", train_loss, global_step=epoch)
        self.writer.add_scalar(
            "Training Accuracy", train_accuracy, global_step=epoch)
        self.writer.add_scalar(
            "Validation Loss", valid_loss, global_step=epoch)
        self.writer.add_scalar(
            "Validation Accuracy", valid_accuracy, global_step=epoch)
        
        # Mixing Train Metrics
        self.writer.add_scalars(
            "Training Metrics (Train Loss / Train Accurary)", {
                "Train Loss": train_loss, "Train Accurary": train_accuracy},
            global_step=epoch
        )

        # Mixing Validation Metrics
        self.writer.add_scalars(
            "Training Metrics (Validation Loss / Validation Accurary)", {
                "Validation Loss": valid_loss, "Validation Accuracy": valid_accuracy},
            global_step=epoch
        )
        
        # Mixing Train and Validation Metrics
        self.writer.add_scalars(
            "Training Metrics (Train Loss / Validation Loss)", {
                "Train Loss": train_loss, "Validation Loss": valid_loss},
            global_step=epoch
        )
        self.writer.add_scalars(
            "Training Metrics (Train Accurary / Validation Accuracy)", {
                "Train Accurary": train_accuracy, "Validation Accuracy": valid_accuracy},
            global_step=epoch
        )

    def train_model(self):

        best_valid_loss = float('inf')

        for epoch in range(self.num_epochs):

            start_time = time.time()

            train_loss, train_accuracy = self.train()
            valid_loss, valid_accuracy = self.evaluate()

            end_time = time.time()

            epoch_mins, epoch_secs = epoch_time(start_time, end_time)

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                checkpoint = {
                    "state_dict": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                }
                save_checkpoint(checkpoint, "checkpoints/my_checkpoint.pth.tar")
            self.show_train_metrics(
                epoch, f"{epoch_mins}m {epoch_secs}s", train_loss,
                train_accuracy, valid_loss, valid_accuracy
            )
            self.save_train_metrics(
                epoch, train_loss,
                train_accuracy, valid_loss, valid_accuracy
            )            

    def calculate_blue_score(self):
        """
            BLEU (bilingual evaluation understudy) is an algorithm for evaluating 
            the quality of text which has been machine-translated from one natural 
            language to another.
        """
        # running on entire test data takes a while
        score = bleu(self.spacy_cv, self.test_data,
                     self.model, self.cv_criole, self.english, self.device)
        print(f"Bleu score: {score * 100:.2f}")

    def calculate_meteor_score(self):
        """
            METEOR (Metric for Evaluation of Translation with Explicit ORdering) is 
            a metric for the evaluation of machine translation output. The metric is 
            based on the harmonic mean of unigram precision and recall, with recall 
            weighted higher than precision.
        """
        # running on entire test data takes a while
        score = meteor(self.untokenized_translation, self.spacy_cv, self.test_data,
                       self.model, self.cv_criole, self.english, self.device)
        print(f"Meteor score: {score * 100:.2f}")
    
    def calculate_ter_score(self):
        """
            TER. Translation Error Rate (TER) is a character-based automatic metric for 
            measuring the number of edit operations needed to transform the 
            machine-translated output into a human translated reference.
        """
        score = ter(self.spacy_cv, self.test_data,
                     self.model, self.cv_criole, self.english, self.device)
        print(f"TER score: {score * 100:.2f}")
    
    def count_hyperparameters(self) -> None:
        print(
            f'\nThe model has {count_parameters(self.model):,} trainable parameters')

    def untokenized_translation(self, translated_sentence_list) -> str:
        """
            Method to untokenuze the pedicted translation.
            Returning it on as an str.
        """
        translated_sentence_str = []
        for word in translated_sentence_list:
            if(word != "<eos>" and word != "<unk>"):
                translated_sentence_str.append(word)
        translated_sentence = TreebankWordDetokenizer().detokenize(translated_sentence_str)
        return self.grammar.check_sentence(translated_sentence)

    def translate_sentence(self, sentence) -> str:
        """
            Method that performers the translation and return the prediction.
        """
        translated_sentence_list = translate_sentence(
            self.spacy_cv, self.model, sentence, self.cv_criole, self.english, self.device, max_length=50
        )
        sentence = self.untokenized_translation(translated_sentence_list)
        return sentence
