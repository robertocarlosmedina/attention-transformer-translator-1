from tqdm import tqdm
from termcolor import colored
import math
from nltk.tokenize.treebank import TreebankWordDetokenizer
import spacy
import os

from src.grammar_checker import Grammar_checker
from src.transformer import Transformer
from src.utils import epoch_time, translate_sentence, bleu, meteor,\
    save_checkpoint, load_checkpoint, ter, epoch_time, remove_special_notation, \
    check_dataset

import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator


class Sequence_to_Sequence_Transformer:

    spacy_models = {
        "en": spacy.load("en_core_web_sm"),
        "pt": spacy.load("pt_core_news_sm"),
        "cv": spacy.load("pt_core_news_sm"),
    }
    grammar = Grammar_checker()

    def __init__(self, source_languague: str, target_languague: str) -> None:
        self.source_languague, self.target_languague = source_languague, target_languague
        check_dataset()
        self.source = Field(tokenize=self.tokenize_src,
                               lower=True, init_token="<sos>", eos_token="<eos>")

        self.target = Field(
            tokenize=self.tokenize_trg, lower=True, init_token="<sos>", eos_token="<eos>"
        )
        self.train_data, self.valid_data, self.test_data = Multi30k.splits(
            exts=(f".{self.source_languague}", f".{target_languague}"), 
            fields=(self.source, self.target), test="test", 
            path=".data/crioleSet"
        )

        # print(self.train_data.examples[122].src, self.train_data.examples[122].trg)
        # exit()

        self.source.build_vocab(self.train_data, max_size=10000, min_freq=2)
        self.target.build_vocab(self.train_data, max_size=10000, min_freq=2)
        # We're ready to define everything we need for training our Seq2Seq model
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.save_model = True
        # Training hyperparameters
        self.num_epochs = 100
        self.learning_rate = 3e-4
        self.batch_size = 22
        # Model hyperparameters
        self.src_vocab_size = len(self.source.vocab)
        self.trg_vocab_size = len(self.target.vocab)
        self.embedding_size = 512
        self.num_heads = 8
        self.num_encoder_layers = 3
        self.num_decoder_layers = 3
        self.dropout = 0.05
        self.max_len = 100
        self.forward_expansion = 4
        self.src_pad_idx = self.target.vocab.stoi["<pad>"]
        # Tensorboard to get nice loss plot
        self.writer = SummaryWriter()
        self.step = 0
        # Start the model configurations
        self.starting_model_preparation()

    def tokenize_src(self, text):
        """
            Exctract all the tokens from the CV sentences
        """
        return [tok.text for tok in self.spacy_models[self.source_languague].tokenizer(text)]

    def tokenize_trg(self, text):
        """
            Exctract all the tokens from the EN sentences
        """
        return [tok.text for tok in self.spacy_models[self.target_languague].tokenizer(text)]

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

        self.pad_idx = self.target.vocab.stoi["<pad>"]
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.pad_idx)

        try:
            load_checkpoint(
                torch.load(
                    f"checkpoints/transformer1-{self.source_languague}-{self.target_languague}.pth.tar"
                ),
                self.model, self.optimizer
            )
        except:
            print(colored("=> No checkpoint to Load", "red"))
    
    def get_test_data(self) -> list:
        return [(test.src, test.trg) for test in self.test_data.examples[0:20]]

    def evaluate(self, epoch: int, progress_bar: object):
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

                progress_bar.set_postfix(
                    epoch=f" {epoch}, val loss= {round(epoch_loss / (i + 1), 4)}, val accu: {train_acc / (i + 1):.4f}", 
                    refresh=True)
                progress_bar.update()

        return epoch_loss / len_valid_iterator, train_acc / len_valid_iterator

    def train(self, epoch: int, progress_bar: object):
        self.model.eval()
        self.model.train()

        losses = []
        epoch_loss = 0
        train_acc, correct_train, target_count = 0, 0, 0
        len_iterator = len(self.train_iterator)

        for batch_index, batch in enumerate(self.train_iterator):
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
            progress_bar.set_postfix(
                epoch=f" {epoch}, train loss= {round(epoch_loss / (batch_index + 1), 4)}, train accu: {train_acc / (batch_index + 1):.4f}", 
                refresh=True)
            progress_bar.update()
            
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
            f"Training Loss ({self.source_languague}-{self.target_languague})", 
            train_loss, global_step=epoch)
        self.writer.add_scalar(
            f"Training Accuracy ({self.source_languague}-{self.target_languague})", 
            train_accuracy, global_step=epoch)
        self.writer.add_scalar(
            f"Validation Loss ({self.source_languague}-{self.target_languague})", 
            valid_loss, global_step=epoch)
        self.writer.add_scalar(
            f"Validation Accuracy ({self.source_languague}-{self.target_languague})", 
            valid_accuracy, global_step=epoch)
        
        # Mixing Train Metrics
        self.writer.add_scalars(
            f"Training Loss & Accurary ({self.source_languague}-{self.target_languague})", 
            {"Train Loss": train_loss, "Train Accurary": train_accuracy},
            global_step=epoch
        )

        # Mixing Validation Metrics
        self.writer.add_scalars(
            f"Validation Loss & Accurary  ({self.source_languague}-{self.target_languague})", 
            {"Validation Loss": valid_loss, "Validation Accuracy": valid_accuracy},
            global_step=epoch
        )
        
        # Mixing Train and Validation Metrics
        self.writer.add_scalars(
            f"Train Loss & Validation Loss ({self.source_languague}-{self.target_languague})", 
            {"Train Loss": train_loss, "Validation Loss": valid_loss},
            global_step=epoch
        )
        self.writer.add_scalars(
            f"Train Accurary & Validation Accuracy ({self.source_languague}-{self.target_languague})",
            {"Train Accurary": train_accuracy, "Validation Accuracy": valid_accuracy},
            global_step=epoch
        )

    def train_model(self):

        best_valid_loss = float('inf')

        for epoch in range(self.num_epochs):
            progress_bar = tqdm(
                total=len(self.train_iterator)+len(self.valid_iterator), 
                bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}', unit=' batches', ncols=200
            )

            start_time = time.time()

            train_loss, train_accuracy = self.train(epoch + 1, progress_bar)
            valid_loss, valid_accuracy = self.evaluate(epoch + 1, progress_bar)

            end_time = time.time()

            epoch_mins, epoch_secs = epoch_time(start_time, end_time)

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                checkpoint = {
                    "state_dict": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                }
                save_checkpoint(
                    checkpoint, 
                    f"checkpoints/transformer1-{self.source_languague}-{self.target_languague}.pth.tar"
                )
            self.show_train_metrics(
                epoch + 1, f"{epoch_mins}m {epoch_secs}s", train_loss,
                train_accuracy, valid_loss, valid_accuracy
            )
            self.save_train_metrics(
                epoch + 1, train_loss,
                train_accuracy, valid_loss, valid_accuracy
            )            

    def console_model_test(self) -> None:
        os.system("clear")
        print("\n                     CV Creole Translator ")
        print("-------------------------------------------------------------\n")
        while True:
            Sentence = str(input(f'  Sentence ({self.source_languague}): '))
            translation = self.translate_sentence(Sentence)

            print(colored(f'  Predicted ({self.target_languague}): {translation}\n', 'blue', attrs=['bold']))

    def calculate_blue_score(self):
        """
            BLEU (bilingual evaluation understudy) is an algorithm for evaluating 
            the quality of text which has been machine-translated from one natural 
            language to another.
        """
        # running on entire test data takes a while
        score = bleu(self.spacy_models[self.source_languague], self.test_data,
                     self.model, self.source, self.target, self.device)
        print(colored(f"==> TER score: {score * 100:.2f}\n", 'blue'))

    def calculate_meteor_score(self):
        """
            METEOR (Metric for Evaluation of Translation with Explicit ORdering) is 
            a metric for the evaluation of machine translation output. The metric is 
            based on the harmonic mean of unigram precision and recall, with recall 
            weighted higher than precision.
        """
        # running on entire test data takes a while
        score = meteor(self.spacy_models[self.source_languague], self.test_data,
                       self.model, self.source, self.target, self.device)
        print(colored(f"==> Meteor score: {score * 100:.2f}\n", 'blue'))
    
    def calculate_ter_score(self):
        """
            TER. Translation Error Rate (TER) is a character-based automatic metric for 
            measuring the number of edit operations needed to transform the 
            machine-translated output into a human translated reference.
        """
        score = ter(self.spacy_models[self.source_languague], self.test_data,
                     self.model, self.source, self.target, self.device)
        print(colored(f"==> TER score: {score * 100:.2f}\n", 'blue'))
    
    def count_hyperparameters(self) -> None:
        total_parameters =  sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(colored(f'\n==> The model has {total_parameters:,} trainable parameters\n', 'blue'))

    def untokenize_sentence(self, translated_sentence: list) -> str:
        """
            Method to untokenuze the pedicted translation.
            Returning it on as an str.
        """
        translated_sentence = remove_special_notation(translated_sentence)
        if self.source_languague == "cv":
            translated_sentence = TreebankWordDetokenizer().detokenize(translated_sentence)
            return self.grammar.check_sentence(translated_sentence)

        return " ".join(translated_sentence)

    def translate_sentence(self, sentence: str) -> str:
        """
            Method that performers the translation and return the prediction.
        """
        translated_sentence_list = translate_sentence(
            self.spacy_models[self.source_languague], self.model, sentence, self.source, self.target, self.device, max_length=50
        )
        sentence = self.untokenize_sentence(translated_sentence_list)
        return sentence

    def test_model(self) -> None:
        test_data = self.get_test_data()
        os.system("clear")
        print("\n                  CV Creole Translator Test ")
        print("-------------------------------------------------------------\n")
        for data_tuple in test_data:
            src, trg = " ".join(
                data_tuple[0]), " ".join(data_tuple[1])
            translation = self.translate_sentence(src)
            print(f'  Source ({self.source_languague}): {src}')
            print(colored(f'  Target ({self.target_languague}): {trg}', attrs=['bold']))
            print(
                colored(f'  Predicted ({self.target_languague}): {translation}\n', 'blue', attrs=['bold'])
            )
