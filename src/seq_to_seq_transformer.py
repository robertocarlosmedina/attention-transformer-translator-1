import spacy

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator

from nltk.tokenize.treebank import TreebankWordDetokenizer

from src.grammer_checker import Grammar_checker
from src.transformer import Transformer
from src.utils import translate_sentence, bleu, meteor, wer_score, gleu,\
    save_checkpoint, load_checkpoint


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

    def train_model(self, test_senteces):
        train_acc, correct_train, target_count = 0, 0, 0
        for epoch in range(self.num_epochs):
            print(f"[Epoch {epoch} / {self.num_epochs}]")

            if self.save_model:
                checkpoint = {
                    "state_dict": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                }
                save_checkpoint(checkpoint)

            self.model.eval()
            print(
                "\n--------------------------\nTEST SENTENCES\n--------------------------\n")
            [print(f"""CV: {sentence}  =>  EN: {self.untokenized_translation(translate_sentence(
                self.spacy_cv, self.model, sentence, self.cv_criole, self.english, self.device,
            ))}""") for sentence in test_senteces]
            print(
                "\n--------------------------------------------------------------------\n")
            # print(f"Translated example sentence: \n {translated_sentence}")
            self.model.train()
            losses = []

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
                train_acc = (correct_train) / target_count

                print(
                    f"Epoch: {epoch}/{self.num_epochs}; Iteration: {batch_index}/{len(self.train_iterator)}; Loss: {loss.item():.4f}; Accuracy: {train_acc:.4f}")

                # Back prop
                loss.backward()
                # Clip to avoid exploding gradient issues, makes sure grads are
                # within a healthy range
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=1)

                # Gradient descent step
                self.optimizer.step()

                 
                self.writer.add_scalar(
                    "Training accuracy", train_acc, global_step=self.step)
                self.step += 1

            mean_loss = sum(losses) / len(losses)
            self.scheduler.step(mean_loss)

    def calculate_blue_score(self):
        """
            BLEU (bilingual evaluation understudy) is an algorithm for evaluating 
            the quality of text which has been machine-translated from one natural 
            language to another.
        """
        # running on entire test data takes a while
        score = bleu(self.untokenized_translation, self.spacy_cv, self.test_data,
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

    def calculate_wer_score(self):
        """
            Word error rate (WER) is a common metric of the performance of a speech 
            recognition or machine translation system. The general difficulty of 
            measuring performance lies in the fact that the recognized word sequence
            can have a different length from the reference word sequence (supposedly 
            the correct one).
        """
        score = wer_score(self.untokenized_translation, self.spacy_cv, self.test_data,
                          self.model, self.cv_criole, self.english, self.device)
        print(f"WER score: {score * 100:.2f}")

    def calculate_gleu_score(self):
        """
            NLP evaluation metric used in Machine Translation tasks
            Suitable for measuring sentence level similarity
            Range: 0 (no match) to 1 (exact match)
        """
        score = gleu(self.untokenized_translation, self.spacy_cv, self.test_data,
                     self.model, self.cv_criole, self.english, self.device)
        print(f"GLEU score: {score * 100:.2f}")

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
