from nltk.translate.meteor_score import meteor_score
import pyter

import torch
from torchtext.data.metrics import bleu_score


# If it's needed to dowload the nltk packages
# nltk.download()

PAD_TOKEN = '<pad>'
SOS_TOKEN = '<s>'
EOS_TOKEN = '</s>'
UNK_TOKEN = '<unk>'


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def translate_sentence(spacy_cv, model, sentence, cv_creole, english, device, max_length=50):
    # Load cv_creole tokenizer
    # Create tokens using spacy and everything in lower case (which is what our vocab is)
    if type(sentence) == str:
        tokens = [token.text.lower() for token in spacy_cv(sentence)]
    else:
        tokens = [token.lower() for token in sentence]

    # Add <SOS> and <EOS> in beginning and end respectively
    tokens.insert(0, cv_creole.init_token)
    tokens.append(cv_creole.eos_token)

    # Go through each cv_creole token and convert to an index
    text_to_indices = [cv_creole.vocab.stoi[token] for token in tokens]

    # Convert to Tensor
    sentence_tensor = torch.LongTensor(text_to_indices).unsqueeze(1).to(device)
    attention_scores = torch.zeros(max_length, 1, len(text_to_indices)).to(device)

    outputs = [english.vocab.stoi["<sos>"]]
    for i in range(max_length):
        trg_tensor = torch.LongTensor(outputs).unsqueeze(1).to(device)

        with torch.no_grad():
            output = model(sentence_tensor, trg_tensor)

        best_guess = output.argmax(2)[-1, :].item()
        outputs.append(best_guess)

        if best_guess == english.vocab.stoi["<eos>"]:
            break

    translated_sentence = [english.vocab.itos[idx] for idx in outputs]
    # remove start token
    return translated_sentence[1:]


def bleu(spacy_cv, data, model, cv_creole, english, device):
    targets = []
    outputs = []

    for example in data:
        src = vars(example)["src"]
        trg = vars(example)["trg"]

        predictions = []

        for _ in range(3):

            prediction = translate_sentence(
                spacy_cv, model, src, cv_creole, english, device)
            predictions.append(prediction[:-1]) # remove <eos> token

        print(f'  Source (cv): {" ".join(src)}')
        print(f'  Target (en): {trg}')
        print(f'  Predictions (en):')
        [print(f'      - {prediction}') for prediction in predictions]
        print("\n")

        targets.append(trg)
        outputs.append(predictions)

    return bleu_score(targets, outputs)


def meteor(untokenize_translation, spacy_cv, data, model, cv_creole, english, device):
    all_meteor_scores = []

    for example in data:
        src = vars(example)["src"]
        trg = vars(example)["trg"]
        predictions = []

        for _ in range(4):
            prediction = translate_sentence(
                spacy_cv, model, src, cv_creole, english, device)
            prediction = prediction[:-1]  # remove <eos> token
            predictions.append(untokenize_translation(prediction))

        all_meteor_scores.append(meteor_score(
            predictions, untokenize_translation(trg)
        ))
        print(f'  Source (cv): {" ".join(src)}')
        print(f'  Target (en): {untokenize_translation(trg)}')
        print(f'  Predictions (en): ')
        [print(f'      - {prediction}') for prediction in predictions]
        print("\n")

    return sum(all_meteor_scores)/len(all_meteor_scores)


def ter(spacy_cv, test_data, model, cv_creole, english, device):
    """
        TER. Translation Error Rate (TER) is a character-based automatic metric for 
        measuring the number of edit operations needed to transform the 
        machine-translated output into a human translated reference.
    """
    all_translation_ter = 0
    for example in test_data:
        src = vars(example)["src"]
        trg = vars(example)["trg"]
        prediction = translate_sentence(
            spacy_cv, model, src, cv_creole, english, device)[:-1]
        print(f'  Source (cv): {" ".join(src)}')
        print(f'  Target (en): {" ".join(trg)}')
        print(f'  Predictions (en): {" ".join(prediction)}\n')
        all_translation_ter += pyter.ter(prediction, trg)
    return all_translation_ter/len(test_data)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
    

def save_checkpoint(state, filename="checkpoints/my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
