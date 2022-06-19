from termcolor import colored
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


def translate_sentence(spacy_srg, model, sentence, source, target, device, max_length=50):
    # Load source tokenizer
    # Create tokens using spacy and everything in lower case (which is what our vocab is)
    if type(sentence) == str:
        tokens = [token.text.lower() for token in spacy_srg(sentence)]
    else:
        tokens = [token.lower() for token in sentence]

    # Add <SOS> and <EOS> in beginning and end respectively
    tokens.insert(0, source.init_token)
    tokens.append(source.eos_token)

    # Go through each source token and convert to an index
    text_to_indices = [source.vocab.stoi[token] for token in tokens]

    # Convert to Tensor
    sentence_tensor = torch.LongTensor(text_to_indices).unsqueeze(1).to(device)

    outputs = [target.vocab.stoi["<sos>"]]
    for i in range(max_length):
        trg_tensor = torch.LongTensor(outputs).unsqueeze(1).to(device)

        with torch.no_grad():
            output = model(sentence_tensor, trg_tensor)

        best_guess = output.argmax(2)[-1, :].item()
        outputs.append(best_guess)

        if best_guess == target.vocab.stoi["<eos>"]:
            break

    translated_sentence = [target.vocab.itos[idx] for idx in outputs]
    # remove start token
    return translated_sentence[1:]


def remove_special_notation(sentence: list):
    return [token for token in sentence if token not in ["<unk>", "<eos>", "<sos>"]]


def bleu(spacy_srg, data, model, source, target, device):
    targets = []
    outputs = []

    for example in data:
        src = vars(example)["src"]
        trg = vars(example)["trg"]

        predictions = []

        for _ in range(3):

            prediction = translate_sentence(
                spacy_srg, model, src, source, target, device)
            prediction = remove_special_notation(prediction)
            predictions.append(prediction)

        print(f'  Source (cv): {" ".join(src)}')
        print(colored(f'  Target (en): {" ".join(trg)}', attrs=['bold']))
        print(colored(f'  Predictions (en):', 'blue', attrs=['bold']))
        [print(colored(f'      - {" ".join(prediction)}', 'blue', attrs=['bold'])) 
            for prediction in predictions]
        print("\n")

        targets.append(trg)
        outputs.append(predictions)

    return bleu_score(targets, outputs)


def meteor(spacy_srg, data, model, source, target, device):
    all_meteor_scores = []

    for example in data:
        src = vars(example)["src"]
        trg = vars(example)["trg"]
        predictions = []

        for _ in range(4):
            prediction = translate_sentence(
                spacy_srg, model, src, source, target, device)
            prediction = remove_special_notation(prediction)
            predictions.append(" ".join(prediction))

        all_meteor_scores.append(meteor_score(
            predictions, " ".join(trg)
        ))
        print(f'  Source (cv): {" ".join(src)}')
        print(colored(f'  Target (en): {" ".join(trg)}', attrs=['bold']))
        print(colored(f'  Predictions (en):', 'blue', attrs=['bold']))
        [print(colored(f'      - {prediction}', 'blue', attrs=['bold'])) for prediction in predictions]
        print("\n")

    return sum(all_meteor_scores)/len(all_meteor_scores)


def ter(spacy_srg, test_data, model, source, target, device):
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
            spacy_srg, model, src, source, target, device)[:-1]
        prediction = remove_special_notation(prediction)
        print(f'  Source (cv): {" ".join(src)}')
        print(colored(f'  Target (en): {" ".join(trg)}', attrs=['bold']))
        print(colored(f'  Prediction (en): {" ".join(prediction)}\n', 'blue', attrs=['bold']))
        all_translation_ter += pyter.ter(prediction, trg)
    return all_translation_ter/len(test_data)
    

def save_checkpoint(state, filename):
    print(colored("=> Saving checkpoint", 'cyan'))
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer):
    print(colored("=> Loading checkpoint", "cyan"))
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
