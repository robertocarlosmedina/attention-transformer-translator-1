import torch
from jiwer import wer
from torchtext.data.metrics import bleu_score
from nltk.translate.gleu_score import sentence_gleu
from nltk.translate.meteor_score import meteor_score


# nltk.download()


def translate_sentence(spacy_cv, model, sentence, cv_creole, english, device, max_length=50):
    # Load cv_creole tokenizer
    # spacy_cv = spacy.load("pt_core_news_lg")

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


def bleu(untokenize_translation, spacy_cv, data, model, cv_creole, english, device):
    targets = []
    outputs = []

    for example in data:
        src = vars(example)["src"]
        trg = vars(example)["trg"]

        prediction = translate_sentence(
            spacy_cv, model, src, cv_creole, english, device)
        prediction = prediction[:-1]  # remove <eos> token
        print(
            f"CV: {untokenize_translation(src)}  =>  EN: {untokenize_translation(prediction)}")

        targets.append([trg])
        outputs.append(prediction)

    return bleu_score(outputs, targets)


def meteor(untokenize_translation, spacy_cv, data, model, cv_creole, english, device):
    all_meteor_scores = []

    for example in data:
        src = vars(example)["src"]
        trg = vars(example)["trg"]
        print(
            f"CV: {untokenize_translation(src)}  =>  EN: {untokenize_translation(trg)}")
        predictions = []

        for _ in range(4):
            prediction = translate_sentence(
                spacy_cv, model, src, cv_creole, english, device)
            prediction = prediction[:-1]  # remove <eos> token
            predictions.append(untokenize_translation(prediction))

        all_meteor_scores.append(meteor_score(
            predictions, untokenize_translation(trg)
        ))

    return sum(all_meteor_scores)/len(all_meteor_scores)


def wer_score(untokenize_translation, spacy_cv, data, model, cv_creole, english, device):
    all_wer_scores = []

    for example in data:
        src = vars(example)["src"]
        trg = vars(example)["trg"]
        print(
            f"CV: {untokenize_translation(src)}  =>  EN: {untokenize_translation(trg)}")
        prediction = translate_sentence(
            spacy_cv, model, src, cv_creole, english, device)
        prediction = prediction[:-1]  # remove <eos> token

        all_wer_scores.append(wer(untokenize_translation(
            prediction), untokenize_translation(trg)))

    return sum(all_wer_scores)/len(all_wer_scores)


def gleu(untokenize_translation, spacy_cv, data, model, cv_creole, english, device):
    all_gleu_scores = []

    for example in data:
        src = vars(example)["src"]
        trg = vars(example)["trg"]
        print(
            f"CV: {untokenize_translation(src)}  =>  EN: {untokenize_translation(trg)}")
        prediction = translate_sentence(
            spacy_cv, model, src, cv_creole, english, device)
        prediction = prediction[:-1]  # remove <eos> token

        all_gleu_scores.append(
            sentence_gleu([untokenize_translation(prediction)], untokenize_translation(trg))
        )

    return sum(all_gleu_scores)/len(all_gleu_scores)


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
