# import os
# import argparse
from src.flask_api import Resfull_API
from src.seq_to_seq_transformer import Sequence_to_Sequence_Transformer as seq_to_seq_trans

# arg_pr = argparse.ArgumentParser()
# arg_pr.add_argument("-a", "--action", required=True,
#    help="Add an action to run this project")
# arg_pr.add_argument("-m", "--mode", required=False,
#    default="terminal",
#    help="Add the firts mode option to run this project")
# args = vars(arg_pr.parse_args())
mode = "api"

test_list = [
    "ondê ke bô ta?", "mim ene sebê.", "M te fliz.",\
    "M tite bei p xcolá", "m te xpêra.",\
    "m tite andá.", "m te bei xpiá.", "sodad de bô.",\
    "manera?", "nos terra.", "mim ê de Santo Antão", \
    "M oia dos psoa.", "Tava te pensa n bô", "iss foi condê?", \
    "Talvez porkê nhe irmá ê advogada, agoh um kris també", \
    "M tava gosta de oiob"
]

def execute_console_translations() -> None:
    transformer = seq_to_seq_trans()
    while True:
        cv_sentence = str(input("CV phrase: "))
        print(f"EN Translation: {transformer.translate_sentence(cv_sentence)}")


def run_translation_api() -> None:
    Resfull_API.start()


def execute_single_test() -> None:
    transformer = seq_to_seq_trans()
    [print(f"{sentence}  =>  {transformer.translate_sentence(sentence)}") for sentence in test_list]


def train_the_translation_model() -> None:
    transformer = seq_to_seq_trans()
    transformer.train_model(test_list)

if mode == "console":
    execute_console_translations()
elif mode == "train":
    train_the_translation_model()
elif mode == "test":
    execute_single_test()
elif mode == "api":
    run_translation_api()
