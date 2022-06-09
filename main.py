import argparse
import os


arg_pr = argparse.ArgumentParser()

arg_pr.add_argument(
    "-a", "--action", nargs="+", required=True,
    choices=[
        "console", "train", "test", "api", "blue_score",
        "meteor_score", "wer_score", "gleu_score"
    ],
    help="Add an action to run this project"
)
args = vars(arg_pr.parse_args())


from src.seq_to_seq_transformer import Sequence_to_Sequence_Transformer as seq_to_seq_trans
from src.flask_api import Resfull_API


def get_test_data(start_index=0, end_index=10) -> list:
    cv_test_list, en_test_list = [], []
    cv_test_file_reader = open(".data/criolSet/test.cv", "r")
    en_test_file_reader = open(".data/criolSet/test.en", "r")
    [cv_test_list.append(text.strip()) for text in cv_test_file_reader.readlines()
        [start_index:end_index]]
    [en_test_list.append(text.strip()) for text in en_test_file_reader.readlines()
        [start_index:end_index]]

    return [(cv, en) for cv, en in zip(cv_test_list, en_test_list)]


transformer = seq_to_seq_trans()
test_list = get_test_data()


def execute_console_translations() -> None:
    os.system("clear")
    print("\n                     CV Creole Translator ")
    print("-------------------------------------------------------------\n")
    while True:
        cv_sentence = str(input("  CV phrase: "))
        print(
            f"  EN Translation: {transformer.translate_sentence(cv_sentence)}\n"
        )


def execute_single_test() -> None:
    for i in range(5):
        print(f"\nITERATION {i}:\n")
        [print(f"{sentence}  =>  {transformer.translate_sentence(sentence)}")
            for sentence in test_list[0]]


def train_transformer_model() -> None:
    transformer.train_model(test_list[0])


def execute_main_actions():
    """
        Function the execute the action according to the users need
    """
    actions_dict = {
        "console": execute_console_translations,
        "train": train_transformer_model,
        "test": execute_single_test,
        "api": Resfull_API.start,
        "blue_score": transformer.calculate_blue_score,
        "meteor_score": transformer.calculate_meteor_score,
        "wer_score": transformer.calculate_wer_score,
        "gleu_score": transformer.calculate_gleu_score
    }

    [actions_dict[action]() for action in args["action"]]


if __name__ == "__main__":
    execute_main_actions()
