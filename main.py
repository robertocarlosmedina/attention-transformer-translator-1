import argparse

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


from src.flask_api import Resfull_API
from src.seq_to_seq_transformer import Sequence_to_Sequence_Transformer as seq_to_seq_trans


def get_test_data(start_index=0, end_index=10) -> list:
    test_list = []
    test_file_reader = open(".data/multi30k/test.cv", "r")
    for text in test_file_reader.readlines()[start_index:end_index]:
        test_list.append(text.strip())
    return test_list


transformer = seq_to_seq_trans()
test_list = get_test_data()


def execute_console_translations() -> None:
    while True:
        cv_sentence = str(input("CV phrase: "))
        print(
            f"EN Translation: {transformer.translate_sentence(cv_sentence)}")


def execute_single_test() -> None:
    for i in range(5):
        print(f"\nITERATION {i}:\n")
        [print(f"{sentence}  =>  {transformer.translate_sentence(sentence)}")
         for sentence in test_list]


def train_transformer_model() -> None:
    transformer.train_model(test_list)
    

def execute_main_actions():

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

    for action in args["action"]:
        actions_dict[action]()


if __name__ == "__main__":
    execute_main_actions()
