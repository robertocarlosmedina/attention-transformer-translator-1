import argparse
import os


arg_pr = argparse.ArgumentParser()

arg_pr.add_argument(
    "-a", "--action", nargs="+", required=True,
    choices=[
        "console", "train", "test_model", "flask_api", "blue_score",
        "meteor_score", "count_parameters", "ter_score"
    ],
    help="Add an action to run this project"
)

arg_pr.add_argument(
    "-s", "--source", required=True,
    choices=[
        "en", "cv"
    ],
    help="Source languague for the translation"
)

arg_pr.add_argument(
    "-t", "--target", required=True,
    choices=[
        "en", "cv"
    ],
    help="Target languague for the translation"
)

args = vars(arg_pr.parse_args())


from src.seq_to_seq_transformer import Sequence_to_Sequence_Transformer as seq_to_seq_trans
# from src.flask_api import Resfull_API


transformer = seq_to_seq_trans(args["source"], args["target"])


def execute_main_actions():
    """
        Function the execute the action according to the users need
    """
    actions_dict = {
        "console": transformer.console_model_test,
        "train": transformer.train_model,
        "test_model": transformer.test_model,
        # "flask_api": Resfull_API.start,
        "blue_score": transformer.calculate_blue_score,
        "meteor_score": transformer.calculate_meteor_score,
        "count_parameters": transformer.count_hyperparameters,
        "ter_score": transformer.calculate_ter_score
    }

    [actions_dict[action]() for action in args["action"]]


if __name__ == "__main__":
    execute_main_actions()
