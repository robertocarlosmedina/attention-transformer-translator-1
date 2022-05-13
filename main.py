import argparse
from seq_to_seq_transformer import Sequence_to_Sequence_Transformer as seq_to_seq_trans

# arg_pr = argparse.ArgumentParser()
# arg_pr.add_argument("-a", "--action", required=True,
#    help="Add an action to run this project")
# arg_pr.add_argument("-m", "--mode", required=False,
#    default="terminal",
#    help="Add the firts mode option to run this project")
# args = vars(arg_pr.parse_args())


test_list = ["ondê ke bô ta?", "mim ene sebê.", "M te fliz.",\
             "M tite bei p xcolá", "m te xpêra.",\
             "m tite andá.", "m te bei xpiá.", "sodad de bô.",\
             "manera?", "nos terra.", "mim ê de Santo Antão", \
             "M oia dos psoa.", "Tava te pensa n bô", "iss foi condê?", \
             "Talvez porkê nhe irmá ê advogada, agoh um kris també", \
             "M tava gosta de oiob, intressant"]


seq_to_seq_trans = seq_to_seq_trans()

# seq_to_seq_trans.train_model()
[print(f"{sentence}  =>  {seq_to_seq_trans.translate_sentence(sentence)}") for sentence in test_list]