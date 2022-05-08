from seq_to_seq_transformer import Sequence_to_Sequence_Transformer as seq_to_seq_trans

test_list = ["onde k bô ta?", "m t gosta de bô.", "no t bei t lá dia 12.",\
             "m pensa Roberto tava t bem també", "Kes dia lá tava xtud mesm doid."\
             "até cond bô poder."]

seq_to_seq_trans = seq_to_seq_trans()

# seq_to_seq_trans.train_model()
[print(seq_to_seq_trans.translate_sentence(sentence)) for sentence in test_list]