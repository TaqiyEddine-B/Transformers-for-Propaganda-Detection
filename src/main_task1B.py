import logging
import logging.handlers
import os

import pandas as pd
import yaml

import wandb
from models.bmc import BertMultilabelClassifier
from utils import evaluate, read_data_1b, set_seed

logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)


def pipeline(hyper_params:list):

    # create output directories if they don't exist
    OUTPUT_DIR = "output/task1B"

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    wandb.init(
        project="araieval_subtaskB",
        config=hyper_params,
    )

    train_file = "data/task1B_train.jsonl"
    test_file = "data/task1B_dev.jsonl"



    df= pd.read_json(train_file,lines = True)


    unique_labels = set()
    for label_list in df['labels']:
        unique_labels.update(label_list)


    train_texts,train_labels,train_types,trains_ids = read_data_1b(train_file,preprocessing=hyper_params["preprocessing"],unique_labels=unique_labels)
    test_texts,test_labels,test_types,test_ids = read_data_1b(test_file,preprocessing=hyper_params["preprocessing"],unique_labels=unique_labels)

    num_classes = len(unique_labels)

    # instantiate a model
    model_name = hyper_params["model_name"]
    classifier = BertMultilabelClassifier(model_name,num_classes)

    classifier.train(texts = train_texts,
                    labels=train_labels,
                    labels_types=train_types,
                    batch_size =hyper_params['batch_size'],
                    epochs=hyper_params['epochs'],
                    learning_rate=hyper_params['learning_rate'],
                    )

    predicted_labels, probabilities = classifier.predict(test_texts)
    micro_f1, macro_f1 = evaluate(predicted_labels.cpu().tolist(),test_labels,subtask="1A")
    print("micro-F1={:.4f}\tmacro-F1={:.4f}".format(micro_f1, macro_f1))
    # micro_f1, macro_f1 = float(0), float(0)


    run_name= wandb.run.name

    def find_indices(list_to_check, item_to_find):
        return [idx for idx, value in enumerate(list_to_check) if value == item_to_find]


    with open(f"{OUTPUT_DIR}/task1B_output_{run_name}.tsv", "w") as file:
      file.write("id\tlabel\n")
      for (id,pred) in zip(test_ids,predicted_labels.cpu().tolist()):

        r = find_indices(pred, 1)

        label = [list(unique_labels)[item] for item in r]

        file.write(str(id) + "\t" + ",".join(label) + "\n")


        # file.write(str(id)+"\t"+str(label)+"\n")

    return {"micro_f1": micro_f1, "macro_f1": macro_f1}


def launch_experiments():
    with open("exps_1b.yaml", 'r') as stream:
        data_loaded = yaml.safe_load(stream)
    exps =data_loaded['exps_test']

    for exp in exps:
        param = exps[exp]
        print(f'\n** experience : {exp}\n')
        print(param)
        wandb.init(
            project="araieval_subtaskB",
            config =param
        )
        seed = param['seed']
        set_seed(seed)

        metrics = pipeline(param)
        wandb.log(metrics)
        wandb.finish()

# main function
if __name__ == "__main__":
    launch_experiments()
