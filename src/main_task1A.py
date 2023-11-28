import logging
import logging.handlers

import wandb
import yaml
from bbc import BertBinaryClassifier
from utils import evaluate, read_data_1a, set_seed,save_output_file


def pipeline(hyper_params:list):

    train_file = "data/task1A_train.jsonl"

    test_file = "data/task1A_dev.jsonl"
    test_file2 = "data/task1A_test.jsonl"

    train_labels,train_texts,train_types,_ = read_data_1a(train_file,hyper_params['preprocessing'])
    test_labels,test_texts,test_types,test_ids = read_data_1a(test_file,hyper_params['preprocessing'])
    test_labels2,test_texts2,test_types2,test_ids2 = read_data_1a(test_file2,hyper_params['preprocessing'])


    model_path = hyper_params['model_name']
    classifier = BertBinaryClassifier(model_path,archi=hyper_params["model_archi"])

    classifier.train(train_texts,
                     train_labels,
                     train_types,
                     batch_size =hyper_params['batch_size'],
                     epochs=hyper_params['epochs'],
                     learning_rate=hyper_params['learning_rate'],
                     loss=hyper_params['loss_type']
                    )
    # file1 of test
    # prediction
    predicted_labels, _ = classifier.predict(test_texts,test_types)
    pred_labels = predicted_labels.cpu().tolist()

    # evaluation
    micro_f1, macro_f1 = evaluate(pred_labels,test_labels,subtask="1A")
    # micro_f1, macro_f1 = float(0), float(0)

    # file2 of test
    # prediction
    predicted_labels2, _ = classifier.predict(test_texts2,test_types2)
    pred_labels2 = predicted_labels2.cpu().tolist()

    # evaluation
    micro_f1_2, macro_f1_2 = evaluate(pred_labels2,test_labels2,subtask="1A")


    run_name= wandb.run.name

    # model_path= f"output/1a_train/task1A_model_{run_name}.pth"
    # torch.save(classifier, 'model.pth')


    result_path = f"output/1a_test/test1/task1A_output_{run_name}.tsv"
    result_path2 = f"output/1a_test/test2/task1A_output_{run_name}.tsv"

    save_output_file(result_path,test_ids,pred_labels)
    save_output_file(result_path2,test_ids2,pred_labels2)

    print("micro-F1={:.4f}\tmacro-F1={:.4f}".format(micro_f1, macro_f1))
    print("micro-F1_2={:.4f}\tmacro-F1_2={:.4f}".format(micro_f1_2, macro_f1_2))


    return {'micro_f1': micro_f1, 'macro_f1': macro_f1,'micro_f1_2': micro_f1_2, 'macro_f1_2': macro_f1_2}

if __name__ == "__main__":

    logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)


    with open("exps_1a.yaml", 'r') as stream:
        data_loaded = yaml.safe_load(stream)


    exps =data_loaded['exps2']

    for exp in exps:
        param = exps[exp]
        print(f'\n** experience : {exp}\n')
        print(param)
        wandb.init(
            project="araieval_task1A",
            config =param
        )

        set_seed(param['seed'])
        metrics = pipeline(param)
        wandb.log(metrics)

        wandb.finish()
