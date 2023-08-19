import sys
import os.path
import config
from train.trainer import GptTrainer


def main(data_file: str, dump_folder: str, num_train_epochs: int, mode: str):
    gpt_trainer = GptTrainer(config = {
        "model_path": config.MODEL_PATH,
        "data_file": data_file,
        "batch_size": config.BATCH_SIZE,
        "output_folder": config.OUTPUT_FOLDER,
        "context_size": config.CONTEXT_SIZE,
        "test_part": config.TEST_PART,
        "mode": mode 
        })
    gpt_trainer.train(num_train_epochs)
    gpt_trainer.save_model(dump_folder)



if __name__ == '__main__':
    try:
        data_file = sys.argv[1]
        if not os.path.isfile(data_file):
            raise Exception(f"Data file {data_file} not found or corrupted")
        dump_folder = sys.argv[2]
        num_train_epochs = int(sys.argv[3])
        if len(sys.argv) > 4:
            mode = sys.argv[4]
        else:
            mode = None
    except:
        raise Exception("Corrupted call, for example: python train.py temp/books.data finetuned_model/ 10 lora")
    main(data_file, dump_folder, num_train_epochs, mode)
