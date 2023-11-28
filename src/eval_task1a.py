
import torch


if __name__ == "__main__":

    run_name=""
    model_path= f"output/1a_train/task1A_model_{run_name}.pth"

    model = torch.load(model_path)
    model.eval()  # Set dropout and batch normalization layers to evaluation mode before running inference

