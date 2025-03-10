import torch
import numpy as np
import random
import argparse
from torch.utils.data import DataLoader
import utils
import model
import pandas as pd
import zipfile
import os

# Set seed for reproducibility
def setup_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Image Classification")

    parser.add_argument('--test_path', type=str, default="../data/test/", help='Training Dataset Path')
    parser.add_argument('--ckpt_path', type=str, default='../checkpoints/best.pt', help='Path to checkpoint file.')
    parser.add_argument('--img_path', type=str, default='../img/test/', help='Path to save the image.')
    parser.add_argument('--csv_path', type=str, default='../prediction.csv', help='Path to save the prediction csv.')
    parser.add_argument('--zip_path', type=str, default='../submission.zip', help='Path to save the submission zip.')
    parser.add_argument('--device', type=str, default="cuda:0", help='Which device the training is on.')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of worker')
    parser.add_argument('--bs', type=int, default=1, help='Batch size for training.')
    parser.add_argument('--num_classes', type=int, default=100, help='Number of classes.')
    args = parser.parse_args()

    # set seed
    setup_seed(118)

    test_dataset = utils.TestDataset(data_dir=args.test_path)
    test_loader = DataLoader(test_dataset, batch_size=args.bs, shuffle=False)

    myModel = model.Resnet_50(num_classes=args.num_classes)
    myModel.to(args.device)
    myModel.eval()
    myModel.load_state_dict(torch.load(args.ckpt_path))

    pred_list = []
    for i, (img, img_name) in enumerate(test_loader):
        img = img.to(args.device)
        output = myModel(img)
        _, pred = torch.max(output, 1)
        print(f"Image: {os.path.splitext(img_name[0])[0]}, Prediction: {pred.item()}")
        pred_list.append([os.path.splitext(img_name[0])[0], pred.item()])
    
    # 轉成 DataFrame 並儲存 CSV
    df = pd.DataFrame(pred_list, columns=["image_name", "pred_label"])
    csv_path = args.csv_path
    df.to_csv(csv_path, index=False)

    with zipfile.ZipFile(args.zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(csv_path)
