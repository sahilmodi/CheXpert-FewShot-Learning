from load_data import load_data
from train import *
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Variational Few Shot Learning")
    parser.add_argument("--dataset", type=str, default="Omniglot", choices=["Omniglot", "CheXpert"])
    parser.add_argument("--data-root", type=str, default="/home/koyejolab/Omniglot")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--num-epochs", type=int, default=10)
    parser.add_argument("--num-shots", type=int, default=5) ## UPDATE?
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    args = parser.parse_args()

    train_loader, test_loader = load_data(args.dataset, args.data_root, args.batch_size, args.num_workers)
    feature_extractor = train_feature_extractor(args.device, train_loader, lr=args.learning_rate, num_epochs=args.num_epochs)
    test_feature_extractor(feature_extractor, args.device, test_loader)
    # features = extract_features(feature_extractor, args.device, test_loader)

    # print("EXTRACTED FEATURES")
    # print("------------------")
    # print(features)