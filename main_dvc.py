from dog_vs_cat import extract_features, check_feature_exists, train_model, save_model, load_model, test_model
import argparse as ap

if __name__ == "__main__":
    parser = ap.ArgumentParser()
    parser.add_argument("-l", "--load", help="Load a previously saved model")
    parser.add_argument("-t", "--train", help="Train a model with default settings (in dog_vs_cat.py)")
    args = parser.parse_args()
    if args.train:
        extract_features()
        if sum(check_feature_exists()) != 3:
            raise FileExistsError("Missing required features, check correct path")
        model = train_model()
        # save model to default output/store path
        save_model(model)
    elif args.load:
        model = load_model()
        test_model(model)



