from jp_housing_pred.evaluate import evaluate
from jp_housing_pred.features import OUTPUT_FOLDER as GOLD_FOLDER
from jp_housing_pred.models import GlobalMean, GroupMean
from jp_housing_pred.split import load_split


def main():
    train = load_split(GOLD_FOLDER, "train")
    validation = load_split(GOLD_FOLDER, "validation")
    print(f"Train split: {len(train)} samples")
    print(f"Validation split: {len(validation)} samples")


    models = [GlobalMean(), GroupMean(group="pref"), GroupMean(group="MunicipalityCode")]
    for model in models:
        print(f"Training model: {model.__class__.__name__}")
        print(evaluate(model, train, validation))

if __name__ == "__main__":
    main()