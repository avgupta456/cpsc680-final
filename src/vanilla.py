from src.argparser import get_args, parse_args

if __name__ == "__main__":
    args = get_args()
    (
        model,
        train_model,
        dataset_name,
        dataset,
        optimizer,
        epochs,
        debug,
    ) = parse_args(args)

    train_model(model, dataset_name, dataset, optimizer, epochs, debug)
