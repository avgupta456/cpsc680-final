from src.vanilla.argparser import get_args, parse_args

if __name__ == "__main__":
    args = get_args()
    (
        debug,
        dataset,
        dataset_name,
        model,
        train_model,
        target_name,
        optimizer,
        epochs,
    ) = parse_args(args)

    train_model(model, dataset_name, dataset, target_name, optimizer, epochs, debug)
