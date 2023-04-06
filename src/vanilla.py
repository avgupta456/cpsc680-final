from src.argparser import get_args, parse_vanilla_args

if __name__ == "__main__":
    args = get_args()
    model, train_model, dataset, optimizer, epochs = parse_vanilla_args(args)

    train_model(model, dataset, optimizer, epochs)
