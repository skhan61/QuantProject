# import utils
# from models import  SimpleNN
# from utils import pad_sequence, \
#     convert_to_torch, get_era2data, \
#     train_model, train_on_batch
# import torch.optim as optim

# FEATURE_DIM = 225
# OUTPUT_DIM = 1
# HIDDEN_DIM = 128
# MAX_LEN = 500

# device = "cuda" if torch.cuda.is_available() else "cpu"
# print(device)

# # Instantiate and test the SimpleNN
# model = SimpleNN(FEATURE_DIM, HIDDEN_DIM, OUTPUT_DIM)

# model.to(device=device)
# criterion = torch.nn.MSELoss()
# optimizer = optim.Adam(model.parameters(), lr=1e-4)

# # Number of training iterations
# # Train for longer with low LR

# num_epochs = 1
# patience = 5

# from torch.optim.lr_scheduler import StepLR
# scheduler = StepLR(optimizer, step_size=100, gamma=0.1)

# for train_idx, val_idx in cv.split(dataset):
#     train = dataset.loc[train_idx, :]
#     val = dataset.loc[val_idx, :]
    
#     era2data_train = get_era2data(train, features, target) 
#     era2data_val = get_era2data(val, features, target)  # You probably meant to rename this to era2data_val

#     model, best_corr, preds = train_model(model, criterion, optimizer, scheduler, \
#                                     num_epochs, patience, era2data_train, \
#                                     era2data_val, is_lr_scheduler=True)
#     # break

# import optuna
# import os
# import torch
# import torch.optim as optim
# from torch.optim.lr_scheduler import StepLR
# import utils
# from models import SimpleNN
# from utils import pad_sequence, convert_to_torch, \
#     get_era2data, train_model, train_on_batch, metrics_on_batch



# device = "cuda" if torch.cuda.is_available() else "cpu"
# FEATURE_DIM = 225
# OUTPUT_DIM = 1
# MAX_LEN = 500


# def save_best_model(study, trial):
#     try:
#         # If the trial number matches the best trial number, save the model
#         if study.best_trial.number == trial.number:
#             best_model_path = os.path.join(model_dir, f"best_model_trial_{trial.number}.pt")
#             model = trial.user_attrs["model"]  # Get the model from trial's user attributes
#             torch.save(model.state_dict(), best_model_path)
#     except ValueError:
#         pass  # No trials are completed yet, so just pass


# def objective(trial):
#     # 1. Define hyperparameters to optimize
#     HIDDEN_DIM = trial.suggest_int("HIDDEN_DIM", 32, 512, log=True)  # Log scale search for hidden dimensions
#     lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)  # Log scale search for learning rate
#     step_size = trial.suggest_int("step_size", 50, 200)
#     gamma = trial.suggest_float("gamma", 0.01, 0.5, log=True)
#     patience = trial.suggest_int("patience", 3, 10)
#     num_epochs = trial.suggest_int("num_epochs", 1, 10)

#     # 2. Instantiate model and other components using hyperparameters
#     model = SimpleNN(FEATURE_DIM, HIDDEN_DIM, OUTPUT_DIM)
#     model.to(device=device)
#     criterion = torch.nn.MSELoss()
#     optimizer = optim.Adam(model.parameters(), lr=lr)
#     scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)

#     val_scores = []

#     for train_idx, val_idx in cv.split(dataset):
#         train = dataset.loc[train_idx, :]
#         val = dataset.loc[val_idx, :]

#         # Convert your data to appropriate loaders here
#         train_loader = get_era2data(train, features, target)
#         val_loader = get_era2data(val, features, target)

#         _, best_corr, _ = train_model(model, criterion, optimizer, \
#                 scheduler, num_epochs, patience, train_loader, val_loader)

#         metrics = metrics_on_batch(best_corr)
#         val_scores.append(metrics[2])

#     trial.set_user_attr("model", model)

#     return np.mean(val_scores)

# # Directory where models will be saved
# model_dir = "saved_models"
# if not os.path.exists(model_dir):
#     os.makedirs(model_dir)

# study = optuna.create_study(direction="minimize", study_name="SimpleNN")
# study.optimize(objective, n_trials=100, callbacks=[save_best_model])

# # Print best trial's parameters and value
# print(f"Best trial: {study.best_trial.params}")
# print(f"Best value: {study.best_trial.value}")