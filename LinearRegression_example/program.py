from FeaturesExtraction import FeaturesExtraction
from Models import LinearRegression
from Models import NeuralNetwork  # Importing NeuralNetwork class
from Models import SupportVectorRegression  # Importing the SVR class
from Trainer import Trainer
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import torch
from sklearn.model_selection import cross_val_score  # SVR specific cross-validation
from sklearn.preprocessing import StandardScaler
from InputReader import InputReader
from Logger import Logger


# Reading the input file (*.yml file)
inp = InputReader("input.yml")
inp_read = inp.yml_reader()
eta = inp_read.content["eta"]
n_iter = inp_read.content["n_iter"]
random_state = inp_read.content["random_state"]
model_name = inp_read.content["model"]  # 'LinearRegression' or 'NeuralNetwork'
model_arch = inp_read.content.get(
    "architecture", None
)  # Get architecture if it exists, NN only

# Extracting features and labels
features_extract = FeaturesExtraction(
    inp_read.content["datafile_name"]
)  # creating class instance
features_extract.data_analysis()  # storing information of the database
[labs_np, feats_np] = features_extract.feature_extraction(
    inp_read.content["label"], inp_read.content["features"]
)  # extraction of features and labels

# 1. Convert full dataset NumPy arrays to Tensors for Cross-Validation (CV)
# Uses the full dataset (test_ratio=0.0) converted to Tensors.
X_full, _, Y_full, _ = features_extract.data_split(
    feats_np, labs_np, test_ratio=0.0, random_state=random_state
)

# Train Test Split Implementation; Split and Tensor Conversion
X_train, X_test, y_train, y_test = features_extract.data_split(
    feats_np,
    labs_np,
    test_ratio=inp_read.content["test_split_ratio"],
    random_state=inp_read.content["random_state"],
)
# --------------------------------------------------------------
# Scaling for NN and LR Models (Features X and Labels Y)
# --------------------------------------------------------------

# Initialise two separate scalers
scaler_X = StandardScaler()
scaler_y = StandardScaler()

# ONly apply external scaling if NOT using SVR
# SVR has built-in scaling via sklearn
if model_name != "support_vector_regression":
    print("\n Applying Standard Scaling to Features for LR and NN Models")

    # Scale Features X
    # 1. Fit the scaler only on the training data - convert Tensor to numpy, fit and then transform
    X_train_scaled = scaler_X.fit_transform(X_train.numpy())

    # 2. Transform the test data, using the training scaler
    X_test_scaled = scaler_X.transform(X_test.numpy())

    # 3. Transform X_full for the CV step later
    X_full_scaled = scaler_X.transform(X_full.numpy())

    # 4. Convert back to Tensors
    X_train = torch.tensor(X_train_scaled, dtype=torch.double)
    X_test = torch.tensor(X_test_scaled, dtype=torch.double)
    X_full = torch.tensor(X_full_scaled, dtype=torch.double)

    # Scale Labels Y
    # 1. Reshape Y to (N, 1), standard scaler expects 2D input
    y_train_np = y_train.numpy().reshape(-1, 1)
    y_test_np = y_test.numpy().reshape(-1, 1)
    Y_full_np = Y_full.numpy().reshape(-1, 1)

    # 2 . Fit the scaler only on the training labels
    y_train_scaled = scaler_y.fit_transform(y_train_np)
    y_test_scaled = scaler_y.transform(y_test_np)
    Y_full_scaled = scaler_y.transform(Y_full_np)

    # 3. Convert back to Tensors
    y_train = torch.tensor(y_train_scaled, dtype=torch.double).view(-1, 1)
    y_test = torch.tensor(y_test_scaled, dtype=torch.double).view(-1, 1)
    Y_full = torch.tensor(Y_full_scaled, dtype=torch.double).view(-1, 1)
# --------------------------------------------


# Model
# hashed#lr_model1 = LinearRegression(inp_read.content['model'], X_train, random_state) # instance of a linear regression model
# hashed#lr_model2 = LinearRegression(inp_read.content['model'],X_train, random_state) # other instance only for demonstartion purposes

is_svr = False  # Flag to indicate if SVR model is used

if model_name.lower() == "linear_regression":
    ml_model1 = LinearRegression(
        "Linear Regression SGD", X_train, random_state
    )  # instance of a linear regression model
    ml_model2 = LinearRegression(
        "Linear Regression GD", X_train, random_state
    )  # other instance only for demonstartion purposes

elif model_name.lower() == "neural_network":
    if model_arch is None:
        raise ValueError(
            "Architecture must be specified for Neural Network model in input.yml"
        )

    ml_model1 = NeuralNetwork(model_name, X_train, random_state, model_arch)
    # Create a second instance for GD training plot
    ml_model2 = NeuralNetwork(model_name, X_train, random_state, model_arch)

elif inp_read.content["model"] == "support_vector_regression":
    # Logic for Support Vector Regression model

    is_svr = True
    ml_model1 = SupportVectorRegression(
        inp_read.content["model"],
        X_train,
        y_train,
        inp_read.content["random_state"],
        inp_read.content["svr_parameters"],
    )

else:
    raise ValueError(f"Unsupported model type: {model_name}")

# Separate paths for SVR and Other Models


# --------------------------------------------
# SVR Model Path
# --------------------------------------------
if is_svr:
    print(f"\n Running {model_name}")

    # 1. Cross-Validation using scikit-learn's cross_val_score
    print("SVR Cross Validation")
    # We can access the underlying sklearn SVR model
    if isinstance(ml_model1, SupportVectorRegression):

        try:
            scores = cross_val_score(
                ml_model1.model,
                X_full.detach().numpy(),
                Y_full.detach().numpy().ravel(),
                cv=5,
                scoring="neg_mean_squared_error",
            )
            avg_cv_loss = -np.mean(scores)
            print(
                f"Final Average Cross-Validation Loss (5-Fold MSE): {avg_cv_loss:.6f}"
            )
        except AttributeError:
            print("Could not run cross-validation: underlying model not found.")
            avg_cv_loss = 0.0

    else:  # Fallback
        print("Model flag was SVR, but object type is not SupportVectorRegression.")
        avg_cv_loss = 0.0

    # 2. Prediction
    prediction_test = ml_model1.predict(X_test)
    print("SVR skipping Loss Plotting as training is non-iterative.")


# --------------------------------------------
# Other Models Path (Linear Regression, Neural Network)
# --------------------------------------------
else:
    # Training
    trainer1 = Trainer(
        inp_read.content["trainer"],
        ml_model1,
        inp_read.content["eta"],
        inp_read.content["n_iter"],
        inp_read.content["batch_size"],
    )  # instance of training class
    trainer2 = Trainer(
        "GD_Custom",
        ml_model2,
        inp_read.content["eta"],
        inp_read.content["n_iter"],
        inp_read.content["batch_size"],
    )
    # lr_model changed to ml_model

    # --- START CROSS-VALIDATION ---
    print("\n*** Starting Cross-Validation for SGD Model Performance Estimate ***")
    # Note: trainer1 model is repeatedly re-initialized inside cross_validate for fairness
    avg_cv_loss, cv_losses = trainer1.cross_validate(X_full, Y_full, n_splits=5)
    print(f"Final Average Cross-Validation Loss (5-Fold MSE): {avg_cv_loss:.6f}")
    # --- END CROSS-VALIDATION ---

    # [labs, feats] = features_extract.feature_extraction(label_name,names)
    # [labs2, feats2] = features_extract.feature_extraction(label_name,names)

    log = Logger(ml_model1, trainer1)  # lr_model changed to ml_model
    # set calculated CV loss attribute in the logger object
    log.avg_cv_loss = float(avg_cv_loss)
    log.log("w")

    train1 = trainer1.training(
        X_test, y_test, X_train, y_train
    )  # stochastic gradient descent training
    # train2 = trainer2.gd_optim(X_test, y_test, X_train, y_train)  # gradient descent training

    # Second part of the log file
    log = Logger(ml_model1, trainer1)  # lr_model changed to ml_model
    log.log("a")

    # Plot of the training
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))

    # Plot SDG Loss, using results from the train1
    ax[0].plot(
        range(1, len(train1.avg_losses) + 1), np.log10(train1.avg_losses), marker="o"
    )
    ax[0].set_xlabel("Number of iter")
    ax[0].set_ylabel("Log(Mean Squared Error)")
    ax[0].set_title(
        f"Average Training Losses with GD for {model_name}"
    )  # model_name used here

    # Plot GD Loss, using results from train2
    ax[1].plot(
        range(1, len(train1.test_losses) + 1), np.log10(train1.test_losses), marker="o"
    )
    ax[1].set_xlabel("Number of iter")
    ax[1].set_ylabel("Log(Mean Squared Error)")
    ax[1].set_title(f"Test Losses for {model_name}")  # model_name used here
    plt.show()

    # prediction of the model trained with a "good" learning rate
    # prediction = test1.predict(feats)
    prediction_test = ml_model1.predict(
        X_test
    )  ##changed to predict on unseen testing data
    # lr_model changed to ml_model

# -------------------------------------------------------------
# Final Evaluation and Plotting
# -------------------------------------------------------------
# If SVR - it is already in real units, which is handled in the class
# If LR or NN - inverse transform the scaled predictions and true values back to original units

if model_name != "support_vector_regression":
    # Inverse transform to get back to mmol/g
    prediction_final = scaler_y.inverse_transform(prediction_test.detach().numpy())
    y_test_final = scaler_y.inverse_transform(y_test.detach().numpy())
else:
    # SVR case - no scaling to reverse
    prediction_final = prediction_test.detach().numpy()
    y_test_final = y_test.detach().numpy()

# Calculate final test MSE loss
final_mse = np.mean((y_test_final - prediction_final) ** 2)
print(f"\nFinal Test Loss (MSE) on Unseen Data: {final_mse:.6f}")

# Plotting Predictions vs True Values
plt.figure(figsize=(6, 6))
plt.scatter(y_test_final, prediction_final, alpha=0.7)

# Plot the 1:1 line for visual reference: this line represents where the prediction = the true value
min_val = min(np.min(y_test_final), np.min(prediction_final))
max_val = max(np.max(y_test_final), np.max(prediction_final))
plt.plot(
    [min_val, max_val],
    [min_val, max_val],
    color="r",
    linestyle="--",
    label="Ideal Prediction",
)

plt.xlabel("True Values (mmol/g)")
plt.ylabel("Predicted Values (mmol/g)")
plt.title(f"Prediction vs True Values on Test Data: {model_name}")
plt.grid(True)
plt.legend()
plt.show()


"""
# Convert Tensors to Numpy for plotting
y_test_np = y_test.detach().numpy()
prediction_test_np = prediction_test.detach().numpy()

# plt.scatter(labs,prediction)
plt.scatter(y_test_np, prediction_test_np)  ##compare true vs predicted for test data

# Compare true vs predicted values for test data
plt.scatter(y_test_np, prediction_test_np)

# Plot the 1:1 line for visual reference: this line represents where the prediction = the true value
plt.plot(
    np.linspace(min(y_test_np), max(y_test_np)),
    np.linspace(min(y_test_np), max(y_test_np)),
    color="r",
)
plt.xlabel("True Values")
plt.ylabel("Predicted Values")
plt.grid()
plt.title(f"Prediction vs True Values on Test Data: {model_name}")
plt.show()
"""

"""
# Prediction of the model trained with stochastic gradient descent
prediction = ml_model1.predict(feats)

plt.scatter(labs,prediction.detach().numpy())
plt.plot(np.linspace(0,70),np.linspace(0,70), color='r')
plt.xlabel('true values')
plt.ylabel('predicted values')
plt.grid()
plt.show()
"""

"""
# example of a training with a "good" learning rate
#test1 = LinearRegressionGD(eta1,n_iter).fit(feats,labs) --- original line
test1 = LinearRegressionGD(eta1, n_iter).fit(X_train, y_train) ##changed to use training data from split
ax[0].plot(range(1, len(test1.losses_) + 1),np.log10(test1.losses_), marker='o')
ax[0].set_xlabel('Number of iter')
ax[0].set_ylabel('log(Mean squared error)')
ax[0].set_title(f'Learning rate {eta1}')

# example of a training with a "bad" learning rate
#test2 = LinearRegressionGD(eta2, n_iter).fit(feats2, labs2) --- original line
test2 = LinearRegressionGD(eta2, n_iter).fit(X_train, y_train) ##changed to use training data from split
ax[1].plot(range(1, len(test2.losses_) + 1),np.log10(test2.losses_), marker='o')
# example of a stochastic gradiesnt descent optimisation algorithm
ax[0].plot(range(1, len(train1.losses) + 1),np.log10(train1.losses), marker='o')
ax[0].set_xlabel('Number of iter')
ax[0].set_ylabel('log(Mean squared error)')
ax[0].set_title(f'Stocastic gradient descent')
# example of a linear gradiesnt descent optimisation algorithm
ax[1].plot(range(1, len(trainer2.losses) + 1),np.log10(trainer2.losses), marker='o')
ax[1].set_xlabel('Number of iter')
ax[1].set_ylabel('log(Mean squared error)')
ax[1].set_title(f'gradient descent')
plt.show()
"""
