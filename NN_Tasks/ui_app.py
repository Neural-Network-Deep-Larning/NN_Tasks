import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from data import load_penguins
from preprocess import preprocess, split_by_class
from algorithms import initialize_weights, perceptron_train, adaline_train
from utils import normalize_train_test, compute_confusion
from models import CustomNN

st.set_page_config(page_title="Neural Networks Task1&2", layout="wide")


# ------------------------------------------------------------
# Load Dataset
# ------------------------------------------------------------
uploaded_csv = st.sidebar.file_uploader("Upload penguins.csv", type=["csv"])
df = load_penguins(path=uploaded_csv) if uploaded_csv else load_penguins()
df_proc, enc_info = preprocess(df)

species_list = sorted(df_proc["Species"].unique().tolist())

# Global feature list (used in both training + prediction)
FEATURES = ["CulmenLength", "CulmenDepth", "FlipperLength",
            "OriginLocation_enc", "BodyMass"]


# ------------------------------------------------------------
# Sidebar Menu
# ------------------------------------------------------------
model_type = st.sidebar.selectbox(
    "Choose Model",
    ["Perceptron", "Adaline", "Multi-Layer Neural Network (Backpropagation)"]
)


# ============================================================
# ========== PERCEPTRON / ADALINE ============================
# ============================================================
if model_type in ["Perceptron", "Adaline"]:

    st.header(f"{model_type} Classifier")

    feature_cols = FEATURES.copy()

    selected_features = st.sidebar.multiselect(
        "Choose exactly 2 features", feature_cols, default=feature_cols[:2]
    )

    selected_classes = st.sidebar.multiselect(
        "Choose exactly 2 classes",
        species_list,
        default=species_list[:2]
    )

    eta = st.sidebar.number_input("Learning rate", value=0.01, format="%.4f")
    epochs = st.sidebar.number_input("Epochs", value=50, min_value=1)
    use_bias = st.sidebar.checkbox("Use bias", value=True)
    seed = st.sidebar.number_input("Random seed", value=42)

    if st.sidebar.button("Run Training"):
        if len(selected_features) != 2:
            st.error(" Please choose exactly 2 features.")
            st.stop()

        if len(selected_classes) != 2:
            st.error(" Please choose exactly 2 classes.")
            st.stop()

        # Split data
        train_df, test_df = split_by_class(
            df_proc, "Species",
            tuple(selected_classes),
            seed=seed
        )

        X_train = train_df[selected_features].to_numpy(float)
        X_test = test_df[selected_features].to_numpy(float)

        label_map = {selected_classes[0]: -1, selected_classes[1]: 1}
        y_train = train_df["Species"].map(label_map).to_numpy(int)
        y_test = test_df["Species"].map(label_map).to_numpy(int)

        # Normalize
        X_train_std, X_test_std, mu, sigma = normalize_train_test(X_train, X_test)

        # Initialize
        w, b = initialize_weights(n_features=2, use_bias=use_bias, seed=seed)

        # Train
        if model_type == "Perceptron":
            w_trained, b_trained, history = perceptron_train(
                X_train_std, y_train, w, b,
                eta=eta, epochs=epochs, use_bias=use_bias
            )
        else:
            w_trained, b_trained, history = adaline_train(
                X_train_std, y_train, w, b,
                eta=eta, epochs=epochs, use_bias=use_bias
            )

        # Predict
        y_pred = np.where(
            np.dot(X_test_std, w_trained) + (b_trained if use_bias else 0) >= 0,
            1,
            -1
        )

        # Confusion Matrix
        cm = compute_confusion(y_test, y_pred, pos_label=1)
        acc = (cm["TP"] + cm["TN"]) / len(y_test) * 100

        st.subheader(" Confusion Matrix")
        st.json(cm)

        st.success(f"Accuracy: {acc:.2f}%")

        # Plot Decision Boundary
        fig, ax = plt.subplots()
        ax.scatter(X_train_std[:, 0], X_train_std[:, 1], c=y_train, cmap="coolwarm")
        x_vals = np.linspace(X_train_std[:, 0].min(), X_train_std[:, 0].max(), 100)

        if use_bias:
            y_vals = -(w_trained[0] * x_vals + b_trained) / (w_trained[1] + 1e-9)
        else:
            y_vals = -(w_trained[0] * x_vals) / (w_trained[1] + 1e-9)

        ax.plot(x_vals, y_vals, "k--")
        ax.set_xlabel(selected_features[0])
        ax.set_ylabel(selected_features[1])
        st.pyplot(fig)

        # -----------------------------------------------------------
        #   Predict New Sample (Perceptron / Adaline)
        # -----------------------------------------------------------
        st.subheader(" Predict a New Sample")

        # Store trained model info
        st.session_state["weights"] = w_trained
        st.session_state["bias"] = b_trained
        st.session_state["mu"] = mu
        st.session_state["sigma"] = sigma
        st.session_state["use_bias"] = use_bias
        st.session_state["class_pair"] = selected_classes
        st.session_state["selected_features"] = selected_features

    col1, col2 = st.columns(2)
    with col1:
            feat1_val = st.number_input(f"Enter {selected_features[0]}:", value=0.0)
    with col2:
            feat2_val = st.number_input(f"Enter {selected_features[1]}:", value=0.0)

    if st.button("Predict Class"):
            X_new = np.array([[feat1_val, feat2_val]])

            mu = st.session_state["mu"]
            sigma = st.session_state["sigma"]
            X_new_std = (X_new - mu) / sigma

            weights = st.session_state["weights"]
            bias = st.session_state["bias"]
            use_bias = st.session_state["use_bias"]

            # Net input
            net_value = np.dot(X_new_std, weights) + (bias if use_bias else 0)

            # Signum activation
            y_output = 1 if net_value >= 0 else -1

            # Map back to class name
            class_pair = st.session_state["class_pair"]
            inv_map = {-1: class_pair[0], 1: class_pair[1]}
            predicted_label = inv_map[int(y_output)]

            st.write(f"**Net value:** {float(net_value):.4f}")
            st.write(f"**Signum output:** {int(y_output)}")
            st.success(f"Predicted Class: **{predicted_label}**")


# ============================================================
# ==================== BACKPROP MLP ==========================
# ============================================================
elif model_type == "Multi-Layer Neural Network (Backpropagation)":

    st.header(" Multi-Layer Neural Network (Backpropagation)")

    # --------------------------
    # Network Hyperparameters
    # --------------------------
    num_layers = st.sidebar.number_input("Number of Hidden Layers",
                                         min_value=1, max_value=5, value=1)

    neurons_text = st.sidebar.text_input(
        "Neurons per hidden layer (comma-separated)", "5"
    )
    num_neurons = [int(x.strip()) for x in neurons_text.split(",")]

    lr = st.sidebar.number_input("Learning Rate", value=0.01, format="%.4f")
    epochs = st.sidebar.number_input("Epochs", value=200, min_value=10)
    activation = st.sidebar.selectbox("Activation Function", ["sigmoid", "tanh"])
    use_bias = st.sidebar.checkbox("Use Bias", value=True)
    seed = st.sidebar.number_input("Random Seed", value=42)

    # --------------------------
    # Training
    # --------------------------
    if st.sidebar.button("Train Network"):

        selected_classes = species_list

        train_df, test_df = split_by_class(
            df_proc, "Species",
            tuple(selected_classes), seed=seed
        )

        X_train = train_df[FEATURES].to_numpy(float)
        X_test = test_df[FEATURES].to_numpy(float)

        class_to_idx = {c: i for i, c in enumerate(selected_classes)}
        y_train = train_df["Species"].map(class_to_idx).to_numpy(int)
        y_test = test_df["Species"].map(class_to_idx).to_numpy(int)

        X_train_std, X_test_std, mu, sigma = normalize_train_test(X_train, X_test)

        nn = CustomNN()
        nn.initialize(
            num_of_layers=num_layers,
            num_neurons=num_neurons,
            activation=activation,
            use_bias=use_bias,
            lr=lr,
            random_state=seed
        )

        history = nn.train(X_train_std, y_train, epochs=epochs)

        # Save model
        st.session_state["mlp_model"] = nn
        st.session_state["mu"] = mu
        st.session_state["sigma"] = sigma

        # Plot Loss
        st.subheader(" Loss Curve")
        fig1, ax1 = plt.subplots()
        ax1.plot(history["loss"])
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        st.pyplot(fig1)

        # Accuracy
        st.subheader(" Accuracy Curve")
        fig2, ax2 = plt.subplots()
        ax2.plot(history["train_acc"])
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy")
        st.pyplot(fig2)

        # Confusion Matrix
        y_pred = nn.predict(X_test_std)
        cm_matrix = pd.crosstab(y_test, y_pred, rownames=["Actual"], colnames=["Predicted"])
        st.subheader(" Confusion Matrix")
        st.dataframe(cm_matrix)

        acc = np.mean(y_pred == y_test) * 100
        st.success(f"Overall Accuracy: {acc:.2f}%")

    # --------------------------
    # Predict Single Sample
    # --------------------------
    st.subheader(" Predict Single Sample (MLP)")

    col1, col2 = st.columns(2)
    with col1:
        s1 = st.number_input("Culmen Length", value=40.0)
        s2 = st.number_input("Culmen Depth", value=18.0)
        s3 = st.number_input("Flipper Length", value=190.0)

    with col2:
        origin_map = enc_info["origin_map"]
        origin_value = st.selectbox("Origin", list(origin_map.keys()))
        s4 = origin_map[origin_value]
        s5 = st.number_input("Body Mass", value=4000.0)

    if st.button("Predict MLP Class"):
        if "mlp_model" not in st.session_state:
            st.error(" Train the MLP first.")
            st.stop()

        nn = st.session_state["mlp_model"]
        mu = st.session_state["mu"]
        sigma = st.session_state["sigma"]

        sample = np.array([[s1, s2, s3, s4, s5]], float)
        sample_std = (sample - mu) / sigma

        pred = nn.predict(sample_std)[0]
        class_name = species_list[pred]

        st.success(f"Predicted Class: **{class_name}**")
