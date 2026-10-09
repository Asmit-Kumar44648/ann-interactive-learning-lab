
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="ANN Interactive Learning Lab",
    page_icon="🧠",
    layout="wide",
)

SEED = 42

st.title("🧠 ANN Interactive Learning Lab")
st.markdown(
    """
    **Explore how an Artificial Neural Network learns.**

    Adjust the network architecture and training parameters,
    train a TensorFlow model, and observe its performance on
    the Iris flower classification dataset.
    """
)

st.info(
    "This demo uses the Iris dataset: 150 flower samples, "
    "four numerical input features, and three flower classes."
)

# --------------------------------------------------
# LOAD AND PREPARE DATA
# --------------------------------------------------
@st.cache_data
def load_dataset():
    dataset = load_iris()

    X = dataset.data.astype(np.float32)
    y = dataset.target.astype(np.int32)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=SEED,
        stratify=y,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train).astype(
        np.float32
    )
    X_test_scaled = scaler.transform(X_test).astype(
        np.float32
    )

    return (
        dataset,
        X_train,
        X_test,
        y_train,
        y_test,
        X_train_scaled,
        X_test_scaled,
        scaler,
    )


(
    dataset,
    X_train,
    X_test,
    y_train,
    y_test,
    X_train_scaled,
    X_test_scaled,
    scaler,
) = load_dataset()

class_names = dataset.target_names.tolist()
feature_names = dataset.feature_names

# --------------------------------------------------
# DATASET OVERVIEW
# --------------------------------------------------
with st.expander("Explore the dataset", expanded=False):
    st.write("Input features:", feature_names)
    st.write("Output classes:", class_names)
    st.write("Dataset shape:", dataset.data.shape)

    preview = pd.DataFrame(
        dataset.data,
        columns=feature_names,
    )
    preview["species"] = [
        class_names[index] for index in dataset.target
    ]

    st.dataframe(preview.head(10), use_container_width=True)

# --------------------------------------------------
# SIDEBAR: ANN CONFIGURATION
# --------------------------------------------------
st.sidebar.header("⚙️ Neural Network Settings")

hidden_layer_count = st.sidebar.selectbox(
    "Number of hidden layers",
    [1, 2, 3],
    index=1,
)

neurons_1 = st.sidebar.slider(
    "Neurons in hidden layer 1", 4, 64, 16, step=4
)
neurons_2 = st.sidebar.slider(
    "Neurons in hidden layer 2", 4, 64, 12, step=4
)
neurons_3 = st.sidebar.slider(
    "Neurons in hidden layer 3", 4, 64, 8, step=4
)

learning_rate = st.sidebar.select_slider(
    "Learning rate",
    options=[0.0001, 0.0005, 0.001, 0.005, 0.01],
    value=0.001,
    format_func=lambda value: f"{value:g}",
)

epochs = st.sidebar.slider(
    "Training epochs", 10, 150, 50, step=10
)
batch_size = st.sidebar.select_slider(
    "Batch size",
    options=[8, 16, 32, 64],
    value=16,
)

neurons = [neurons_1, neurons_2, neurons_3][:hidden_layer_count]

st.sidebar.markdown("---")
st.sidebar.caption("Random seed: 42")
st.sidebar.caption("Test split: 20%")
st.sidebar.caption("Optimizer: Adam")
st.sidebar.caption("Activation: ReLU / Softmax")

# --------------------------------------------------
# BUILD ANN
# --------------------------------------------------
def build_model():
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(SEED)

    model = tf.keras.Sequential()
    model.add(
        tf.keras.layers.Input(shape=(X_train_scaled.shape[1],))
    )

    for units in neurons:
        model.add(
            tf.keras.layers.Dense(
                units,
                activation="relu",
            )
        )

    model.add(
        tf.keras.layers.Dense(
            len(class_names),
            activation="softmax",
        )
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=learning_rate
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


# --------------------------------------------------
# TRAINING
# --------------------------------------------------
st.header("1. Train your neural network")

st.write(
    "Change the settings in the sidebar, then train the network. "
    "The model is evaluated on test samples not used for training."
)

if st.button(
    "🚀 Train ANN",
    type="primary",
    use_container_width=True,
):
    with st.spinner("Training the neural network..."):
        model = build_model()

        history = model.fit(
            X_train_scaled,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.20,
            shuffle=True,
            verbose=0,
        )

        test_loss, test_accuracy = model.evaluate(
            X_test_scaled,
            y_test,
            verbose=0,
        )

        probabilities = model.predict(
            X_test_scaled,
            verbose=0,
        )
        predictions = np.argmax(probabilities, axis=1)

        st.session_state["ann_model"] = model
        st.session_state["ann_scaler"] = scaler
        st.session_state["ann_history"] = history.history
        st.session_state["ann_test_loss"] = float(test_loss)
        st.session_state["ann_test_accuracy"] = float(
            test_accuracy
        )
        st.session_state["ann_predictions"] = predictions
        st.session_state["ann_y_test"] = y_test.copy()
        st.session_state["ann_neurons"] = neurons.copy()
        st.session_state["ann_learning_rate"] = learning_rate
        st.session_state["ann_epochs"] = epochs
        st.session_state["ann_batch_size"] = batch_size

    st.success("Training completed successfully!")

# --------------------------------------------------
# RESULTS AND VISUALIZATIONS
# --------------------------------------------------
if "ann_model" not in st.session_state:
    st.warning(
        "Train the ANN to display evaluation metrics, learning "
        "curves, and interactive predictions."
    )

else:
    model = st.session_state["ann_model"]
    history_data = st.session_state["ann_history"]
    predictions = st.session_state["ann_predictions"]
    actual = st.session_state["ann_y_test"]

    st.header("2. Model Evaluation")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Test Accuracy",
        f"{st.session_state['ann_test_accuracy']:.2%}",
    )
    col2.metric(
        "Test Loss",
        f"{st.session_state['ann_test_loss']:.4f}",
    )
    col3.metric(
        "Trainable Parameters",
        f"{model.count_params():,}",
    )

    st.caption(
        "Results depend on the selected network settings. "
        "Accuracy is measured on the held-out test set."
    )

    # Learning curves
    st.header("3. Learning Curves")

    history_df = pd.DataFrame(history_data)
    history_df.index = np.arange(1, len(history_df) + 1)
    history_df.index.name = "Epoch"

    left, right = st.columns(2)

    with left:
        st.subheader("Accuracy")
        accuracy_df = history_df[
            ["accuracy", "val_accuracy"]
        ].rename(
            columns={
                "accuracy": "Training accuracy",
                "val_accuracy": "Validation accuracy",
            }
        )
        st.line_chart(accuracy_df)

    with right:
        st.subheader("Loss")
        loss_df = history_df[
            ["loss", "val_loss"]
        ].rename(
            columns={
                "loss": "Training loss",
                "val_loss": "Validation loss",
            }
        )
        st.line_chart(loss_df)

    with st.expander("View training history data"):
        st.dataframe(history_df, use_container_width=True)

    # Confusion matrix
    st.header("4. Test Set Predictions")

    cm = confusion_matrix(
        actual,
        predictions,
        labels=np.arange(len(class_names)),
    )

    fig, ax = plt.subplots(figsize=(6, 4))
    image = ax.imshow(cm, cmap="Blues")

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=25, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Actual class")
    ax.set_title("Confusion Matrix")

    for row in range(cm.shape[0]):
        for col in range(cm.shape[1]):
            ax.text(
                col,
                row,
                str(cm[row, col]),
                ha="center",
                va="center",
                color="black",
            )

    fig.colorbar(image, ax=ax, fraction=0.046)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Classification Report")

    report = classification_report(
        actual,
        predictions,
        labels=np.arange(len(class_names)),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    report_df = pd.DataFrame(report).transpose()
    st.dataframe(
        report_df.round(3),
        use_container_width=True,
    )

    # Interactive prediction
    st.header("5. Try Your Own Flower Measurements")

    st.write(
        "Use the sliders to enter four flower measurements. "
        "The trained ANN will predict the species."
    )

    input_columns = st.columns(4)
    sample_values = []

    for index, column in enumerate(input_columns):
        minimum = float(dataset.data[:, index].min())
        maximum = float(dataset.data[:, index].max())
        default = float(dataset.data[:, index].mean())

        with column:
            value = st.slider(
                feature_names[index],
                min_value=minimum,
                max_value=maximum,
                value=default,
                step=0.1,
                key=f"flower_feature_{index}",
            )
            sample_values.append(value)

    if st.button("🔍 Predict Flower Species"):
        sample = np.array(
            [sample_values],
            dtype=np.float32,
        )
        sample_scaled = st.session_state[
            "ann_scaler"
        ].transform(sample)

        sample_probabilities = model.predict(
            sample_scaled,
            verbose=0,
        )[0]

        predicted_index = int(
            np.argmax(sample_probabilities)
        )

        st.success(
            f"Predicted species: {class_names[predicted_index]}"
        )

        probability_df = pd.DataFrame(
            {
                "Species": class_names,
                "Predicted probability": sample_probabilities,
            }
        )

        st.bar_chart(
            probability_df.set_index("Species")
        )
        st.dataframe(
            probability_df.assign(
                **{
                    "Predicted probability": probability_df[
                        "Predicted probability"
                    ].map(lambda probability: f"{probability:.2%}")
                }
            ),
            use_container_width=True,
        )

    # Model architecture
    st.header("6. Network Architecture")

    st.write(
        f"Input layer: {X_train_scaled.shape[1]} features → "
        f"Hidden layers: {st.session_state['ann_neurons']} → "
        f"Output layer: {len(class_names)} classes"
    )

    with st.expander("View model summary"):
        summary_lines = []
        model.summary(print_fn=summary_lines.append)
        st.code("\n".join(summary_lines))

st.markdown("---")
st.caption(
    "ANN Interactive Learning Lab | TensorFlow/Keras | "
    "Dataset: Iris | Reproducibility seed: 42"
)
