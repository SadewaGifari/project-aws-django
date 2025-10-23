
# =========================================================
# 0. IMPORT LIBRARIES
# =========================================================
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# =========================================================
# 1. MODEL SENSOR (TABULAR)
# =========================================================
def train_sensor_model(csv_path="disease_growth_level.csv"):
    # Load data
    data = pd.read_csv(csv_path)
    data.columns = [c.strip().lower().replace(' ', '_') for c in data.columns]

    # Handle missing values
    data['ventilation'] = data['ventilation'].fillna('unknown')
    data['light_intensity'] = data['light_intensity'].fillna('unknown')
    data['ph'] = data['ph'].fillna(data['ph'].median())

    # Encode categorical features and target
    le_vent = LabelEncoder()
    le_light = LabelEncoder()
    le_y = LabelEncoder()

    data['vent_enc'] = le_vent.fit_transform(data['ventilation'].str.lower())
    data['light_enc'] = le_light.fit_transform(data['light_intensity'].str.lower())
    y_enc = le_y.fit_transform(data['disease_growth_possibility_level'].str.lower())

    # Features and target
    X = data[['temperature', 'humidity', 'ph', 'vent_enc', 'light_enc']]
    y = y_enc

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train Decision Tree Classifier
    clf = DecisionTreeClassifier(criterion='entropy', max_depth=6, random_state=42)
    clf.fit(X_train, y_train)

    # Evaluate
    y_pred = clf.predict(X_test)
    print("\n=== SENSOR MODEL EVALUATION ===")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred, target_names=le_y.classes_))

    # Visualize Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d',
                xticklabels=le_y.classes_, yticklabels=le_y.classes_,
                cmap='Blues')
    plt.title("Confusion Matrix - Sensor Model")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    # Visualize Decision Tree
    plt.figure(figsize=(20,10))
    plot_tree(clf,
              feature_names=X.columns,
              class_names=le_y.classes_,
              filled=True,
              rounded=True,
              fontsize=8)
    plt.title("Decision Tree Visualization - Sensor Model")
    plt.show()

    # Feature Importance
    importances = pd.Series(clf.feature_importances_, index=X.columns)
    plt.figure(figsize=(6,4))
    importances.sort_values().plot(kind='barh', color='teal')
    plt.title("Feature Importance - Sensor Model")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.show()

    # Save model and encoders
    joblib.dump(clf, "sensor_model.pkl")
    joblib.dump(le_y, "sensor_label_encoder.pkl")
    joblib.dump(le_vent, "ventilation_encoder.pkl")
    joblib.dump(le_light, "light_encoder.pkl")

    # Prediction function for sensor data
    def predict_sensor(temp, hum, ph, ventilation, light_intensity):
        vent_enc = le_vent.transform([ventilation.lower()])[0] if ventilation.lower() in le_vent.classes_ else le_vent.transform(['unknown'])[0]
        light_enc = le_light.transform([light_intensity.lower()])[0] if light_intensity.lower() in le_light.classes_ else le_light.transform(['unknown'])[0]
        X_new = pd.DataFrame([[temp, hum, ph, vent_enc, light_enc]], columns=X.columns)
        pred = clf.predict(X_new)
        return le_y.inverse_transform(pred)[0]

    return clf, le_y, le_vent, le_light, predict_sensor

# =========================================================
# 2. MODEL GAMBAR (CNN)
# =========================================================
def train_image_model(image_root="dataset_images"):
    """
    Dataset folder structure:
    dataset_images/
        train/
            High/
            Low/
            Moderate/
        val/
            High/
            Low/
            Moderate/
    """
    img_size = (224, 224)
    batch_size = 32

    train_ds = tf.keras.utils.image_dataset_from_directory(
        f"{image_root}/train", image_size=img_size, batch_size=batch_size)
    val_ds = tf.keras.utils.image_dataset_from_directory(
        f"{image_root}/val", image_size=img_size, batch_size=batch_size)

    train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=img_size + (3,), include_top=False, weights='imagenet')
    base_model.trainable = False

    model = keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.2),
        layers.Dense(len(train_ds.class_names), activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    print("\n=== TRAINING IMAGE MODEL ===")
    history = model.fit(train_ds, validation_data=val_ds, epochs=15)

    model.save("bread_cnn_model.h5")

    # Prediction function for image
    def predict_image(img_path):
        img = tf.keras.utils.load_img(img_path, target_size=img_size)
        arr = tf.keras.utils.img_to_array(img)
        arr = tf.expand_dims(arr, 0)  # batch dimension
        pred = model.predict(arr)
        class_idx = tf.argmax(pred[0]).numpy()
        class_name = train_ds.class_names[class_idx]
        return class_name

    return model, predict_image

# =========================================================
# 3. PREDIKSI GABUNGAN
# =========================================================
def combined_prediction(sensor_predict_func, image_predict_func,
                        temp, hum, ph, ventilation, light_intensity, img_path):
    """
    Menggabungkan prediksi sensor dan gambar.
    """
    sensor_result = sensor_predict_func(temp, hum, ph, ventilation, light_intensity)
    image_result = image_predict_func(img_path)
    return {"sensor_result": sensor_result, "image_result": image_result}

# =========================================================
# 4. PROGRAM UTAMA
# =========================================================
if __name__ == "__main__":
    # Latih dan evaluasi model sensor
    clf_sensor, le_y, le_vent, le_light, predict_sensor = train_sensor_model("disease_growth_level.csv")

    # Contoh prediksi sensor
    print("\nContoh Prediksi Sensor:")
    print(predict_sensor(25.0, 80, 7.0, "low", "high"))

    # Jika dataset gambar sudah siap, latih model gambar dan lakukan prediksi gabungan
    # Uncomment bagian ini jika dataset gambar tersedia

    # model_img, predict_img = train_image_model("dataset_images")
    # hasil_gabungan = combined_prediction(
    #     predict_sensor, predict_img,
    #     25.0, 80, 7.0, "low", "high",
    #     "dataset_images/test/high/sample.jpg"
    # )
    # print("\nPrediksi Gabungan:")
    # print(hasil_gabungan)
