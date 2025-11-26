import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import plotly.express as px


df = pd.read_csv("leon.csv")
df = df.drop(columns=["Id"])


label_encoder = LabelEncoder()
df["Species"] = label_encoder.fit_transform(df["Species"])


X = df.drop(columns=["Species"])
y = df["Species"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

modelo = DecisionTreeClassifier()
modelo.fit(X_train, y_train)


y_pred = modelo.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="macro")
recall = recall_score(y_test, y_pred, average="macro")
f1 = f1_score(y_test, y_pred, average="macro")


st.title(" Clasificador de Flores Iris – Dashboard Interactivo")
st.sidebar.title("Integrantes del Proyecto")
st.sidebar.write("Johan osorio")


st.subheader("📊 Métricas del Modelo")
st.write(f"*Accuracy:* {accuracy*100:.2f}%")
st.write(f"*Precisión:* {precision*100:.2f}%")
st.write(f"*Recall:* {recall*100:.2f}%")
st.write(f"*F1-score:* {f1*100:.2f}%")

# 4. Entradas del usuario
st.subheader("Parámetros para Predicción")

sepal_length = st.slider("Longitud del Sépalo (cm)", 4.0, 8.0, 5.0)
sepal_width = st.slider("Anchura del Sépalo (cm)", 2.0, 4.5, 3.0)
petal_length = st.slider("Longitud del Pétalo (cm)", 1.0, 7.0, 4.0)
petal_width = st.slider("Anchura del Pétalo (cm)", 0.1, 2.5, 1.2)

entrada_usuario = pd.DataFrame({
    "SepalLengthCm": [sepal_length],
    "SepalWidthCm": [sepal_width],
    "PetalLengthCm": [petal_length],
    "PetalWidthCm": [petal_width]
})
# 5. Predicción + Gráfico 3D con muestra del usuario
st.subheader("🌐 Visualización 3D del Dataset + Tu Muestra")

if st.button("🔮 Predecir Especie"):
    pred = modelo.predict(entrada_usuario)[0]
    especie_predicha = label_encoder.inverse_transform([pred])[0]

    st.success(f"La especie predicha es: *{especie_predicha}*")

    df_plot = df.copy()
    df_plot["SpeciesName"] = df_plot["Species"].map({
        0: "Iris-setosa",
        1: "Iris-versicolor",
        2: "Iris-virginica"
    })

    fig = px.scatter_3d(
        df_plot,
        x="SepalLengthCm",
        y="SepalWidthCm",
        z="PetalLengthCm",
        color="SpeciesName",
        title="Dataset Iris + Muestra del Usuario"
    )

    fig.add_scatter3d(
        x=[sepal_length],
        y=[sepal_width],
        z=[petal_length],
        mode="markers",
        marker=dict(size=6, color="red", symbol="diamond"),
        name="Tu muestra"
    )

    st.plotly_chart(fig)

else:
    st.info("Ingresa los valores y presiona *Predecir Especie* para ver tu punto en el gráfico 3D.")


st.header("📈 Visualizaciones Adicionales del Dataset Iris")


st.subheader("📊 Histograma: Longitud del Sépalo")
fig_hist1 = px.histogram(
    df,
    x="SepalLengthCm",
    color=df["Species"].map({0:"Iris-setosa",1:"Iris-versicolor",2:"Iris-virginica"}),
    title="Distribución de la Longitud del Sépalo"
)
st.plotly_chart(fig_hist1)


st.subheader("📊 Histograma: Longitud del Pétalo")
fig_hist2 = px.histogram(
    df,
    x="PetalLengthCm",
    color=df["Species"].map({0:"Iris-setosa",1:"Iris-versicolor",2:"Iris-virginica"}),
    title="Distribución de la Longitud del Pétalo"
)
st.plotly_chart(fig_hist2)


st.subheader("🔍 Matriz de Dispersión (Scatter Matrix)")

df_scatter = df.copy()
df_scatter["SpeciesName"] = df_scatter["Species"].map({
    0: "Iris-setosa",
    1: "Iris-versicolor",
    2: "Iris-virginica"
})

fig_matrix = px.scatter_matrix(
    df_scatter,
    dimensions=["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"],
    color="SpeciesName",
    title="Matriz de Dispersión del Dataset Iris"
)
st.plotly_chart(fig_matrix)