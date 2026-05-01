import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import plotly.graph_objects as go
import plotly.express as px

NUM_NODES = 2
INPUT_STEPS = 12
OUTPUT_STEPS = 12

st.set_page_config(page_title="Traffic AI Dashboard", layout="wide")

st.markdown("""
<style>

.stApp {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    color: white;
}

h1, h2, h3, h4 {
    color: white;
}

.block-container {
    padding-top: 2rem;
}

.card {
    padding: 20px;
    border-radius: 20px;
    text-align: center;
    color: white;
    font-weight: bold;
    backdrop-filter: blur(12px);
    box-shadow: 0 8px 30px rgba(0,0,0,0.3);
}

.card1 {background: linear-gradient(135deg,#ff416c,#ff4b2b);}
.card2 {background: linear-gradient(135deg,#36d1dc,#5b86e5);}
.card3 {background: linear-gradient(135deg,#f7971e,#ffd200);}
.card4 {background: linear-gradient(135deg,#00c9ff,#92fe9d);}

[data-testid="stDataFrame"] {
    background: rgba(255,255,255,0.05);
    border-radius: 12px;
}

</style>
""", unsafe_allow_html=True)

st.markdown("""
<h1 style='text-align:center;
background: linear-gradient(to right, #00c9ff, #92fe9d);
-webkit-background-clip: text;
-webkit-text-fill-color: transparent;'>
Traffic Flow Prediction Dashboard
</h1>
""", unsafe_allow_html=True)

class GRUModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(NUM_NODES, 64, num_layers=2, batch_first=True)
        self.fc = nn.Linear(64, OUTPUT_STEPS * NUM_NODES)

    def forward(self, x):
        x = x.squeeze(-1)
        out, _ = self.gru(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out.view(-1, OUTPUT_STEPS, NUM_NODES)

model = GRUModel()
model.load_state_dict(torch.load("Gru_model.pth", map_location="cpu"))
model.eval()

uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head(), use_container_width=True)

    x_cols = [c for c in df.columns if "x_" in c]
    y_cols = [c for c in df.columns if "y_" in c]

    X = df[x_cols].values.reshape(-1, INPUT_STEPS, NUM_NODES, 1)

    scaler = StandardScaler()
    scaler.fit(X.reshape(-1,1))
    X = scaler.transform(X.reshape(-1,1)).reshape(X.shape)

    X = torch.tensor(X, dtype=torch.float32)

    with torch.no_grad():
        preds = model(X).numpy()

    preds = scaler.inverse_transform(preds.reshape(-1,1)).reshape(preds.shape)

    st.subheader("Predictions Table")
    preds_2d = preds[:5].reshape(5, -1)
    st.dataframe(preds_2d, use_container_width=True)

    if len(y_cols) > 0:

        y_true = df[y_cols].values.reshape(-1, OUTPUT_STEPS, NUM_NODES)

        actual = y_true[:100].flatten()
        predicted = preds[:100].flatten()

        mae = mean_absolute_error(actual, predicted)
        rmse = np.sqrt(mean_squared_error(actual, predicted))
        r2 = r2_score(actual, predicted)
        mape = np.mean(np.abs((actual - predicted) / (actual + 1e-5))) * 100

        c1, c2, c3, c4 = st.columns(4)

        c1.markdown(f"<div class='card card1'>MAE<br><h2>{mae:.2f}</h2></div>", unsafe_allow_html=True)
        c2.markdown(f"<div class='card card2'>RMSE<br><h2>{rmse:.2f}</h2></div>", unsafe_allow_html=True)
        c3.markdown(f"<div class='card card3'>MAPE<br><h2>{mape:.2f}</h2></div>", unsafe_allow_html=True)
        c4.markdown(f"<div class='card card4'>R2 Score<br><h2>{r2:.2f}</h2></div>", unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Prediction vs Actual")

            fig = go.Figure()
            fig.add_trace(go.Scatter(y=actual, mode='lines', name='Actual'))
            fig.add_trace(go.Scatter(y=predicted, mode='lines', name='Predicted'))

            fig.update_layout(template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Error Distribution")

            fig2 = px.histogram(actual - predicted, nbins=30)
            fig2.update_layout(template="plotly_dark")
            st.plotly_chart(fig2, use_container_width=True)

        st.subheader("Node-wise Prediction")

        sample = preds[0]

        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(y=sample[:,0], mode='lines', name='Node 0'))
        fig3.add_trace(go.Scatter(y=sample[:,1], mode='lines', name='Node 1'))
        fig3.update_layout(template="plotly_dark")

        st.plotly_chart(fig3, use_container_width=True)

        score = max(0, min(100, r2 * 100))

        st.progress(int(score))
        st.write(f"Prediction Score: {score:.2f}%")