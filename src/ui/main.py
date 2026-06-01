import streamlit as st
import pandas as pd
from datetime import datetime


from src.pipeline.predict_pipeline import PredictPipeline
from src.api.schemas import DemandData


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


st.set_page_config(page_title="Retail Demand Forecasting", layout="wide")

st.title("Retail Demand Forecasting — Inference & Optimization")

with st.sidebar:
    st.header("Inventory Optimization")
    enable_optimization = st.checkbox("Enable optimization", value=False)
    ordering_cost = None
    holding_cost = None
    lead_time_days = None
    safety_stock = 0.0
    if enable_optimization:
        ordering_cost = st.number_input("Ordering cost", value=50.0, min_value=0.0)
        holding_cost = st.number_input(
            "Holding cost (annual per unit)", value=2.0, min_value=0.0
        )
        lead_time_days = st.number_input("Lead time (days)", value=7, min_value=0)
        safety_stock = st.number_input("Safety stock (units)", value=0, min_value=0)


tab1, tab2 = st.tabs(["Single prediction", "Batch prediction"])


with tab1:
    st.subheader("Single row prediction")
    with st.form("single_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            date = st.date_input("Date", value=datetime.now())
            store_id = st.text_input("Store ID", value="S001")
            product_id = st.text_input("Product ID", value="P0001")

        with col2:
            region = st.text_input("Region", value="North")
            category = st.text_input("Category", value="Electronics")
            weather = st.text_input("Weather Condition", value="Sunny")

        with col3:
            price = st.number_input("Price", value=100.0, min_value=0.0)
            discount = st.number_input(
                "Discount (%)", value=0.0, min_value=0.0, max_value=100.0
            )
            promotion = st.checkbox("Promotion", value=False)

        competitor_pricing = st.number_input(
            "Competitor Pricing", value=100.0, min_value=0.0
        )
        seasonality = st.text_input("Seasonality", value="Summer")
        epidemic = st.selectbox("Epidemic (0/1)", options=[0, 1], index=0)

        submit = st.form_submit_button("Predict")

    if submit:
        try:
            data = DemandData(
                Date=datetime.combine(date, datetime.min.time()),
                StoreID=store_id,
                ProductID=product_id,
                Category=category,
                Region=region,
                Price=float(price),
                Discount=float(discount),
                WeatherCondition=weather,
                Promotion=bool(promotion),
                CompetitorPricing=float(competitor_pricing),
                Seasonality=seasonality,
                Epidemic=int(epidemic),
            )

            pipeline = (
                PredictPipeline(
                    ordering_cost=ordering_cost,
                    holding_cost=holding_cost,
                    lead_time_days=lead_time_days,
                    safety_stock=safety_stock,
                    enable_optimization=bool(enable_optimization),
                )
                if enable_optimization
                else PredictPipeline()
            )

            result = pipeline.predict([data.to_model_input()])
            st.success("Prediction completed")
            cols = st.columns(3)
            cols[0].metric("Possible Demand", f"{result['PredictedDemand'].iloc[0]: .2f} units")
            cols[1].metric("Order Quantity", f"{result['OptimalOrderQuantity'].iloc[0]: .2f} units")
            cols[2].metric("Order Threashold", f"{result['SafetyStock'].iloc[0]} units")

            st.download_button(
                "Download prediction CSV",
                data=to_csv_bytes(result),
                file_name="prediction.csv",
                mime="text/csv",
            )

        except Exception as e:
            st.error(f"Prediction failed: {e}")


with tab2:
    st.subheader("Batch predictions from CSV")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)

            pipeline = (
                PredictPipeline(
                    ordering_cost=ordering_cost,
                    holding_cost=holding_cost,
                    lead_time_days=lead_time_days,
                    safety_stock=safety_stock,
                    enable_optimization=bool(enable_optimization),
                )
                if enable_optimization
                else PredictPipeline()
            )

            result = pipeline.predict(df)
            st.success(f"Predictions completed: {len(result)} rows")
            st.dataframe(result)

            st.download_button(
                "Download predictions CSV",
                data=to_csv_bytes(result),
                file_name="predictions.csv",
                mime="text/csv",
            )

        except Exception as e:
            st.error(f"Batch prediction failed: {e}")
