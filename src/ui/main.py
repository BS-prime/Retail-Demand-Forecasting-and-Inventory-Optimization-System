import streamlit as st
import pandas as pd
from datetime import datetime

from src.pipeline.predict_pipeline import PredictPipeline
from src.services.inventory_service import InventoryService
from src.api.schemas import DemandData


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


st.set_page_config(page_title="Retail Demand Forecasting", layout="wide")
st.title("Retail Demand Forecasting — Inference & Optimization Architecture")

# Shared Operational Constraints via Sidebar
with st.sidebar:
    st.header("Logistics Controls")
    ordering_cost = st.number_input("Ordering Cost ($)", value=50.0, min_value=0.1)
    holding_cost = st.number_input(
        "Holding Cost (Annual per unit)", value=2.0, min_value=0.1
    )
    lead_time_days = st.number_input("Supplier Lead Time (Days)", value=7, min_value=0)
    safety_stock = st.number_input("Safety Stock Parameter", value=0.0, min_value=0.0)

    st.markdown("---")
    horizon_selection = st.selectbox(
        "Prediction Timeframe Horizon",
        options=["Annual (365 Days)", "Monthly (30 Days)", "Weekly (7 Days)"],
        index=0,
    )
    horizon_mapping = {
        "Annual (365 Days)": 365,
        "Monthly (30 Days)": 30,
        "Weekly (7 Days)": 7,
    }
    days_in_horizon = horizon_mapping[horizon_selection]

tab1, tab2 = st.tabs(["Single Point Prediction", "Batch File Processing"])

# Instantiate Predictor and Logistics Service with shared parameters
predictor = PredictPipeline()
logistics_service = InventoryService(
    ordering_cost=ordering_cost,
    holding_cost=holding_cost,
    lead_time_days=int(lead_time_days),
    safety_stock=safety_stock,
)

with tab1:
    st.subheader("Single Target Assessment")
    with st.form("single_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            date = st.date_input("Execution Date", value=datetime.now())
            store_id = st.selectbox(
                "Store ID", ["S001", "S002", "S003", "S004", "S005"]
            )
            product_id = st.text_input("Product Identifier", value="P0001")
        with col2:
            region = st.selectbox("Region Focus", ["North", "South", "East", "West"])
            category = st.selectbox(
                "Category Group",
                ["Electronics", "Clothing", "Groceries", "Toys", "Furniture"],
            )
            weather = st.selectbox(
                "Weather Matrix", ["Sunny", "Cloudy", "Rainy", "Snowy"]
            )
        with col3:
            price = st.number_input("Unit Retail Price", value=100.0, min_value=0.0)
            discount = st.number_input(
                "Discount (%)", value=0.0, min_value=0.0, max_value=100.0
            )
            promotion = st.checkbox("Promotion", value=False)

        competitor_pricing = st.number_input(
            "Competitor Pricing Index", value=100.0, min_value=0.0
        )
        seasonality = st.selectbox(
            "Season Context", ["Winter", "Spring", "Summer", "Autumn"]
        )
        epidemic = st.selectbox("Disruption Event (0/1)", options=[0, 1], index=0)

        submit = st.form_submit_button("Run Analysis")

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

            # Step 1: Pure ML Inference
            raw_predictions = predictor.predict([data.to_model_input()])

            # Step 2: OR Optimization through independent service
            final_plan = logistics_service.process_operational_plan(
                raw_predictions, days_in_horizon=days_in_horizon
            )

            st.success("Operational tracking complete.")
            cols = st.columns(4)
            cols[0].metric(
                "Forecasted Horizon Demand",
                f"{final_plan['PredictedDemand'].iloc[0]: .2f} units",
            )
            cols[1].metric(
                "Daily Demand Run Rate",
                f"{final_plan['DailyDemand'].iloc[0]: .2f} units/day",
            )
            cols[2].metric(
                "Economic Order Size (EOQ)",
                f"{final_plan['OptimalOrderQuantity'].iloc[0]: .2f} units",
            )
            cols[3].metric(
                "Reorder Threshold (ROP)",
                f"{final_plan['ReorderPoint'].iloc[0]: .2f} units",
            )

        except Exception as e:
            st.error(f"Execution Error Encountered: {e}")

with tab2:
    st.subheader("Mass CSV File Inference Processing")
    uploaded = st.file_uploader("Upload Target Inventory Frame", type=["csv"])
    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)

            # Streamlined process sequence
            raw_predictions = predictor.predict(df)
            final_plan = logistics_service.process_operational_plan(
                raw_predictions, days_in_horizon=days_in_horizon
            )

            st.success(
                f"Batch execution complete for {len(final_plan)} active records."
            )
            st.dataframe(final_plan)

            st.download_button(
                "Export Full Optimization Data",
                data=to_csv_bytes(final_plan),
                file_name="optimized_inventory_plan.csv",
                mime="text/csv",
            )
        except Exception as e:
            st.error(f"Batch Processing Failure: {e}")
