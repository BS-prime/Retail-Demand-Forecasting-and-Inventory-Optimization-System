import pandas as pd
from src.optimization_strategy.optimization import InventoryOptimizer


class InventoryService:
    """
    Translates raw machine learning forecasts into actionable logistics metrics
    by safely handling varying time horizons.
    """

    def __init__(
        self,
        ordering_cost: float,
        holding_cost: float,
        lead_time_days: int,
        safety_stock: float = 0,
    ):
        self.ordering_cost = ordering_cost
        self.holding_cost = holding_cost
        self.lead_time_days = lead_time_days
        self.safety_stock = safety_stock
        self.optimizer = InventoryOptimizer()

    def process_operational_plan(
        self, prediction_df: pd.DataFrame, days_in_horizon: int = 365
    ) -> pd.DataFrame:
        """
        Takes machine learning predictions, standardizes them across the given
        time horizon, and appends valid operational insights.
        """
        output_df = prediction_df.copy()

        # Step 1: Safely derive daily run-rates based on actual horizon length (fixes the 12x month bug)
        output_df["DailyDemand"] = output_df["PredictedDemand"] / days_in_horizon

        # Step 2: Scale up to annual specifically for the EOQ formula (since holding costs are annual)
        output_df["AnnualizedDemandEquivalent"] = output_df["DailyDemand"] * 365

        # Step 3: Run Logistics Calculations
        output_df["ReorderPoint"] = output_df["DailyDemand"].apply(
            lambda x: self.optimizer.reorder_point(
                daily_demand=x,
                lead_time_days=self.lead_time_days,
                safety_stock=self.safety_stock,
            )
        )

        # EOQ requires the annualized sum of total expected product run rates
        total_annualized_demand = output_df["AnnualizedDemandEquivalent"].sum()
        output_df["OptimalOrderQuantity"] = self.optimizer.economic_order_quantity(
            annual_demand=total_annualized_demand,
            ordering_cost=self.ordering_cost,
            holding_cost=self.holding_cost,
        )

        output_df["SafetyStock"] = self.safety_stock
        return output_df
