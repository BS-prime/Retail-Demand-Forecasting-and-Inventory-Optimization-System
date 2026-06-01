from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class DemandData(BaseModel):
    Date: datetime = Field(default_factory=datetime.now)
    StoreID: Literal["S001", "S002", "S003", "S004", "S005"] = "S001"
    ProductID: Literal[
        "P0001",
        "P0002",
        "P0003",
        "P0004",
        "P0005",
        "P0006",
        "P0007",
        "P0008",
        "P0009",
        "P0010",
        "P0011",
        "P0012",
        "P0013",
        "P0014",
        "P0015",
        "P0016",
        "P0017",
        "P0018",
        "P0019",
        "P0020",
    ] = "P0001"
    Category: Literal["Electronics", "Clothing", "Groceries", "Toys", "Furniture"] = (
        "Clothing"
    )
    Region: Literal["North", "South", "East", "West"] = "East"
    Price: float = Field(default=100.0, gt=0)
    Discount: float = Field(default=0.0, ge=0, lt=100)
    WeatherCondition: Literal["Snowy", "Cloudy", "Sunny", "Rainy"] = "Cloudy"
    Promotion: bool = False
    CompetitorPricing: float = Field(default=100.0, ge=0)
    Seasonality: Literal["Winter", "Spring", "Summer", "Autumn"] = "Autumn"
    Epidemic: int = Field(default=0, ge=0, le=1)

    def to_model_input(self) -> dict:
        return {
            "Date": self.Date,
            "Store ID": self.StoreID,
            "Product ID": self.ProductID,
            "Category": self.Category,
            "Region": self.Region,
            "Price": self.Price,
            "Discount": self.Discount,
            "Weather Condition": self.WeatherCondition,
            "Promotion": self.Promotion,
            "Competitor Pricing": self.CompetitorPricing,
            "Seasonality": self.Seasonality,
            "Epidemic": self.Epidemic,
        }


class OptimizationInput(BaseModel):
    ordering_cost: float = Field(gt=0)
    holding_cost: float = Field(gt=0)
    lead_time_days: int = Field(ge=0)
    safety_stock: float = Field(default=0, ge=0)


class PredictionRequest(BaseModel):
    data: DemandData
    optimization: OptimizationInput | None = None


class BatchPredictionRequest(BaseModel):
    data: list[DemandData] = Field(min_length=1)
    optimization: OptimizationInput | None = None


class PredictionResponse(BaseModel):
    predicted_demand: float
    daily_demand: float | None = None
    annual_demand: float | None = None
    optimal_order_quantity: float | None = None
    reorder_point: float | None = None
    safety_stock: float | None = None
