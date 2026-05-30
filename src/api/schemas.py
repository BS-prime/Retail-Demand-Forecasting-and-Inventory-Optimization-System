from datetime import datetime

from pydantic import BaseModel, Field


class DemandData(BaseModel):
    Date: datetime = Field(default_factory=datetime.now)
    StoreID: str = "S001"
    ProductID: str = "P0001"
    Category: str = "Electronics"
    Region: str = "North"
    InventoryLevel: int = Field(default=100, ge=0)
    UnitsSold: int = Field(default=0, ge=0)
    UnitsOrdered: int = Field(default=0, ge=0)
    Price: float = Field(default=100.0, ge=0)
    Discount: float = Field(default=0.0, ge=0, le=100)
    WeatherCondition: str = "Sunny"
    Promotion: bool = False
    CompetitorPricing: float = Field(default=100.0, ge=0)
    Seasonality: str = "Summer"
    Epidemic: int = Field(default=0, ge=0, le=1)

    def to_model_input(self) -> dict:
        return {
            "Date": self.Date,
            "Store ID": self.StoreID,
            "Product ID": self.ProductID,
            "Category": self.Category,
            "Region": self.Region,
            "Inventory Level": self.InventoryLevel,
            "Units Sold": self.UnitsSold,
            "Units Ordered": self.UnitsOrdered,
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
    lead_time_days: float = Field(ge=0)
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
