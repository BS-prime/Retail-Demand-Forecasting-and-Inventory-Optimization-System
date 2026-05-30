import sys
from statistics import NormalDist
from typing import Union

from src.exception import CustomException
from src.logger import logging


class InventoryOptimizer:
    """
    A comprehensive inventory optimization system using EOQ and reorder point models.
    
    This class provides methods to calculate:
    - Economic Order Quantity (EOQ): Optimal order size minimizing total inventory costs
    - Reorder Point (ROP): Inventory level at which new orders should be placed
    """

    @staticmethod
    def economic_order_quantity(
        annual_demand: Union[int, float],
        ordering_cost: Union[int, float],
        holding_cost: Union[int, float],
        decimals: int = 2,
    ) -> float:
        try:
            if annual_demand <= 0 or ordering_cost <= 0 or holding_cost <= 0:
                raise ValueError(
                    f"Invalid parameters: annual_demand={annual_demand}, "
                    f"ordering_cost={ordering_cost}, holding_cost={holding_cost}. "
                    "All values must be positive."
                )
            
            eoq = ((2 * annual_demand * ordering_cost) / holding_cost) ** 0.5
            eoq_rounded = round(eoq, decimals)
            
            logging.info(
                f"EOQ calculated: {eoq_rounded} units "
                f"(demand={annual_demand}, order_cost={ordering_cost}, hold_cost={holding_cost})"
            )
            
            return eoq_rounded
            
        except Exception as e:
            raise CustomException(e, sys)

    @staticmethod
    def reorder_point(
        daily_demand: Union[int, float],
        lead_time_days: Union[int, float],
        safety_stock: Union[int, float] = 0,
        decimals: int = 2,
    ) -> float:
        try:
            if daily_demand < 0 or lead_time_days < 0 or safety_stock < 0:
                raise ValueError(
                    f"Invalid parameters: daily_demand={daily_demand}, "
                    f"lead_time_days={lead_time_days}, safety_stock={safety_stock}. "
                    "All values must be non-negative."
                )
            
            rop = (daily_demand * lead_time_days) + safety_stock
            rop_rounded = round(rop, decimals)
            
            logging.info(
                f"Reorder point calculated: {rop_rounded} units "
                f"(daily_demand={daily_demand}, lead_time={lead_time_days}, safety_stock={safety_stock})"
            )
            
            return rop_rounded
            
        except Exception as e:
            raise CustomException(e, sys)

    @staticmethod
    def calculate_safety_stock(
        service_level: float = 0.95,
        demand_std: Union[int, float] = 0,
        lead_time_std: Union[int, float] = 0,
        average_daily_demand: Union[int, float] = 0,
        lead_time_days: Union[int, float] = 1,
        decimals: int = 2,
    ) -> float:
        try:
            if not 0 < service_level < 1:
                raise ValueError("service_level must be between 0 and 1.")

            if (
                demand_std < 0
                or lead_time_std < 0
                or average_daily_demand < 0
                or lead_time_days < 0
            ):
                raise ValueError(
                    "demand_std, lead_time_std, average_daily_demand, and "
                    "lead_time_days must be non-negative."
                )

            z_score = NormalDist().inv_cdf(service_level)
            variance = (lead_time_days * demand_std**2) + (
                average_daily_demand**2 * lead_time_std**2
            )
            safety_stock = round(z_score * variance**0.5, decimals)

            logging.info(
                f"Safety stock calculated: {safety_stock} units "
                f"(service level: {service_level})"
            )

            return safety_stock
            
        except Exception as e:
            raise CustomException(e, sys)
