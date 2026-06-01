import sys
import math
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
        """
        Calculate Economic Order Quantity (EOQ) to minimize total inventory costs.
        
        EOQ = sqrt((2 * D * S) / H)
        """
        try:
            if annual_demand <= 0 or ordering_cost < 0 or holding_cost <= 0:
                raise ValueError(
                    f"Invalid parameters: annual_demand={annual_demand}, "
                    f"ordering_cost={ordering_cost}, holding_cost={holding_cost}. "
                    "All values must be positive."
                )
            
            eoq = ((2 * annual_demand * ordering_cost) / holding_cost) ** 0.5
            eoq_rounded = round(eoq, decimals)
            
            logging.info(f"EOQ calculated: {eoq_rounded} units.")
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
        """
        Calculate Reorder Point (ROP) - inventory level to trigger new orders.
        
        ROP = (Daily Demand × Lead Time) + Safety Stock
        """
        try:
            if daily_demand < 0 or lead_time_days < 0 or safety_stock < 0:
                raise ValueError("All values must be non-negative.")
            
            rop = (daily_demand * lead_time_days) + safety_stock
            rop_rounded = round(rop, decimals)
            
            logging.info(f"Reorder point calculated: {rop_rounded} units.")
            return rop_rounded
            
        except Exception as e:
            raise CustomException(e, sys)

    @staticmethod
    def calculate_safety_stock(
        service_level: float = 0.95,
        average_lead_time_days: Union[int, float] = 1,
        demand_std: Union[int, float] = 0,
        lead_time_std: Union[int, float] = 0,
        average_daily_demand: Union[int, float] = 0,
        decimals: int = 2,
    ) -> float:
        """
        Calculate safety stock using the comprehensive service level approach.
        Accounts for both demand variability and lead time variability.
        
        Formula:
        Safety Stock = Z * sqrt((lead_time * demand_std^2) + (daily_demand^2 * lead_time_std^2))
        
        Args:
            service_level: Desired service cycle probability (e.g., 0.95 for 95%)
            average_lead_time_days: Expected supplier lead time (L)
            demand_std: Standard deviation of daily demand (σ_D)
            lead_time_std: Standard deviation of supplier lead time (σ_L)
            average_daily_demand: Average daily demand (D)
            decimals: Number of decimal places to round to
        """
        try:
            if service_level <= 0 or service_level >= 1:
                raise ValueError("Service level must be strictly between 0 and 1.")
                
            # Expanded Z-score lookup table for common corporate service levels
            z_scores = {
                0.80: 0.84,
                0.85: 1.04,
                0.90: 1.28,
                0.95: 1.645,
                0.98: 2.05,
                0.99: 2.33,
                0.999: 3.09
            }
            
            # Fallback to 95% if an uncommon service level is requested
            z_score = z_scores.get(service_level, 1.645)
            if service_level not in z_scores:
                logging.warning(f"Service level {service_level} not mapped. Defaulting to Z=1.645 (95%).")

            # Calculate variances
            demand_variance_component = average_lead_time_days * (demand_std ** 2)
            lead_time_variance_component = (average_daily_demand ** 2) * (lead_time_std ** 2)
            
            # Combined standard deviation
            combined_std = math.sqrt(demand_variance_component + lead_time_variance_component)
            
            safety_stock = z_score * combined_std
            safety_stock_rounded = round(safety_stock, decimals)
            
            logging.info(f"Safety stock calculated: {safety_stock_rounded} units (Service Level: {service_level})")
            return safety_stock_rounded
            
        except Exception as e:
            raise CustomException(e, sys)


# Backward compatible functions
def economic_order_quantity(demand, ordering_cost, holding_cost):
    return InventoryOptimizer.economic_order_quantity(demand, ordering_cost, holding_cost)


def reorder_point(daily_demand, lead_time, safety_stock=0):
    return InventoryOptimizer.reorder_point(daily_demand, lead_time, safety_stock)