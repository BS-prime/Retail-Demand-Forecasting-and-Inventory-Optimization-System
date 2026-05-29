def economic_order_quantity(demand, ordering_cost, holding_cost):
    """
    To calculate optimal inventory size minimizing.
    """

    eoq = ((2 * demand * ordering_cost) / holding_cost) ** 0.5

    return round(eoq, 2)


def reorder_point(daily_demand, lead_time, safety_stock):
    """
    To calculate when inventory should be replenished.
    """
    return daily_demand * lead_time + safety_stock
