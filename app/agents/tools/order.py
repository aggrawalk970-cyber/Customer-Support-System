from langchain_core.tools import tool
from app.db.connection import SessionLocal
from app.db.models import Order

@tool
def get_order_status(order_id: str) -> str:
    """Check the shipping and status of an order using its order ID.
    
    Args:
        order_id: The order ID string (e.g. ORD12345)
    """
    db = SessionLocal()
    try:
        order = db.query(Order).filter(Order.order_id == order_id).first()
        if order:
            return (
                f"Order ID: {order.order_id}\n"
                f"Email: {order.user_email}\n"
                f"Item: {order.item_name}\n"
                f"Price: ${order.price:.2f}\n"
                f"Status: {order.status}\n"
                f"Est. Delivery Date: {order.delivery_date or 'N/A'}"
            )
        else:
            return f"Order with ID '{order_id}' was not found in our database."
    except Exception as e:
        return f"Error retrieving order status: {str(e)}"
    finally:
        db.close()
