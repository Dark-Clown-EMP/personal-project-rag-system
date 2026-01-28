from backend.core.database import get_db_connection
from backend.dtos.query_dto import FeedbackRequest

class FeedbackService:
    @staticmethod
    def save_feedback(data: FeedbackRequest):
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO feedback (user_query, ai_response, rating) VALUES (?, ?, ?)",
                (data.query, data.response, data.rating)
            )
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            print(f"❌ [DB ERROR] {e}")
            raise e