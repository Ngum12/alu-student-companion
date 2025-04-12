import os
import sys

# Set environment variables for Hugging Face
os.environ["TRANSFORMERS_CACHE"] = "/tmp/model_cache"
os.environ["HF_HOME"] = "/tmp/model_cache"
os.environ["SENTENCE_TRANSFORMERS_HOME"] = "/tmp/model_cache"
os.environ["PYTHONUNBUFFERED"] = "1"

# Print startup diagnostic info
print("=== STARTUP: Beginning application initialization ===")
print(f"=== STARTUP: PORT environment variable: {os.environ.get('PORT')} ===")

# First import just the app from main
from main import app

# THEN import other components
from main import conversation_memory
from data_integration.alu_api_connector import ALUDataConnector
from analytics.conversation_analytics import ConversationAnalytics

@app.get("/api/alu-events")
async def get_alu_events(campus: str = "all", days: int = 7):
    """Get upcoming events at ALU"""
    try:
        alu_connector = ALUDataConnector()
        events = alu_connector.get_upcoming_events(campus, days)
        return {"events": events}
    except Exception as e:
        print(f"Error fetching ALU events: {e}")
        return {"events": [], "error": "Could not fetch events"}

@app.get("/api/analytics/dashboard")
async def get_analytics_dashboard():
    """Get analytics dashboard data"""
    try:
        if not conversation_memory:
            return {"error": "Conversation memory not initialized"}
            
        analytics = ConversationAnalytics(conversation_memory)
        dashboard_data = analytics.generate_dashboard_data()
        return dashboard_data
    except Exception as e:
        print(f"Error generating analytics dashboard: {e}")
        return {"error": f"Could not generate analytics: {str(e)}"}

# This is needed for Hugging Face Spaces
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 7860))  # Hugging Face uses port 7860
    print(f"Starting server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)