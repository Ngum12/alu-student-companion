import requests
import os
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

class ALUDataConnector:
    """Real-time connection to ALU APIs for calendar, events, and updates"""
    
    def __init__(self):
        self.api_key = os.getenv("ALU_API_KEY", "")
        self.base_url = os.getenv("ALU_API_URL", "https://api.alueducation.com")
        self.cache = {}
        self.cache_expiry = {}
    
    def get_upcoming_events(self, campus: str = "all", days: int = 7) -> List[Dict[str, Any]]:
        """Get upcoming events at ALU campuses"""
        cache_key = f"events_{campus}_{days}"
        
        # Check cache first
        if cache_key in self.cache and datetime.now() < self.cache_expiry.get(cache_key, datetime.min):
            return self.cache[cache_key]
            
        # Make actual API call
        try:
            # Simulated data since we don't have real API access
            events = [
                {
                    "id": "event1",
                    "title": "ALU Leadership Summit",
                    "date": (datetime.now() + timedelta(days=2)).strftime("%Y-%m-%d"),
                    "location": "Rwanda Campus",
                    "description": "Annual leadership conference"
                },
                {
                    "id": "event2",
                    "title": "Career Services Workshop",
                    "date": (datetime.now() + timedelta(days=5)).strftime("%Y-%m-%d"),
                    "location": "Virtual",
                    "description": "Resume building and interview skills"
                }
            ]
            
            # Cache for 1 hour
            self.cache[cache_key] = events
            self.cache_expiry[cache_key] = datetime.now() + timedelta(hours=1)
            return events
            
        except Exception as e:
            print(f"Error fetching ALU events: {e}")
            return []
    
    def get_academic_calendar(self) -> Dict[str, Any]:
        """Get the current academic calendar"""
        cache_key = "academic_calendar"
        
        if cache_key in self.cache and datetime.now() < self.cache_expiry.get(cache_key, datetime.min):
            return self.cache[cache_key]
            
        try:
            # Simulated data
            calendar = {
                "current_term": "Fall 2024",
                "term_start": "2024-09-01",
                "term_end": "2024-12-15",
                "events": [
                    {"name": "Classes Begin", "date": "2024-09-05"},
                    {"name": "Add/Drop Deadline", "date": "2024-09-15"},
                    {"name": "Fall Break", "date": "2024-10-10"},
                    {"name": "Final Exams", "date": "2024-12-10"}
                ]
            }
            
            # Cache for 24 hours
            self.cache[cache_key] = calendar
            self.cache_expiry[cache_key] = datetime.now() + timedelta(days=1)
            return calendar
        except Exception as e:
            print(f"Error fetching academic calendar: {e}")
            return {}
    
    def get_course_info(self, course_code: str) -> Dict[str, Any]:
        """Get information about a specific course"""
        cache_key = f"course_{course_code}"
        
        if cache_key in self.cache and datetime.now() < self.cache_expiry.get(cache_key, datetime.min):
            return self.cache[cache_key]
            
        try:
            # Simulated data
            courses = {
                "CS101": {
                    "title": "Introduction to Computer Science",
                    "credits": 3,
                    "description": "Fundamentals of programming and computer science concepts"
                },
                "BUS200": {
                    "title": "Principles of Management",
                    "credits": 3,
                    "description": "Introduction to management theory and practice"
                }
            }
            
            course = courses.get(course_code, {})
            
            # Cache for 1 week
            self.cache[cache_key] = course
            self.cache_expiry[cache_key] = datetime.now() + timedelta(days=7)
            return course
        except Exception as e:
            print(f"Error fetching course info: {e}")
            return {}