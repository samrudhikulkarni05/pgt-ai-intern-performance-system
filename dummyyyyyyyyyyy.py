# requirements.txt
"""
streamlit==1.40.0
plotly==5.24.0
pandas==2.2.2
google-generativeai==0.8.2
python-dotenv==1.0.1
sqlite3
"""

# main.py
import streamlit as st
import sqlite3
import json
import time
import hashlib
from datetime import datetime, date, timedelta
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import google.generativeai as genai
import os
from dotenv import load_dotenv
import uuid
import random

# Load environment variables
load_dotenv()

# --- Database Setup ---
def init_db():
    """Initialize SQLite database with all required tables"""
    conn = sqlite3.connect('interntrack.db', check_same_thread=False)
    cursor = conn.cursor()
    
    # Jobs table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS jobs (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            domain TEXT NOT NULL,
            description TEXT,
            required_skills TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Users table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT,
            assigned_job_id TEXT,
            skills TEXT,
            onboarded BOOLEAN DEFAULT 0,
            analysis TEXT,
            performance_metrics TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (assigned_job_id) REFERENCES jobs (id)
        )
    ''')
    
    # Attendance table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS attendance (
            id TEXT PRIMARY KEY,
            intern_id TEXT NOT NULL,
            date TEXT NOT NULL,
            time_in TEXT NOT NULL,
            time_out TEXT,
            task TEXT,
            resources TEXT,
            duration INTEGER,
            score INTEGER,
            status TEXT,
            quiz_results TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (intern_id) REFERENCES users (id)
        )
    ''')
    
    # Performance metrics table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS performance_metrics (
            id TEXT PRIMARY KEY,
            intern_id TEXT NOT NULL,
            date TEXT NOT NULL,
            metric_type TEXT NOT NULL,
            value REAL NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (intern_id) REFERENCES users (id)
        )
    ''')
    
    # Insert sample jobs if empty
    cursor.execute("SELECT COUNT(*) FROM jobs")
    if cursor.fetchone()[0] == 0:
        sample_jobs = [
            ('job-1', 'Frontend Developer', 'Web Development', 
             'Advanced UI engineering with React, TypeScript, and high-performance rendering patterns.',
             json.dumps([{"name": "React", "minLevel": 4}, {"name": "TypeScript", "minLevel": 3}, 
                        {"name": "JavaScript", "minLevel": 4}, {"name": "CSS", "minLevel": 3},
                        {"name": "HTML", "minLevel": 3}])),
            ('job-2', 'AI Research Associate', 'Machine Learning',
             'Development of neural architectures, Python-based pipelines, and statistical modeling.',
             json.dumps([{"name": "Python", "minLevel": 5}, {"name": "Machine Learning", "minLevel": 4},
                        {"name": "Statistics", "minLevel": 4}, {"name": "TensorFlow", "minLevel": 3},
                        {"name": "Data Analysis", "minLevel": 4}])),
            ('job-3', 'DevOps Engineer', 'Cloud & Infrastructure',
             'Cloud infrastructure management, CI/CD pipelines, and container orchestration.',
             json.dumps([{"name": "AWS", "minLevel": 4}, {"name": "Docker", "minLevel": 4},
                        {"name": "Kubernetes", "minLevel": 3}, {"name": "Linux", "minLevel": 4},
                        {"name": "Networking", "minLevel": 3}]))
        ]
        cursor.executemany('''
            INSERT INTO jobs (id, title, domain, description, required_skills)
            VALUES (?, ?, ?, ?, ?)
        ''', sample_jobs)
    
    conn.commit()
    return conn

# Database connection
db_conn = init_db()

# --- AI Service ---
class AIService:
    def __init__(self):
        genai.configure(api_key=os.getenv("API_KEY"))
        self.model_pro = genai.GenerativeModel('gemini-1.5-pro-latest')
        self.model_flash = genai.GenerativeModel('gemini-1.5-flash-latest')
    
    def get_analysis(self, job: Dict, user_skills: List[Dict]) -> Dict:
        """Perform comprehensive skill gap analysis"""
        prompt = f"""Perform a detailed skill gap analysis for the role: {job['title']}.
        
        ROLE REQUIREMENTS:
        - Job Description: {job['description']}
        - Required Skills (with minimum levels): {json.dumps(job['required_skills'], indent=2)}
        
        INTERN'S CURRENT SKILLS:
        {json.dumps(user_skills, indent=2)}
        
        Provide a comprehensive analysis including:
        1. Similarity percentage (0-100%)
        2. Identified skill gaps with detailed explanations
        3. Learning roadmap with priorities
        4. EXACTLY 5 REAL working YouTube video links (beginner to advanced)
        5. EXACTLY 5 REAL documentation links (official documentation, tutorials, guides)
        
        IMPORTANT: All URLs MUST be real, working links. Do not make up URLs.
        
        Return JSON with this exact structure:
        {{
            "similarity": 75,
            "gaps": [
                {{
                    "skill": "skill_name",
                    "currentLevel": 2,
                    "requiredLevel": 4,
                    "gapLevel": 2,
                    "reason": "Detailed explanation of the gap",
                    "priority": "HIGH/MEDIUM/LOW",
                    "estimatedImprovementTime": "2-4 weeks"
                }}
            ],
            "recommendations": {{
                "videos": [
                    {{
                        "title": "Video title",
                        "url": "https://www.youtube.com/watch?v=...",
                        "duration": "1:30:00",
                        "level": "Beginner/Intermediate/Advanced",
                        "description": "Brief description"
                    }}
                ],
                "documentation": [
                    {{
                        "title": "Documentation title",
                        "url": "https://official-docs.com/...",
                        "type": "Official Docs/Tutorial/Guide",
                        "description": "Brief description"
                    }}
                ]
            }},
            "learningPath": [
                {{
                    "week": 1,
                    "focus": "Topic to focus on",
                    "resources": ["Resource 1", "Resource 2"],
                    "milestone": "What to achieve"
                }}
            ]
        }}"""
        
        try:
            response = self.model_pro.generate_content(prompt)
            text = response.text.strip()
            
            # Clean and parse JSON
            if text.startswith('```json'):
                text = text[7:]
            if text.startswith('```'):
                text = text[3:]
            if text.endswith('```'):
                text = text[:-3]
            
            text = text.strip()
            
            try:
                analysis = json.loads(text)
                
                # Validate URLs are present
                if 'recommendations' in analysis:
                    if 'videos' in analysis['recommendations']:
                        for video in analysis['recommendations']['videos']:
                            if not video.get('url', '').startswith('http'):
                                raise ValueError("Invalid video URL")
                    if 'documentation' in analysis['recommendations']:
                        for doc in analysis['recommendations']['documentation']:
                            if not doc.get('url', '').startswith('http'):
                                raise ValueError("Invalid documentation URL")
                
                return analysis
                
            except json.JSONDecodeError:
                import re
                json_match = re.search(r'\{.*\}', text, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group())
                else:
                    raise
                    
        except Exception as e:
            print(f"AI Analysis Error: {e}")
            # Enhanced fallback response
            return self.get_fallback_analysis(job, user_skills)
    
    def get_fallback_analysis(self, job: Dict, user_skills: List[Dict]) -> Dict:
        """Provide detailed fallback analysis"""
        skill_names = [skill.get('name', '').lower() for skill in user_skills]
        required_skills = job['required_skills']
        
        # Calculate similarity
        matched = 0
        gaps = []
        
        for req in required_skills:
            req_name = req['name'].lower()
            req_level = req.get('minLevel', 3)
            
            # Find matching user skill
            user_skill = next((s for s in user_skills if s.get('name', '').lower() == req_name), None)
            
            if user_skill:
                user_level = user_skill.get('level', 0)
                gap = max(0, req_level - user_level)
                
                if gap > 0:
                    gaps.append({
                        "skill": req['name'],
                        "currentLevel": user_level,
                        "requiredLevel": req_level,
                        "gapLevel": gap,
                        "reason": f"Need to improve from level {user_level} to level {req_level}",
                        "priority": "HIGH" if gap >= 2 else "MEDIUM",
                        "estimatedImprovementTime": f"{gap * 2} weeks"
                    })
                    matched += (user_level / req_level) * 20
                else:
                    matched += 20
            else:
                gaps.append({
                    "skill": req['name'],
                    "currentLevel": 0,
                    "requiredLevel": req_level,
                    "gapLevel": req_level,
                    "reason": "Skill not found in your current skill set",
                    "priority": "HIGH",
                    "estimatedImprovementTime": f"{req_level * 2} weeks"
                })
        
        similarity = min(100, int((matched / len(required_skills)) * 100)) if required_skills else 0
        
        return {
            "similarity": similarity,
            "gaps": gaps[:5],  # Limit to 5 gaps
            "recommendations": {
                "videos": [
                    {
                        "title": "React Tutorial for Beginners",
                        "url": "https://www.youtube.com/watch?v=Ke90Tje7VS0",
                        "duration": "2:15:00",
                        "level": "Beginner",
                        "description": "Complete React tutorial from scratch"
                    },
                    {
                        "title": "TypeScript Crash Course",
                        "url": "https://www.youtube.com/watch?v=BCg4U1FzODs",
                        "duration": "1:45:00",
                        "level": "Intermediate",
                        "description": "Learn TypeScript fundamentals"
                    },
                    {
                        "title": "Advanced React Patterns",
                        "url": "https://www.youtube.com/watch?v=DPbSf8vQWLo",
                        "duration": "1:30:00",
                        "level": "Advanced",
                        "description": "Advanced React concepts and patterns"
                    },
                    {
                        "title": "Modern JavaScript Tutorial",
                        "url": "https://www.youtube.com/watch?v=iLWTnMzWtj4",
                        "duration": "3:20:00",
                        "level": "Beginner",
                        "description": "Complete JavaScript tutorial"
                    },
                    {
                        "title": "CSS Grid & Flexbox Masterclass",
                        "url": "https://www.youtube.com/watch?v=RSIclWvNTdQ",
                        "duration": "2:00:00",
                        "level": "Intermediate",
                        "description": "Modern CSS layout techniques"
                    }
                ],
                "documentation": [
                    {
                        "title": "React Official Documentation",
                        "url": "https://react.dev/learn",
                        "type": "Official Docs",
                        "description": "Official React documentation and tutorials"
                    },
                    {
                        "title": "TypeScript Handbook",
                        "url": "https://www.typescriptlang.org/docs/",
                        "type": "Official Docs",
                        "description": "Complete TypeScript documentation"
                    },
                    {
                        "title": "MDN Web Docs",
                        "url": "https://developer.mozilla.org/en-US/",
                        "type": "Reference",
                        "description": "Web technology references and guides"
                    },
                    {
                        "title": "JavaScript Info",
                        "url": "https://javascript.info/",
                        "type": "Tutorial",
                        "description": "Modern JavaScript tutorial"
                    },
                    {
                        "title": "Frontend Developer Roadmap",
                        "url": "https://roadmap.sh/frontend",
                        "type": "Guide",
                        "description": "Step-by-step frontend development guide"
                    }
                ]
            },
            "learningPath": [
                {
                    "week": 1,
                    "focus": "React Fundamentals",
                    "resources": ["React Docs", "Video Tutorials"],
                    "milestone": "Build basic React components"
                },
                {
                    "week": 2,
                    "focus": "TypeScript Integration",
                    "resources": ["TypeScript Handbook", "Practice Projects"],
                    "milestone": "Type-safe React components"
                }
            ]
        }
    
    def get_daily_quiz(self, task: str, resources: List[str]) -> List[Dict]:
        """Generate comprehensive MCQ quiz"""
        prompt = f"""Generate a 10-question MCQ quiz for task: "{task}".
        Resources studied: {', '.join(resources)}.
        
        Requirements:
        1. Each question should test practical knowledge
        2. Include 4 options per question
        3. Mark the correct answer (0-3 index)
        4. Include explanation for correct answer
        5. Questions should range from basic to advanced
        
        Return JSON array: [
            {{
                "question": "Question text",
                "options": ["Option A", "Option B", "Option C", "Option D"],
                "correctAnswer": 0,
                "explanation": "Explanation of why this is correct",
                "difficulty": "Easy/Medium/Hard"
            }}
        ]"""
        
        try:
            response = self.model_pro.generate_content(prompt)
            text = response.text.strip()
            
            if text.startswith('```json'):
                text = text[7:]
            if text.endswith('```'):
                text = text[:-3]
            
            text = text.strip()
            
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                import re
                json_match = re.search(r'\[.*\]', text, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group())
                else:
                    raise
                    
        except Exception as e:
            print(f"Quiz Generation Error: {e}")
            return self.get_fallback_quiz(task)
    
    def get_fallback_quiz(self, task: str) -> List[Dict]:
        """Provide fallback quiz questions"""
        quiz_templates = {
            "react": [
                {
                    "question": "What is the primary purpose of React?",
                    "options": [
                        "Building user interfaces",
                        "Database management",
                        "Server-side rendering only",
                        "Mobile app development without JavaScript"
                    ],
                    "correctAnswer": 0,
                    "explanation": "React is a JavaScript library for building user interfaces, particularly for single-page applications.",
                    "difficulty": "Easy"
                },
                {
                    "question": "Which hook is used for state management in functional components?",
                    "options": [
                        "useState",
                        "useEffect",
                        "useContext",
                        "All of the above"
                    ],
                    "correctAnswer": 0,
                    "explanation": "useState is specifically designed for state management in functional components.",
                    "difficulty": "Easy"
                }
            ],
            "python": [
                {
                    "question": "What is a Python decorator?",
                    "options": [
                        "A function that modifies another function",
                        "A special type of variable",
                        "A class method",
                        "An import statement"
                    ],
                    "correctAnswer": 0,
                    "explanation": "A decorator is a function that takes another function and extends its behavior without modifying it.",
                    "difficulty": "Medium"
                }
            ]
        }
        
        # Determine quiz type based on task
        task_lower = task.lower()
        if any(keyword in task_lower for keyword in ['react', 'frontend', 'ui']):
            return quiz_templates['react']
        elif any(keyword in task_lower for keyword in ['python', 'ml', 'ai']):
            return quiz_templates['python']
        else:
            return quiz_templates['react']  # Default
    
    def get_performance_analysis(self, attendance_data: List[Dict], skill_data: Dict) -> Dict:
        """Generate performance analysis"""
        if not attendance_data:
            return {
                "overallScore": 0,
                "consistency": 0,
                "improvementRate": 0,
                "strengths": [],
                "weaknesses": [],
                "recommendations": []
            }
        
        scores = [entry['score'] for entry in attendance_data if entry['score']]
        durations = [entry['duration'] for entry in attendance_data]
        
        overall_score = sum(scores) / len(scores) if scores else 0
        
        # Calculate consistency (standard deviation inverse)
        if len(scores) > 1:
            import statistics
            consistency = max(0, 100 - (statistics.stdev(scores) * 20))
        else:
            consistency = 100
        
        # Calculate improvement rate
        if len(scores) > 3:
            recent_avg = sum(scores[-3:]) / 3
            older_avg = sum(scores[:3]) / 3
            improvement_rate = ((recent_avg - older_avg) / older_avg * 100) if older_avg > 0 else 0
        else:
            improvement_rate = 0
        
        # Analyze strengths and weaknesses
        strengths = []
        weaknesses = []
        
        if skill_data and 'gaps' in skill_data:
            for gap in skill_data['gaps'][:3]:
                if gap.get('priority') == 'LOW':
                    strengths.append(gap['skill'])
                else:
                    weaknesses.append(gap['skill'])
        
        return {
            "overallScore": round(overall_score, 1),
            "consistency": round(consistency, 1),
            "improvementRate": round(improvement_rate, 1),
            "strengths": strengths[:3],
            "weaknesses": weaknesses[:3],
            "recommendations": [
                "Focus on consistent daily practice",
                "Review previous quiz mistakes",
                "Apply concepts to small projects"
            ]
        }
    
    def get_feedback(self, topic: str, score: int, duration: int, quiz_results: Dict) -> str:
        """Generate detailed feedback"""
        prompt = f"""Provide constructive feedback for an intern who studied "{topic}" 
        for {duration} minutes and scored {score}/10 on their assessment.
        
        Quiz Performance Details:
        - Total Questions: {quiz_results.get('total_questions', 10)}
        - Correct Answers: {quiz_results.get('correct_answers', score)}
        - Areas of Strength: {', '.join(quiz_results.get('strengths', []))}
        - Areas for Improvement: {', '.join(quiz_results.get('weaknesses', []))}
        
        Provide specific, actionable feedback in 2-3 sentences. Focus on both what was done well and concrete suggestions for improvement."""
        
        try:
            response = self.model_flash.generate_content(prompt)
            return response.text.strip()
        except:
            return f"Good effort on {topic}! Your score of {score}/10 shows understanding, but there's room for improvement. Focus on reviewing incorrect answers and apply the concepts in practice."

# Initialize AI service
ai_service = AIService()

# --- Database Operations ---
class DatabaseService:
    def __init__(self, conn):
        self.conn = conn
    
    def register_intern(self, name: str, email: str, password: str, job_id: str) -> bool:
        """Register new intern"""
        cursor = self.conn.cursor()
        
        # Check if email exists
        cursor.execute("SELECT COUNT(*) FROM users WHERE email = ?", (email,))
        if cursor.fetchone()[0] > 0:
            return False
        
        # Create user
        user_id = str(uuid.uuid4())
        cursor.execute('''
            INSERT INTO users (id, name, email, password, assigned_job_id, onboarded, performance_metrics)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (user_id, name, email, password, job_id, 0, json.dumps({})))
        
        self.conn.commit()
        return True
    
    def login_intern(self, email: str, password: str) -> Optional[Dict]:
        """Login intern with robust column checking"""
        cursor = self.conn.cursor()
        
        # First check if performance_metrics column exists
        cursor.execute("PRAGMA table_info(users)")
        columns = [column[1] for column in cursor.fetchall()]
        
        # Build query based on available columns
        if 'performance_metrics' in columns:
            query = '''
                SELECT id, name, email, assigned_job_id, skills, onboarded, analysis, performance_metrics
                FROM users WHERE email = ? AND password = ?
            '''
        else:
            # Fallback for older databases without performance_metrics
            query = '''
                SELECT id, name, email, assigned_job_id, skills, onboarded, analysis, NULL
                FROM users WHERE email = ? AND password = ?
            '''
        
        cursor.execute(query, (email, password))
        
        row = cursor.fetchone()
        if row:
            return {
                "id": row[0],
                "name": row[1],
                "email": row[2],
                "assigned_job_id": row[3],
                "skills": json.loads(row[4]) if row[4] else [],
                "onboarded": bool(row[5]),
                "analysis": json.loads(row[6]) if row[6] else None,
                "performance_metrics": json.loads(row[7]) if row[7] else None
            }
        return None
    
    def update_intern(self, user: Dict) -> None:
        """Update intern data"""
        cursor = self.conn.cursor()
        
        # Check if performance_metrics column exists
        cursor.execute("PRAGMA table_info(users)")
        columns = [column[1] for column in cursor.fetchall()]
        
        if 'performance_metrics' in columns:
            cursor.execute('''
                UPDATE users 
                SET skills = ?, onboarded = ?, analysis = ?, performance_metrics = ?
                WHERE id = ?
            ''', (
                json.dumps(user['skills']),
                int(user['onboarded']),
                json.dumps(user['analysis']),
                json.dumps(user.get('performance_metrics', {})),
                user['id']
            ))
        else:
            # Fallback for older databases
            cursor.execute('''
                UPDATE users 
                SET skills = ?, onboarded = ?, analysis = ?
                WHERE id = ?
            ''', (
                json.dumps(user['skills']),
                int(user['onboarded']),
                json.dumps(user['analysis']),
                user['id']
            ))
        
        self.conn.commit()
    
    def update_performance_metrics(self, intern_id: str, metrics: Dict) -> None:
        """Update performance metrics"""
        cursor = self.conn.cursor()
        
        # Store in performance_metrics table
        for metric_type, value in metrics.items():
            metric_id = str(uuid.uuid4())
            cursor.execute('''
                INSERT INTO performance_metrics (id, intern_id, date, metric_type, value)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                metric_id,
                intern_id,
                date.today().isoformat(),
                metric_type,
                value
            ))
        
        self.conn.commit()
    
    def get_jobs(self) -> List[Dict]:
        """Get all job roles"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT id, title, domain, description, required_skills FROM jobs ORDER BY created_at DESC")
        rows = cursor.fetchall()
        
        jobs = []
        for row in rows:
            jobs.append({
                "id": row[0],
                "title": row[1],
                "domain": row[2],
                "description": row[3],
                "required_skills": json.loads(row[4])
            })
        return jobs
    
    def get_job_by_id(self, job_id: str) -> Optional[Dict]:
        """Get specific job by ID"""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT id, title, domain, description, required_skills 
            FROM jobs WHERE id = ?
        ''', (job_id,))
        
        row = cursor.fetchone()
        if row:
            return {
                "id": row[0],
                "title": row[1],
                "domain": row[2],
                "description": row[3],
                "required_skills": json.loads(row[4])
            }
        return None
    
    def upsert_job(self, job: Dict) -> None:
        """Create or update job"""
        cursor = self.conn.cursor()
        
        # Check if exists
        cursor.execute("SELECT COUNT(*) FROM jobs WHERE id = ?", (job['id'],))
        if cursor.fetchone()[0] > 0:
            # Update
            cursor.execute('''
                UPDATE jobs SET title = ?, domain = ?, description = ?, required_skills = ?
                WHERE id = ?
            ''', (
                job['title'],
                job['domain'],
                job['description'],
                json.dumps(job['required_skills']),
                job['id']
            ))
        else:
            # Insert
            cursor.execute('''
                INSERT INTO jobs (id, title, domain, description, required_skills)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                job['id'],
                job['title'],
                job['domain'],
                job['description'],
                json.dumps(job['required_skills'])
            ))
        
        self.conn.commit()
    
    def delete_job(self, job_id: str) -> None:
        """Delete job"""
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
        self.conn.commit()
    
    def log_attendance(self, log: Dict) -> str:
        """Log attendance/session and return log ID"""
        cursor = self.conn.cursor()
        log_id = str(uuid.uuid4())
        
        cursor.execute('''
            INSERT INTO attendance (id, intern_id, date, time_in, time_out, task, resources, duration, score, status, quiz_results)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            log_id,
            log['intern_id'],
            log['date'],
            log['time_in'],
            log['time_out'],
            log['task'],
            json.dumps(log['resources']),
            log['duration'],
            log['score'],
            log['status'],
            json.dumps(log.get('quiz_results', {}))
        ))
        
        self.conn.commit()
        return log_id
    
    def get_attendance_for_intern(self, intern_id: str) -> List[Dict]:
        """Get attendance logs for specific intern"""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT id, date, time_in, time_out, task, resources, duration, score, status, quiz_results
            FROM attendance WHERE intern_id = ? ORDER BY date DESC, time_in DESC
        ''', (intern_id,))
        
        rows = cursor.fetchall()
        attendance = []
        for row in rows:
            attendance.append({
                "id": row[0],
                "date": row[1],
                "time_in": row[2],
                "time_out": row[3],
                "task": row[4],
                "resources": json.loads(row[5]),
                "duration": row[6],
                "score": row[7],
                "status": row[8],
                "quiz_results": json.loads(row[9]) if row[9] else {}
            })
        return attendance
    
    def get_all_attendance(self) -> List[Dict]:
        """Get all attendance logs"""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT id, intern_id, date, time_in, time_out, task, resources, duration, score, status, quiz_results
            FROM attendance ORDER BY date DESC, time_in DESC
        ''')
        
        rows = cursor.fetchall()
        attendance = []
        for row in rows:
            attendance.append({
                "id": row[0],
                "intern_id": row[1],
                "date": row[2],
                "time_in": row[3],
                "time_out": row[4],
                "task": row[5],
                "resources": json.loads(row[6]),
                "duration": row[7],
                "score": row[8],
                "status": row[9],
                "quiz_results": json.loads(row[10]) if row[10] else {}
            })
        return attendance
    
    def get_all_interns(self) -> List[Dict]:
        """Get all interns with robust column checking"""
        cursor = self.conn.cursor()
        
        # Check if performance_metrics column exists
        cursor.execute("PRAGMA table_info(users)")
        columns = [column[1] for column in cursor.fetchall()]
        
        if 'performance_metrics' in columns:
            query = '''
                SELECT id, name, email, assigned_job_id, skills, onboarded, analysis, performance_metrics
                FROM users ORDER BY name
            '''
        else:
            # Fallback for older databases
            query = '''
                SELECT id, name, email, assigned_job_id, skills, onboarded, analysis, NULL
                FROM users ORDER BY name
            '''
        
        cursor.execute(query)
        
        rows = cursor.fetchall()
        interns = []
        for row in rows:
            interns.append({
                "id": row[0],
                "name": row[1],
                "email": row[2],
                "assigned_job_id": row[3],
                "skills": json.loads(row[4]) if row[4] else [],
                "onboarded": bool(row[5]),
                "analysis": json.loads(row[6]) if row[6] else None,
                "performance_metrics": json.loads(row[7]) if row[7] else None
            })
        return interns
    
    def get_performance_metrics(self, intern_id: str, days: int = 30) -> List[Dict]:
        """Get performance metrics for an intern"""
        cursor = self.conn.cursor()
        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        
        cursor.execute('''
            SELECT date, metric_type, value 
            FROM performance_metrics 
            WHERE intern_id = ? AND date >= ? 
            ORDER BY date
        ''', (intern_id, start_date))
        
        rows = cursor.fetchall()
        metrics = []
        for row in rows:
            metrics.append({
                "date": row[0],
                "metric_type": row[1],
                "value": row[2]
            })
        return metrics

# Initialize database service
db = DatabaseService(db_conn)

# --- Visualization Components ---
def create_skill_gap_pie(similarity: float):
    """Create skill gap pie chart"""
    fig = go.Figure()
    
    fig.add_trace(go.Pie(
        values=[similarity, max(0, 100 - similarity)],
        labels=['Match', 'Gap'],
        hole=0.6,
        marker_colors=['#6366f1', '#f1f5f9'],
        textinfo='none',
        hoverinfo='label+value+percent',
        showlegend=False
    ))
    
    fig.update_layout(
        showlegend=False,
        margin=dict(t=10, b=10, l=10, r=10),
        height=160,
        width=160,
        annotations=[
            dict(
                text=f'<b>{similarity}%</b><br><span style="font-size:8px">Similarity</span>',
                x=0.5, y=0.5,
                font_size=16,
                showarrow=False,
                font_color='#6366f1'
            )
        ]
    )
    
    return fig

def create_performance_pie_chart(attendance_data: List[Dict]):
    """Create performance distribution pie chart"""
    if not attendance_data:
        return go.Figure()
    
    scores = [entry['score'] for entry in attendance_data if entry['score']]
    
    # Calculate performance categories
    excellent = sum(1 for s in scores if s >= 8)
    good = sum(1 for s in scores if 6 <= s < 8)
    average = sum(1 for s in scores if 4 <= s < 6)
    poor = sum(1 for s in scores if s < 4)
    
    categories = ['Excellent (8-10)', 'Good (6-8)', 'Average (4-6)', 'Poor (<4)']
    values = [excellent, good, average, poor]
    colors = ['#10b981', '#3b82f6', '#f59e0b', '#ef4444']
    
    fig = go.Figure()
    
    fig.add_trace(go.Pie(
        labels=categories,
        values=values,
        hole=0.4,
        marker_colors=colors,
        textinfo='label+percent',
        hoverinfo='value+percent'
    ))
    
    fig.update_layout(
        height=300,
        showlegend=False,
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(t=30, b=20, l=20, r=20)
    )
    
    return fig

def create_score_velocity_chart(attendance_data: List[Dict]):
    """Create score velocity chart with pie chart"""
    if not attendance_data:
        return go.Figure()
    
    # Prepare data for line chart
    dates = [entry['date'] for entry in attendance_data][-15:]  # Last 15 sessions
    scores = [entry['score'] for entry in attendance_data][-15:]
    
    # Create subplots
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'scatter'}, {'type': 'domain'}]],
        subplot_titles=("Score Velocity", "Performance Distribution")
    )
    
    # Line chart for score velocity
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=scores,
            mode='lines+markers',
            name='Scores',
            line=dict(color='#6366f1', width=3),
            marker=dict(size=8, color='#6366f1'),
            fill='tozeroy',
            fillcolor='rgba(99, 102, 241, 0.1)'
        ),
        row=1, col=1
    )
    
    # Add trend line
    if len(scores) > 1:
        import numpy as np
        x_numeric = list(range(len(scores)))
        z = np.polyfit(x_numeric, scores, 1)
        p = np.poly1d(z)
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=p(x_numeric),
                mode='lines',
                name='Trend',
                line=dict(color='#ef4444', width=2, dash='dash')
            ),
            row=1, col=1
        )
    
    # Pie chart for performance distribution
    categories = ['Excellent (8-10)', 'Good (6-8)', 'Average (4-6)', 'Poor (<4)']
    excellent = sum(1 for s in scores if s >= 8)
    good = sum(1 for s in scores if 6 <= s < 8)
    average = sum(1 for s in scores if 4 <= s < 6)
    poor = sum(1 for s in scores if s < 4)
    values = [excellent, good, average, poor]
    colors = ['#10b981', '#3b82f6', '#f59e0b', '#ef4444']
    
    fig.add_trace(
        go.Pie(
            labels=categories,
            values=values,
            hole=0.4,
            marker_colors=colors,
            textinfo='label+percent',
            hoverinfo='value+percent'
        ),
        row=1, col=2
    )
    
    fig.update_layout(
        height=400,
        showlegend=False,
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(t=50, b=20, l=20, r=20)
    )
    
    fig.update_xaxes(title_text="Session Date", row=1, col=1)
    fig.update_yaxes(title_text="Score", range=[0, 10], row=1, col=1)
    
    return fig

def create_performance_analysis_chart(attendance_data: List[Dict], performance_metrics: Dict):
    """Create performance analysis dashboard"""
    if not attendance_data:
        return go.Figure()
    
    # Prepare data
    dates = [entry['date'] for entry in attendance_data][-10:]  # Last 10 sessions
    scores = [entry['score'] for entry in attendance_data][-10:]
    durations = [entry['duration'] for entry in attendance_data][-10:]
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Score Trend Over Time",
            "Session Duration",
            "Performance Distribution",
            "Performance Metrics"
        ),
        specs=[
            [{'type': 'scatter'}, {'type': 'bar'}],
            [{'type': 'histogram'}, {'type': 'indicator'}]
        ]
    )
    
    # Score trend
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=scores,
            mode='lines+markers',
            name='Scores',
            line=dict(color='#6366f1', width=2),
            marker=dict(size=8)
        ),
        row=1, col=1
    )
    
    # Duration bars
    fig.add_trace(
        go.Bar(
            x=dates,
            y=durations,
            name='Duration (min)',
            marker_color='#10b981',
            text=durations,
            textposition='auto',
        ),
        row=1, col=2
    )
    
    # Score distribution
    fig.add_trace(
        go.Histogram(
            x=scores,
            nbinsx=10,
            name='Score Distribution',
            marker_color='#8b5cf6'
        ),
        row=2, col=1
    )
    
    # Performance metrics gauge
    overall_score = performance_metrics.get('overallScore', 0)
    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=overall_score,
            title={'text': "Overall Score"},
            domain={'row': 1, 'column': 1},
            gauge={
                'axis': {'range': [0, 10]},
                'bar': {'color': "#6366f1"},
                'steps': [
                    {'range': [0, 5], 'color': "lightgray"},
                    {'range': [5, 7], 'color': "lightyellow"},
                    {'range': [7, 10], 'color': "lightgreen"}
                ]
            }
        ),
        row=2, col=2
    )
    
    fig.update_layout(
        height=600,
        showlegend=False,
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(t=50, b=20, l=20, r=20)
    )
    
    return fig

# --- Pages ---
def landing_page():
    """Main landing page"""
    st.set_page_config(
        page_title="InternTrack | Intelligent Intern Development Platform",
        page_icon="",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # Custom CSS
    st.markdown("""
        <style>
        .main-title {
            font-size: 4.5rem;
            font-weight: 900;
            font-style: italic;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0;
            line-height: 1;
        }
        .subtitle {
            font-size: 0.9rem;
            font-weight: 600;
            color: #64748b;
            letter-spacing: 0.2em;
            text-transform: uppercase;
            margin-top: 0.5rem;
            margin-bottom: 2rem;
        }
        .auth-box {
            background: white;
            border-radius: 2rem;
            padding: 3rem;
            box-shadow: 0 25px 50px -12px rgba(0,0,0,0.15);
            border: 1px solid #f1f5f9;
            position: relative;
            overflow: hidden;
        }
        .auth-box::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 4px;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        }
        .role-btn {
            height: 120px;
            border-radius: 1rem;
            border: 2px solid #f1f5f9;
            transition: all 0.3s ease;
            cursor: pointer;
        }
        .role-btn:hover {
            border-color: #6366f1;
            transform: translateY(-2px);
            box-shadow: 0 10px 25px -5px rgba(99, 102, 241, 0.3);
        }
        .back-btn {
            background: transparent;
            border: 1px solid #e2e8f0;
            border-radius: 0.75rem;
            padding: 0.5rem 1.5rem;
            font-size: 0.875rem;
            transition: all 0.2s;
        }
        .back-btn:hover {
            background: #f8fafc;
            border-color: #cbd5e1;
        }
        </style>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown('<div class="auth-box">', unsafe_allow_html=True)
        
        # Header
        st.markdown('<h1 class="main-title">InternTrack</h1>', unsafe_allow_html=True)
        st.markdown('<p class="subtitle">Intelligent Intern Development Platform</p>', unsafe_allow_html=True)
        
        # Role selection
        if 'auth_view' not in st.session_state:
            st.session_state.auth_view = 'root'
        
        if st.session_state.auth_view == 'root':
            col_a, col_b = st.columns(2)
            
            with col_a:
                if st.button("Intern Portal\n\nAccess your learning dashboard and track progress", 
                           use_container_width=True,
                           help="Intern login/registration"):
                    st.session_state.auth_view = 'intern'
                    st.rerun()
            
            with col_b:
                if st.button("Admin Hub\n\nManage tracks and monitor cohort performance", 
                           use_container_width=True,
                           help="Company admin login"):
                    st.session_state.auth_view = 'company'
                    st.rerun()
        else:
            # Back button
            if st.button("Back to Selection", type="secondary", use_container_width=True):
                st.session_state.auth_view = 'root'
                st.rerun()
            
            st.markdown('<br>', unsafe_allow_html=True)
            
            # Auth forms
            if st.session_state.auth_view == 'company':
                company_login_page()
            else:  # intern
                intern_auth_page()
        
        st.markdown('</div>', unsafe_allow_html=True)

def company_login_page():
    """Company admin login"""
    st.subheader("Admin Authentication")
    st.markdown("Enter your administrative credentials to access the oversight hub.")
    
    with st.form("company_login", clear_on_submit=False):
        col1, col2 = st.columns(2)
        with col1:
            admin_id = st.text_input("Admin ID", placeholder="Enter admin ID", key="admin_id")
        with col2:
            password = st.text_input("Passcode", type="password", placeholder="Enter passcode", key="admin_pass")
        
        if st.form_submit_button("Authenticate & Enter Hub", type="primary", use_container_width=True):
            if admin_id == "pgt" and password == "123":
                st.session_state.role = "COMPANY"
                st.session_state.auth_view = 'root'
                st.success("Authentication successful! Redirecting...")
                time.sleep(1)
                st.rerun()
            else:
                st.error("Authentication failed. Please check your credentials.")

def intern_auth_page():
    """Intern login/registration"""
    if 'is_reg' not in st.session_state:
        st.session_state.is_reg = False
    
    if st.session_state.is_reg:
        st.subheader("Create New Profile")
        st.markdown("Register to start your learning journey with personalized skill tracking.")
    else:
        st.subheader("Intern Authorization")
        st.markdown("Sign in to access your personalized learning dashboard.")
    
    with st.form("intern_auth", clear_on_submit=False):
        if st.session_state.is_reg:
            name = st.text_input("Full Name", placeholder="John Doe", key="reg_name")
        
        email = st.text_input("Email Address", placeholder="john@example.com", key="auth_email")
        password = st.text_input("Password", type="password", placeholder="Enter your password", key="auth_pass")
        
        if st.session_state.is_reg:
            st.markdown("### Select Learning Track")
            jobs = db.get_jobs()
            if jobs:
                job_titles = {j['id']: f"{j['title']} ({j['domain']})" for j in jobs}
                selected_job = st.selectbox(
                    "Assigned Track", 
                    options=list(job_titles.keys()),
                    format_func=lambda x: job_titles[x],
                    index=0
                )
                st.caption("This track will define your learning objectives and skill requirements.")
            else:
                st.warning("No learning tracks available. Please contact administrator.")
                selected_job = None
        
        submit_text = "Create Profile & Start" if st.session_state.is_reg else "Authorize & Continue"
        
        col1, col2 = st.columns([3, 1])
        with col1:
            if st.form_submit_button(submit_text, type="primary", use_container_width=True):
                if st.session_state.is_reg:
                    if not name or not email or not password:
                        st.error("Please fill in all required fields.")
                    elif not selected_job:
                        st.error("Please select a learning track.")
                    else:
                        if db.register_intern(name, email, password, selected_job):
                            st.success("Profile created successfully! Please login with your credentials.")
                            st.session_state.is_reg = False
                            time.sleep(2)
                            st.rerun()
                        else:
                            st.error("Email already registered. Please use a different email or login.")
                else:
                    if not email or not password:
                        st.error("Please enter both email and password.")
                    else:
                        user = db.login_intern(email, password)
                        if user:
                            st.session_state.role = "INTERN"
                            st.session_state.current_user = user
                            st.session_state.auth_view = 'root'
                            st.success("Login successful! Loading your dashboard...")
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.error("Invalid login credentials. Please try again.")
        
        with col2:
            if st.form_submit_button("Clear", use_container_width=True):
                st.rerun()
    
    # Switch between login and register
    st.markdown("---")
    switch_text = "Already have an account? Sign in" if st.session_state.is_reg else "New to InternTrack? Create profile"
    if st.button(switch_text, use_container_width=True):
        st.session_state.is_reg = not st.session_state.is_reg
        st.rerun()

def intern_portal():
    """Intern main portal"""
    user = st.session_state.current_user
    
    # Page config
    st.set_page_config(
        page_title=f"InternTrack | {user['name']}'s Dashboard",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # Header
    col_header1, col_header2, col_header3 = st.columns([3, 1, 1])
    with col_header1:
        st.markdown(f"""
            <h1 style="margin: 0; color: #1e293b;">Welcome back, <span style="color: #6366f1;">{user['name']}</span></h1>
            <p style="margin: 0; color: #64748b; font-size: 0.9rem;">
                {user['email']} • Learning Dashboard
            </p>
        """, unsafe_allow_html=True)
    with col_header2:
        st.markdown("""
            <div style="text-align: right;">
                <span style="font-size: 0.8rem; color: #64748b; font-weight: 600; letter-spacing: 0.1em;">
                    INTERN ACCESS
                </span>
            </div>
        """, unsafe_allow_html=True)
    with col_header3:
        if st.button("Exit Portal", type="secondary", use_container_width=True):
            del st.session_state.role
            del st.session_state.current_user
            st.rerun()
    
    st.markdown("---")
    
    # Check onboarding
    if not user['onboarded']:
        onboard_intern(user)
        return
    
    # Get user data
    job = db.get_job_by_id(user['assigned_job_id'])
    attendance = db.get_attendance_for_intern(user['id'])
    
    # Update performance metrics if attendance exists
    if attendance and user.get('analysis'):
        performance_metrics = ai_service.get_performance_analysis(attendance, user.get('analysis', {}))
        user['performance_metrics'] = performance_metrics
        db.update_intern(user)
    
    # Main dashboard layout
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        # Profile Overview Card
        st.markdown("### Track Overview")
        
        with st.container():
            st.markdown(f"**Assigned Role:**")
            st.markdown(f"### {job['title']}")
            st.caption(f"{job['domain']} • {len(job['required_skills'])} required skills")
            
            st.markdown("---")
            
            # Skill Match Visualization
            similarity = user['analysis'].get('similarity', 0) if user['analysis'] else 0
            
            st.markdown("#### Skill Match Score")
            fig = create_skill_gap_pie(similarity)
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
            
            st.markdown("---")
            
            # Current Skills
            st.markdown("#### Your Current Skills")
            if user['skills']:
                for skill in user['skills'][:5]:  # Show top 5
                    level = skill.get('level', 0)
                    progress = min(100, (level / 5) * 100)
                    st.markdown(f"**{skill.get('name', 'Skill')}**")
                    st.progress(progress / 100, text=f"Level {level}/5")
            else:
                st.info("No skills assessed yet.")
            
            st.markdown("---")
            
            # Required Benchmarks
            st.markdown("#### Required Benchmarks")
            for req in job['required_skills'][:3]:  # Show top 3
                st.markdown(f"**{req['name']}**")
                st.caption(f"Minimum Level: {req['minLevel']}/5")
    
    with col2:
        # Performance & Recommendations Section
        st.markdown("### Performance Analytics")
        
        # Performance Analysis Chart
        fig = create_performance_analysis_chart(attendance, user.get('performance_metrics', {}))
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        
        st.markdown("---")
        
        # Gap Analysis & Recommendations in tabs
        tab_gaps, tab_rec_videos, tab_rec_docs = st.tabs(["Identified Gaps", "Learning Videos", "Documentation"])
        
        with tab_gaps:
            if user['analysis'] and 'gaps' in user['analysis']:
                for gap in user['analysis']['gaps'][:5]:
                    with st.expander(f"**{gap.get('skill', 'Skill')}** | Priority: {gap.get('priority', 'MEDIUM')}", expanded=True):
                        col_g1, col_g2 = st.columns([1, 2])
                        with col_g1:
                            st.metric("Current Level", gap.get('currentLevel', 0))
                            st.metric("Required Level", gap.get('requiredLevel', 0))
                            st.metric("Gap", gap.get('gapLevel', 0))
                        with col_g2:
                            st.markdown("**Analysis:**")
                            st.write(gap.get('reason', 'No detailed analysis available.'))
                            st.caption(f"Estimated improvement time: {gap.get('estimatedImprovementTime', 'Not specified')}")
            else:
                st.info("No skill gaps identified yet.")
        
        with tab_rec_videos:
            if user['analysis'] and 'recommendations' in user['analysis'] and 'videos' in user['analysis']['recommendations']:
                videos = user['analysis']['recommendations']['videos']
                for video in videos[:5]:
                    with st.container():
                        col_v1, col_v2 = st.columns([3, 1])
                        with col_v1:
                            st.markdown(f"#### [{video.get('title', 'Video')}]({video.get('url', '#')})")
                            st.caption(f"{video.get('description', 'No description')}")
                        with col_v2:
                            st.caption(f"{video.get('duration', 'Unknown')}")
                            st.caption(f"{video.get('level', 'All levels')}")
                        st.divider()
            else:
                st.info("No video recommendations available.")
        
        with tab_rec_docs:
            if user['analysis'] and 'recommendations' in user['analysis'] and 'documentation' in user['analysis']['recommendations']:
                docs = user['analysis']['recommendations']['documentation']
                for doc in docs[:5]:
                    with st.container():
                        col_d1, col_d2 = st.columns([3, 1])
                        with col_d1:
                            st.markdown(f"#### [{doc.get('title', 'Documentation')}]({doc.get('url', '#')})")
                            st.caption(f"{doc.get('description', 'No description')}")
                        with col_d2:
                            st.caption(f"{doc.get('type', 'Resource')}")
                        st.divider()
            else:
                st.info("No documentation recommendations available.")
        
        st.markdown("---")
        
        # Recent Activity
        st.markdown("### Recent Activity")
        if attendance:
            for log in attendance[:3]:
                with st.container():
                    col_a1, col_a2, col_a3 = st.columns([3, 1, 1])
                    with col_a1:
                        st.markdown(f"**{log['task']}**")
                        st.caption(f"{log['date']} • {log['duration']} minutes")
                    with col_a2:
                        score_color = "#10b981" if log['score'] >= 6 else "#ef4444"
                        st.markdown(f"<h3 style='color: {score_color}; margin: 0;'>{log['score']}/10</h3>", unsafe_allow_html=True)
                    with col_a3:
                        status_color = "#10b981" if log['status'] == 'COMPLETED' else "#ef4444"
                        st.markdown(f"<span style='color: {status_color}; font-weight: bold;'>{log['status']}</span>", unsafe_allow_html=True)
                    st.divider()
        else:
            st.info("No learning sessions logged yet. Start your first session!")
    
    with col3:
        # Learning Session Console
        st.markdown("### Learning Session")
        
        # Initialize session state
        if 'clocked_in' not in st.session_state:
            st.session_state.clocked_in = False
            st.session_state.start_time = None
            st.session_state.resources = []
            st.session_state.task = ""
            st.session_state.show_quiz = False
            st.session_state.quiz_data = []
            st.session_state.quiz_answers = {}
            st.session_state.quiz_feedback = ""
        
        # Session Configuration
        with st.container():
            st.markdown("#### Session Configuration")
            
            # Task input
            task = st.text_area(
                "Learning Objective",
                placeholder="What specific concept or skill are you focusing on today?",
                disabled=st.session_state.clocked_in,
                value=st.session_state.task,
                key="task_input",
                height=100
            )
            
            # Resource management
            st.markdown("#### Learning Resources")
            st.caption("Add URLs to resources you'll be studying")
            
            resource_input_container = st.container()
            with resource_input_container:
                resource_col1, resource_col2 = st.columns([4, 1])
                with resource_col1:
                    resource_input = st.text_input(
                        "Resource URL",
                        placeholder="https://...",
                        key="resource_input",
                        label_visibility="collapsed",
                        disabled=st.session_state.clocked_in
                    )
                with resource_col2:
                    if st.button("Add", disabled=st.session_state.clocked_in, use_container_width=True):
                        if resource_input and resource_input not in st.session_state.resources:
                            st.session_state.resources.append(resource_input)
                            st.rerun()
            
            # Show added resources
            if st.session_state.resources:
                st.markdown("**Added Resources:**")
                for i, res in enumerate(st.session_state.resources):
                    col_r1, col_r2 = st.columns([5, 1])
                    with col_r1:
                        st.caption(f"{i+1}. {res[:50]}...")
                    with col_r2:
                        if not st.session_state.clocked_in:
                            if st.button("Delete", key=f"del_res_{i}"):
                                st.session_state.resources.pop(i)
                                st.rerun()
            
            # Timer and controls
            st.markdown("#### Session Timer")
            
            if st.session_state.clocked_in:
                elapsed = int(time.time() - st.session_state.start_time)
                hours = elapsed // 3600
                minutes = (elapsed % 3600) // 60
                seconds = elapsed % 60
                
                col_t1, col_t2, col_t3 = st.columns(3)
                with col_t1:
                    st.metric("Hours", f"{hours:02d}")
                with col_t2:
                    st.metric("Minutes", f"{minutes:02d}")
                with col_t3:
                    st.metric("Seconds", f"{seconds:02d}")
                
                if st.button("End Session & Take Quiz", type="primary", use_container_width=True):
                    if task and st.session_state.resources:
                        with st.spinner("Generating assessment questions..."):
                            quiz_data = ai_service.get_daily_quiz(task, st.session_state.resources)
                            st.session_state.quiz_data = quiz_data
                            st.session_state.quiz_answers = {}
                            st.session_state.show_quiz = True
                            st.session_state.clocked_in = False
                        st.rerun()
                    else:
                        st.error("Please specify a learning objective and add at least one resource.")
            else:
                if st.button("Start Learning Session", type="primary", use_container_width=True):
                    if task and st.session_state.resources:
                        st.session_state.clocked_in = True
                        st.session_state.start_time = time.time()
                        st.session_state.task = task
                        st.success("Session started! Focus on your learning objective.")
                        st.rerun()
                    else:
                        st.error("Please specify a learning objective and add at least one resource.")
        
        st.markdown("---")
        
        # Score Velocity Chart
        st.markdown("### Score Velocity Analysis")
        if attendance:
            fig = create_score_velocity_chart(attendance)
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
            
            # Performance metrics
            if user.get('performance_metrics'):
                metrics = user['performance_metrics']
                col_m1, col_m2, col_m3 = st.columns(3)
                with col_m1:
                    st.metric("Overall Score", f"{metrics.get('overallScore', 0):.1f}/10")
                with col_m2:
                    st.metric("Consistency", f"{metrics.get('consistency', 0):.0f}%")
                with col_m3:
                    improvement = metrics.get('improvementRate', 0)
                    arrow = "↑" if improvement > 0 else "↓" if improvement < 0 else "→"
                    st.metric("Improvement Rate", f"{improvement:+.1f}%", delta=arrow)
        else:
            st.info("Your performance analytics will appear here after your first session.")
    
    # Quiz Modal
    if st.session_state.get('show_quiz', False):
        show_quiz_modal(user)

def show_quiz_modal(user):
    """Show quiz in modal-like overlay"""
    st.markdown("---")
    
    # Quiz Header
    col_qh1, col_qh2 = st.columns([3, 1])
    with col_qh1:
        st.markdown("## Mastery Verification Quiz")
        st.markdown(f"**Learning Context:** {st.session_state.task}")
    with col_qh2:
        if st.button("Cancel Quiz", type="secondary"):
            st.session_state.show_quiz = False
            st.session_state.quiz_data = []
            st.session_state.quiz_answers = {}
            st.rerun()
    
    st.markdown("### Answer all questions to complete your learning session")
    
    quiz_data = st.session_state.quiz_data
    answers = st.session_state.quiz_answers
    
    # Display quiz questions
    for i, question in enumerate(quiz_data):
        st.markdown(f"#### Question {i+1}/{len(quiz_data)}")
        st.markdown(f"**{question['question']}**")
        
        # Display options
        options = question['options']
        selected_index = answers.get(i)
        
        # Create radio buttons for options
        selected_option = st.radio(
            f"Select your answer for Question {i+1}:",
            options,
            key=f"quiz_q_{i}",
            index=selected_index if selected_index is not None else None,
            label_visibility="collapsed"
        )
        
        if selected_option:
            st.session_state.quiz_answers[i] = options.index(selected_option)
    
    # Submit button
    if st.button("Submit Assessment", type="primary", use_container_width=True):
        # Calculate score and analyze results
        score = 0
        correct_answers = []
        incorrect_answers = []
        strengths = []
        weaknesses = []
        
        for i, question in enumerate(quiz_data):
            if i in st.session_state.quiz_answers:
                if st.session_state.quiz_answers[i] == question['correctAnswer']:
                    score += 1
                    correct_answers.append(i)
                else:
                    incorrect_answers.append(i)
        
        # Calculate percentage
        total_questions = len(quiz_data)
        percentage = (score / total_questions) * 100 if total_questions > 0 else 0
        
        # Analyze strengths/weaknesses (simplified)
        if correct_answers:
            strengths = [f"Q{i+1}" for i in correct_answers[:2]]
        if incorrect_answers:
            weaknesses = [f"Q{i+1}" for i in incorrect_answers[:2]]
        
        # Generate feedback
        duration = int(time.time() - st.session_state.start_time) // 60 if st.session_state.start_time else 0
        feedback = ai_service.get_feedback(
            st.session_state.task,
            score,
            duration,
            {
                "total_questions": total_questions,
                "correct_answers": score,
                "strengths": strengths,
                "weaknesses": weaknesses
            }
        )
        
        # Log attendance
        log_entry = {
            "intern_id": user['id'],
            "date": date.today().isoformat(),
            "time_in": datetime.fromtimestamp(st.session_state.start_time).strftime("%H:%M:%S") if st.session_state.start_time else datetime.now().strftime("%H:%M:%S"),
            "time_out": datetime.now().strftime("%H:%M:%S"),
            "task": st.session_state.task,
            "resources": st.session_state.resources,
            "duration": duration,
            "score": score,
            "status": "COMPLETED" if score >= 6 else "NEEDS_REVIEW",
            "quiz_results": {
                "total_questions": total_questions,
                "score": score,
                "percentage": percentage,
                "correct_answers": correct_answers,
                "incorrect_answers": incorrect_answers,
                "strengths": strengths,
                "weaknesses": weaknesses,
                "feedback": feedback
            }
        }
        
        log_id = db.log_attendance(log_entry)
        
        # Update performance metrics
        if attendance := db.get_attendance_for_intern(user['id']):
            performance_metrics = ai_service.get_performance_analysis(attendance, user.get('analysis', {}))
            db.update_performance_metrics(user['id'], performance_metrics)
            # Also update user's performance_metrics in users table
            user['performance_metrics'] = performance_metrics
            db.update_intern(user)
        
        # Display results
        st.success(f"Quiz submitted successfully!")
        
        col_r1, col_r2, col_r3 = st.columns(3)
        with col_r1:
            st.metric("Score", f"{score}/{total_questions}")
        with col_r2:
            st.metric("Percentage", f"{percentage:.1f}%")
        with col_r3:
            st.metric("Status", "PASS" if score >= 6 else "REVIEW")
        
        st.markdown("### Detailed Feedback")
        st.info(feedback)
        
        # Reset state
        time.sleep(3)
        st.session_state.clocked_in = False
        st.session_state.start_time = None
        st.session_state.resources = []
        st.session_state.task = ""
        st.session_state.show_quiz = False
        st.session_state.quiz_data = []
        st.session_state.quiz_answers = {}
        
        st.rerun()

def onboard_intern(user):
    """Intern onboarding process"""
    job = db.get_job_by_id(user['assigned_job_id'])
    
    st.title("Track Onboarding & Skill Assessment")
    st.markdown("Complete your profile setup to receive personalized learning recommendations.")
    
    col_on1, col_on2 = st.columns([2, 1])
    
    with col_on1:
        # Job Information
        with st.container():
            st.markdown(f"### Target Role: {job['title']}")
            st.markdown(f"**Domain:** {job['domain']}")
            st.markdown(job['description'])
            
            st.markdown("#### Required Skills & Levels")
            for skill in job['required_skills']:
                col_s1, col_s2 = st.columns([3, 1])
                with col_s1:
                    st.markdown(f"**{skill['name']}**")
                with col_s2:
                    st.markdown(f"Level {skill['minLevel']}/5 required")
    
    with col_on2:
        # Skill Assessment Card
        with st.container():
            st.markdown("### Skill Self-Assessment")
            st.caption("Rate your proficiency for each required skill (1-5)")
            
            # Dynamic skill assessment
            user_skills = []
            for req_skill in job['required_skills']:
                skill_name = req_skill['name']
                skill_level = st.slider(
                    f"{skill_name}",
                    min_value=1,
                    max_value=5,
                    value=3,
                    key=f"skill_{skill_name}",
                    help=f"Rate your {skill_name} skill level"
                )
                user_skills.append({
                    "name": skill_name,
                    "level": skill_level
                })
    
    st.markdown("---")
    
    # Quick assessment option
    st.markdown("### Quick Assessment Option")
    st.caption("Alternatively, enter your skills in comma-separated format")
    
    quick_input = st.text_area(
        "Enter skills (format: `Skill:Level` or just `Skill` for default level 3)",
        placeholder="React:4, TypeScript:3, JavaScript, CSS:4",
        height=100,
        key="quick_skills_input"
    )
    
    col_btn1, col_btn2 = st.columns([1, 1])
    with col_btn1:
        if st.button("Analyze with Quick Input", use_container_width=True):
            if quick_input:
                # Parse quick input
                parsed_skills = []
                for item in quick_input.split(','):
                    item = item.strip()
                    if ':' in item:
                        parts = item.split(':')
                        name = parts[0].strip()
                        level = parts[1].strip() if len(parts) > 1 else '3'
                        try:
                            parsed_skills.append({
                                "name": name,
                                "level": int(level)
                            })
                        except ValueError:
                            parsed_skills.append({
                                "name": name,
                                "level": 3
                            })
                    else:
                        parsed_skills.append({
                            "name": item,
                            "level": 3
                        })
                user_skills = parsed_skills
    
    with col_btn2:
        if st.button("Complete Onboarding & Generate Analysis", type="primary", use_container_width=True):
            if not user_skills:
                st.error("Please assess your skills or use the quick input option.")
            else:
                with st.spinner("Analyzing skill gaps and generating personalized recommendations..."):
                    # Get AI analysis
                    analysis = ai_service.get_analysis(job, user_skills)
                    
                    # Update user
                    user['skills'] = user_skills
                    user['onboarded'] = True
                    user['analysis'] = analysis
                    user['performance_metrics'] = {}
                    db.update_intern(user)
                    
                    # Update session state
                    st.session_state.current_user = user
                    
                    st.success("Onboarding complete! Personalized analysis generated.")
                    time.sleep(2)
                    st.rerun()

def company_portal():
    """Company admin portal"""
    st.set_page_config(
        page_title="InternTrack | Admin Hub",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Header
    col_h1, col_h2 = st.columns([3, 1])
    with col_h1:
        st.title("Admin Command Hub")
        st.markdown("### Institutional Performance Monitoring & Track Management")
    with col_h2:
        if st.button("Exit Admin Hub", type="secondary", use_container_width=True):
            del st.session_state.role
            st.rerun()
    
    st.markdown("---")
    
    # Tabs for different admin functions
    tab_tracks, tab_cohort, tab_analytics = st.tabs(["Track Repository", "Cohort Management", "Advanced Analytics"])
    
    with tab_tracks:
        manage_tracks()
    
    with tab_cohort:
        manage_cohort()
    
    with tab_analytics:
        show_advanced_analytics()

def manage_tracks():
    """Manage job tracks"""
    st.header("Learning Track Repository")
    st.markdown("Create, edit, and manage learning tracks for your interns.")
    
    # Add new track button
    col_add1, col_add2 = st.columns([1, 5])
    with col_add1:
        if st.button("Deploy New Track", type="primary", use_container_width=True):
            st.session_state.show_job_modal = True
            st.session_state.editing_job = None
    
    # Display existing tracks
    jobs = db.get_jobs()
    
    if not jobs:
        st.info("No learning tracks available. Create your first track to get started!")
    else:
        # Create columns for card layout
        cols = st.columns(3)
        for idx, job in enumerate(jobs):
            with cols[idx % 3]:
                with st.container():
                    # Card styling
                    st.markdown(f"""
                        <div style="
                            border: 1px solid #e2e8f0;
                            border-radius: 1rem;
                            padding: 1.5rem;
                            margin-bottom: 1rem;
                            background: white;
                            box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);
                        ">
                    """, unsafe_allow_html=True)
                    
                    # Domain badge
                    st.markdown(f"<span style='background: #f1f5f9; padding: 0.25rem 0.75rem; border-radius: 9999px; font-size: 0.75rem; color: #64748b;'>{job['domain']}</span>", unsafe_allow_html=True)
                    
                    # Title
                    st.markdown(f"### {job['title']}")
                    
                    # Description preview
                    st.markdown(f"{job['description'][:80]}..." if len(job['description']) > 80 else job['description'])
                    
                    # Skills
                    st.markdown("**Required Skills:**")
                    skills_text = " ".join([f"`{skill['name']} L{skill['minLevel']}`" for skill in job['required_skills'][:3]])
                    st.markdown(skills_text)
                    if len(job['required_skills']) > 3:
                        st.caption(f"+{len(job['required_skills']) - 3} more skills")
                    
                    # Actions
                    col_act1, col_act2 = st.columns(2)
                    with col_act1:
                        if st.button("Edit", key=f"edit_{job['id']}", use_container_width=True):
                            st.session_state.show_job_modal = True
                            st.session_state.editing_job = job
                            st.rerun()
                    with col_act2:
                        if st.button("Delete", key=f"del_{job['id']}", type="secondary", use_container_width=True):
                            st.session_state.delete_job_id = job['id']
                    
                    st.markdown("</div>", unsafe_allow_html=True)
    
    # Handle delete confirmation
    if 'delete_job_id' in st.session_state and st.session_state.delete_job_id:
        st.warning("Are you sure you want to delete this learning track? This action cannot be undone.")
        col_conf1, col_conf2, col_conf3 = st.columns([1, 1, 4])
        with col_conf1:
            if st.button("Yes, Delete", type="primary", use_container_width=True):
                db.delete_job(st.session_state.delete_job_id)
                del st.session_state.delete_job_id
                st.success("Track deleted successfully!")
                time.sleep(1)
                st.rerun()
        with col_conf2:
            if st.button("Cancel", use_container_width=True):
                del st.session_state.delete_job_id
                st.rerun()
    
    # Job modal
    if st.session_state.get('show_job_modal', False):
        show_job_modal()

def show_job_modal():
    """Show job creation/editing modal"""
    editing = st.session_state.get('editing_job')
    
    # Modal header
    if editing:
        st.markdown("### Edit Learning Track")
    else:
        st.markdown("### Deploy New Learning Track")
    
    with st.form("job_form", clear_on_submit=False):
        # Basic information
        col_info1, col_info2 = st.columns(2)
        with col_info1:
            title = st.text_input(
                "Track Name",
                value=editing['title'] if editing else "",
                placeholder="e.g., Frontend Developer"
            )
        with col_info2:
            domain_options = ["Web Development", "Mobile Development", "Machine Learning", "Cloud & DevOps", "Data Science", "Cybersecurity", "UX/UI Design"]
            domain = st.selectbox(
                "Domain",
                domain_options,
                index=domain_options.index(editing['domain']) if editing and editing['domain'] in domain_options else 0
            )
        
        # Skills configuration
        st.markdown("#### Skill Benchmarks")
        skills_input = st.text_area(
            "Required Skills (format: Skill:Level)",
            placeholder="React:4, TypeScript:3, JavaScript:4, CSS:3, HTML:3\nPython:5, MachineLearning:4, TensorFlow:3, Statistics:4",
            value=", ".join([f"{s['name']}:{s['minLevel']}" for s in editing['required_skills']]) if editing else "",
            height=100,
            help="Enter skills with minimum proficiency levels (1-5). One per line or comma separated."
        )
        
        # Description
        description = st.text_area(
            "Track Description",
            value=editing['description'] if editing else "",
            placeholder="Describe the learning objectives, responsibilities, and outcomes for this track...",
            height=150
        )
        
        # Form actions
        col_form1, col_form2 = st.columns(2)
        with col_form1:
            submit_label = "Update Track" if editing else "Deploy Track"
            if st.form_submit_button(submit_label, type="primary", use_container_width=True):
                if not title:
                    st.error("Track name is required")
                else:
                    # Parse skills
                    required_skills = []
                    if skills_input:
                        # Handle both comma and newline separated
                        items = skills_input.replace('\n', ',').split(',')
                        for item in items:
                            item = item.strip()
                            if item:
                                if ':' in item:
                                    parts = item.split(':')
                                    name = parts[0].strip()
                                    level = parts[1].strip() if len(parts) > 1 else '3'
                                    try:
                                        required_skills.append({
                                            "name": name,
                                            "minLevel": int(level)
                                        })
                                    except ValueError:
                                        required_skills.append({
                                            "name": name,
                                            "minLevel": 3
                                        })
                                else:
                                    required_skills.append({
                                        "name": item,
                                        "minLevel": 3
                                    })
                    
                    # Create job object
                    job = {
                        "id": editing['id'] if editing else f"job-{uuid.uuid4().hex[:8]}",
                        "title": title,
                        "domain": domain,
                        "description": description,
                        "required_skills": required_skills
                    }
                    
                    db.upsert_job(job)
                    st.session_state.show_job_modal = False
                    if 'editing_job' in st.session_state:
                        del st.session_state.editing_job
                    st.success("Track saved successfully!")
                    time.sleep(1)
                    st.rerun()
        
        with col_form2:
            if st.form_submit_button("Cancel", use_container_width=True):
                st.session_state.show_job_modal = False
                if 'editing_job' in st.session_state:
                    del st.session_state.editing_job
                st.rerun()

def manage_cohort():
    """Manage intern cohort"""
    st.header("Intern Cohort Management")
    st.markdown("Monitor and manage all registered interns.")
    
    interns = db.get_all_interns()
    
    if not interns:
        st.info("No interns registered yet. Interns will appear here once they register.")
        return
    
    # Cohort overview stats
    col_stats1, col_stats2, col_stats3, col_stats4 = st.columns(4)
    with col_stats1:
        st.metric("Total Interns", len(interns))
    with col_stats2:
        onboarded = sum(1 for i in interns if i['onboarded'])
        st.metric("Onboarded", onboarded)
    with col_stats3:
        active = sum(1 for i in interns if db.get_attendance_for_intern(i['id']))
        st.metric("Active", active)
    with col_stats4:
        avg_score = 0
        count = 0
        for intern in interns:
            attendance = db.get_attendance_for_intern(intern['id'])
            if attendance:
                avg_score += sum(a['score'] for a in attendance if a['score']) / len(attendance)
                count += 1
        st.metric("Avg Score", f"{avg_score/count:.1f}" if count > 0 else "N/A")
    
    st.markdown("---")
    
    # Intern selection and details
    col_select, col_details = st.columns([1, 3])
    
    with col_select:
        st.markdown("### Select Intern")
        
        # Create a searchable selectbox
        intern_options = [f"{i['name']} ({i['email']})" for i in interns]
        selected_intern_idx = st.selectbox(
            "Choose intern to view details:",
            range(len(intern_options)),
            format_func=lambda x: intern_options[x],
            label_visibility="collapsed"
        )
        
        selected_intern = interns[selected_intern_idx]
        
        # Quick stats for selected intern
        st.markdown("#### Quick Stats")
        attendance = db.get_attendance_for_intern(selected_intern['id'])
        
        col_q1, col_q2 = st.columns(2)
        with col_q1:
            st.metric("Sessions", len(attendance))
        with col_q2:
            avg_score = sum(a['score'] for a in attendance) / len(attendance) if attendance else 0
            st.metric("Avg Score", f"{avg_score:.1f}")
        
        if selected_intern.get('analysis'):
            similarity = selected_intern['analysis'].get('similarity', 0)
            st.metric("Skill Match", f"{similarity}%")
    
    with col_details:
        if selected_intern:
            show_intern_details(selected_intern)
        else:
            st.info("Select an intern from the list to view detailed performance metrics.")

def show_intern_details(intern):
    """Show detailed intern performance"""
    logs = db.get_attendance_for_intern(intern['id'])
    job = db.get_job_by_id(intern['assigned_job_id']) if intern['assigned_job_id'] else None
    
    st.header(f"{intern['name']} - Performance Profile")
    st.caption(f"Email: {intern['email']} • Track: {job['title'] if job else 'Not assigned'}")
    
    # Performance metrics
    metrics = intern.get('performance_metrics', {})
    
    col_metrics1, col_metrics2, col_metrics3, col_metrics4 = st.columns(4)
    with col_metrics1:
        st.metric("Total Sessions", len(logs))
    with col_metrics2:
        total_mins = sum(l['duration'] for l in logs)
        st.metric("Total Minutes", total_mins)
    with col_metrics3:
        avg_score = sum(l['score'] for l in logs) / len(logs) if logs else 0
        st.metric("Avg Score", f"{avg_score:.1f}")
    with col_metrics4:
        status = "On-Track" if intern['onboarded'] else "Pending Onboarding"
        st.metric("Status", status)
    
    st.markdown("---")
    
    # Charts section
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        st.markdown("#### Performance Trends")
        if logs:
            # Prepare data for trend chart
            recent_logs = logs[-10:]  # Last 10 sessions
            dates = [log['date'] for log in recent_logs]
            scores = [log['score'] for log in recent_logs]
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=dates, y=scores,
                mode='lines+markers',
                name='Scores',
                line=dict(color='#6366f1', width=3),
                marker=dict(size=8)
            ))
            
            fig.update_layout(
                height=300,
                plot_bgcolor='white',
                paper_bgcolor='white',
                margin=dict(t=30, b=20, l=20, r=20),
                showlegend=False
            )
            fig.update_xaxes(title_text="Date")
            fig.update_yaxes(title_text="Score", range=[0, 10])
            
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        else:
            st.info("No performance data available yet.")
    
    with col_chart2:
        st.markdown("#### Study Effort Distribution")
        if logs:
            # Prepare data for effort chart
            recent_logs = logs[-10:]
            dates = [log['date'] for log in recent_logs]
            durations = [log['duration'] for log in recent_logs]
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=dates, y=durations,
                name='Duration (min)',
                marker_color='#10b981',
                text=durations,
                textposition='auto'
            ))
            
            fig.update_layout(
                height=300,
                plot_bgcolor='white',
                paper_bgcolor='white',
                margin=dict(t=30, b=20, l=20, r=20),
                showlegend=False
            )
            fig.update_xaxes(title_text="Date")
            fig.update_yaxes(title_text="Duration (minutes)")
            
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        else:
            st.info("No study effort data available yet.")
    
    # Skill analysis
    st.markdown("#### Skill Analysis")
    if intern.get('analysis'):
        similarity = intern['analysis'].get('similarity', 0)
        
        col_skill1, col_skill2 = st.columns(2)
        with col_skill1:
            # Skill match pie chart
            fig = create_skill_gap_pie(similarity)
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        
        with col_skill2:
            # Performance pie chart
            st.markdown("#### Performance Distribution")
            fig = create_performance_pie_chart(logs)
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    else:
        st.info("Skill analysis not available. Intern needs to complete onboarding.")
    
    # Recent activity
    st.markdown("#### Recent Learning Sessions")
    if logs:
        for log in logs[:5]:
            with st.container():
                col_log1, col_log2, col_log3 = st.columns([3, 1, 1])
                with col_log1:
                    st.markdown(f"**{log['task']}**")
                    st.caption(f"{log['date']} • {log['duration']} minutes")
                with col_log2:
                    score_color = "#10b981" if log['score'] >= 6 else "#ef4444"
                    st.markdown(f"<h3 style='color: {score_color}; margin: 0;'>{log['score']}/10</h3>", unsafe_allow_html=True)
                with col_log3:
                    status_color = "#10b981" if log['status'] == 'COMPLETED' else "#ef4444"
                    st.markdown(f"<span style='color: {status_color}; font-weight: bold;'>{log['status']}</span>", unsafe_allow_html=True)
                
                # Show resources if available
                if log.get('resources'):
                    with st.expander("View Resources"):
                        for res in log['resources'][:3]:
                            st.caption(f"• {res}")
                
                st.divider()
    else:
        st.info("No learning sessions recorded yet.")

def show_advanced_analytics():
    """Show advanced analytics dashboard"""
    st.header("Advanced Analytics Dashboard")
    st.markdown("Comprehensive analytics across all interns and tracks.")
    
    # Get all data
    interns = db.get_all_interns()
    jobs = db.get_jobs()
    all_attendance = db.get_all_attendance()
    
    if not interns:
        st.info("No data available. Add interns and tracks to see analytics.")
        return
    
    # Overall metrics
    col_ov1, col_ov2, col_ov3, col_ov4 = st.columns(4)
    with col_ov1:
        st.metric("Total Interns", len(interns))
    with col_ov2:
        st.metric("Total Tracks", len(jobs))
    with col_ov3:
        total_sessions = len(all_attendance)
        st.metric("Total Sessions", total_sessions)
    with col_ov4:
        avg_duration = sum(a['duration'] for a in all_attendance) / len(all_attendance) if all_attendance else 0
        st.metric("Avg Session", f"{avg_duration:.0f} min")
    
    st.markdown("---")
    
    # Track popularity
    st.markdown("#### Track Popularity & Performance")
    if jobs:
        track_data = []
        for job in jobs:
            interns_in_track = [i for i in interns if i['assigned_job_id'] == job['id']]
            if interns_in_track:
                avg_similarity = 0
                count = 0
                for intern in interns_in_track:
                    if intern.get('analysis') and 'similarity' in intern['analysis']:
                        avg_similarity += intern['analysis']['similarity']
                        count += 1
                
                track_data.append({
                    "Track": job['title'],
                    "Interns": len(interns_in_track),
                    "Avg Match": avg_similarity / count if count > 0 else 0
                })
        
        if track_data:
            df_tracks = pd.DataFrame(track_data)
            fig = px.bar(df_tracks, x='Track', y=['Interns', 'Avg Match'], 
                        barmode='group', title="Track Distribution & Performance")
            st.plotly_chart(fig, use_container_width=True)
    
    # Performance trends over time
    st.markdown("#### Performance Trends Over Time")
    if all_attendance:
        # Aggregate by date
        from collections import defaultdict
        daily_scores = defaultdict(list)
        
        for attendance in all_attendance:
            daily_scores[attendance['date']].append(attendance['score'])
        
        dates = sorted(daily_scores.keys())[-30:]  # Last 30 days
        avg_scores = [sum(daily_scores[d]) / len(daily_scores[d]) for d in dates]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dates, y=avg_scores,
            mode='lines+markers',
            name='Daily Avg Score',
            line=dict(color='#6366f1', width=2),
            marker=dict(size=6)
        ))
        
        fig.update_layout(
            height=400,
            plot_bgcolor='white',
            paper_bgcolor='white',
            margin=dict(t=30, b=20, l=20, r=20),
            showlegend=False
        )
        fig.update_xaxes(title_text="Date")
        fig.update_yaxes(title_text="Average Score", range=[0, 10])
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Intern leaderboard
    st.markdown("#### Top Performing Interns")
    if interns and all_attendance:
        intern_scores = []
        for intern in interns:
            attendance = db.get_attendance_for_intern(intern['id'])
            if attendance:
                avg_score = sum(a['score'] for a in attendance) / len(attendance)
                total_duration = sum(a['duration'] for a in attendance)
                intern_scores.append({
                    "Name": intern['name'],
                    "Track": db.get_job_by_id(intern['assigned_job_id'])['title'] if intern['assigned_job_id'] else "N/A",
                    "Avg Score": round(avg_score, 1),
                    "Sessions": len(attendance),
                    "Total Hours": round(total_duration / 60, 1)
                })
        
        if intern_scores:
            df_leaderboard = pd.DataFrame(intern_scores)
            df_leaderboard = df_leaderboard.sort_values('Avg Score', ascending=False).head(10)
            st.dataframe(df_leaderboard, use_container_width=True, hide_index=True)

# --- Main App Router ---
def main():
    # Initialize session state
    if 'role' not in st.session_state:
        st.session_state.role = None
    if 'auth_view' not in st.session_state:
        st.session_state.auth_view = 'root'
    if 'is_reg' not in st.session_state:
        st.session_state.is_reg = False
    
    # Add custom CSS for better styling
    st.markdown("""
        <style>
        /* Main container padding */
        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        
        /* Better button styling */
        .stButton button {
            border-radius: 0.75rem;
            font-weight: 600;
            transition: all 0.2s;
        }
        
        .stButton button:hover {
            transform: translateY(-1px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }
        
        /* Metric cards */
        [data-testid="stMetric"] {
            background-color: #f8fafc;
            border-radius: 0.75rem;
            padding: 1rem;
            border: 1px solid #f1f5f9;
        }
        
        /* Better tab styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 1rem;
        }
        
        .stTabs [data-baseweb="tab"] {
            border-radius: 0.75rem 0.75rem 0 0;
            padding: 1rem 2rem;
            font-weight: 600;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Route based on role
    if not st.session_state.role:
        landing_page()
    elif st.session_state.role == "INTERN":
        intern_portal()
    else:  # COMPANY
        company_portal()

if __name__ == "__main__":
    main()
