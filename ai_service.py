import google.generativeai as genai
import json
import os
import re
from typing import List, Dict, Any
from dotenv import load_dotenv

load_dotenv()

class AIService:
    def __init__(self):
        genai.configure(api_key=os.getenv("API_KEY"))
        self.model_pro = genai.GenerativeModel('gemini-1.5-pro-latest')
        self.model_flash = genai.GenerativeModel('gemini-1.5-flash-latest')
    
    def get_analysis(self, job: Dict, user_skills: List[Dict]) -> Dict:
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
            
            if text.startswith('```json'):
                text = text[7:]
            if text.startswith('```'):
                text = text[3:]
            if text.endswith('```'):
                text = text[:-3]
            
            text = text.strip()
            
            try:
                analysis = json.loads(text)
                
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
                json_match = re.search(r'\{.*\}', text, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group())
                else:
                    raise
                    
        except Exception as e:
            print(f"AI Analysis Error: {e}")
            return self.get_fallback_analysis(job, user_skills)
    
    def get_fallback_analysis(self, job: Dict, user_skills: List[Dict]) -> Dict:
        skill_names = [skill.get('name', '').lower() for skill in user_skills]
        required_skills = job['required_skills']
        
        matched = 0
        gaps = []
        
        for req in required_skills:
            req_name = req['name'].lower()
            req_level = req.get('minLevel', 3)
            
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
            "gaps": gaps[:5],
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
                json_match = re.search(r'\[.*\]', text, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group())
                else:
                    raise
                    
        except Exception as e:
            print(f"Quiz Generation Error: {e}")
            return self.get_fallback_quiz(task)
    
    def get_fallback_quiz(self, task: str) -> List[Dict]:
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
        
        task_lower = task.lower()
        if any(keyword in task_lower for keyword in ['react', 'frontend', 'ui']):
            return quiz_templates['react']
        elif any(keyword in task_lower for keyword in ['python', 'ml', 'ai']):
            return quiz_templates['python']
        else:
            return quiz_templates['react']
    
    def get_performance_analysis(self, attendance_data: List[Dict], skill_data: Dict) -> Dict:
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
        
        overall_score = sum(scores) / len(scores) if scores else 0
        
        if len(scores) > 1:
            import statistics
            consistency = max(0, 100 - (statistics.stdev(scores) * 20))
        else:
            consistency = 100
        
        if len(scores) > 3:
            recent_avg = sum(scores[-3:]) / 3
            older_avg = sum(scores[:3]) / 3
            improvement_rate = ((recent_avg - older_avg) / older_avg * 100) if older_avg > 0 else 0
        else:
            improvement_rate = 0
        
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
