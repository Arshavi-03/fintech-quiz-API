from fastapi import FastAPI, HTTPException, Depends, Form, Request, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os
import joblib
import random
import time
import json
from dotenv import load_dotenv
from openai import OpenAI
import asyncio

# Load environment variables
load_dotenv()

# Get OpenAI API key from environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("No OpenAI API key found. Please set OPENAI_API_KEY in environment.")

# Create FastAPI app
app = FastAPI(title="Fintech Quiz Generator API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, replace with your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Setup templates
templates = Jinja2Templates(directory="templates")

# Load model data
MODEL_PATH = "models/fintech_quiz_model_data.joblib"
try:
    model_data = joblib.load(MODEL_PATH)
    print(f"✅ Model data loaded successfully from {MODEL_PATH}")
    
    # Extract model components
    QUIZ_TOPICS = model_data['quiz_topics']
    DIFFICULTY_LEVELS = model_data['difficulty_levels']
    DATASET_SUMMARIES = model_data['dataset_summaries']
    
    print(f"Available topics: {list(QUIZ_TOPICS.keys())}")
    print(f"Available difficulties: {list(DIFFICULTY_LEVELS.keys())}")
except Exception as e:
    print(f"Error loading model: {e}")
    raise

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# Pre-generated quizzes cache
QUIZ_CACHE = {}

# Pydantic models
class QuizRequest(BaseModel):
    topic: str
    difficulty: str = "medium"
    num_questions: Optional[int] = None

class QuizSubmission(BaseModel):
    quiz_id: str
    answers: List[int]
    topic: str
    difficulty: str
    questions: List[Dict[str, Any]]
    completion_time: float

class QuizQuestion(BaseModel):
    question: str
    options: List[str]
    correctAnswerIndex: int
    explanation: str
    topic: str
    difficulty: str

class BatchQuizRequest(BaseModel):
    topics: Optional[List[str]] = None
    difficulties: Optional[List[str]] = None
    num_questions: Optional[int] = None

# Quiz generation function
def generate_quiz_questions(dataset_summary, topic, difficulty='medium', num_questions=15):
    """Generate quiz questions based on dataset summary with difficulty level"""
    
    # Get difficulty characteristics
    question_complexity = DIFFICULTY_LEVELS.get(difficulty, DIFFICULTY_LEVELS["medium"])["question_complexity"]
    
    print(f"Generating {num_questions} {difficulty} questions about '{topic}'...")
    
    # Add unique seed to make quizzes different each time
    unique_seed = int(time.time()) % 10000
    
    prompt = f"""
    Based on the following financial dataset information, create {num_questions} multiple-choice quiz questions about {topic}.
    
    Dataset Information:
    {dataset_summary}
    
    Difficulty level: {difficulty} ({question_complexity})
    Use unique seed: {unique_seed} to make this quiz different from others.
    
    For each question:
    1. Create a {difficulty} question related to {topic} based on the dataset
    2. Make questions {question_complexity}
    3. Provide 4 answer options with exactly one correct answer
    4. For {difficulty} difficulty, make the incorrect options plausible and challenging
    5. Mark which option is correct (using the correctAnswerIndex)
    6. Provide a detailed explanation for why that answer is correct and why others are incorrect
    
    Format your response as a JSON array with the following structure for each question:
    {{
        "question": "The question text",
        "options": ["Option A", "Option B", "Option C", "Option D"],
        "correctAnswerIndex": 0, // 0-based index of the correct answer
        "explanation": "Explanation of why this answer is correct",
        "topic": "{topic}",
        "difficulty": "{difficulty}"
    }}
    
    Return ONLY the JSON array with no additional text. Ensure all questions are unique.
    """
    
    try:
        # Add some randomness to temperature based on difficulty
        if difficulty == "easy":
            temp = 0.5
        elif difficulty == "medium":
            temp = 0.7
        else:  # hard
            temp = 0.8
            
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an expert in finance and fintech who creates educational content."},
                {"role": "user", "content": prompt}
            ],
            temperature=temp,
            max_tokens=3000
        )
        
        # Extract the JSON content from the response
        content = response.choices[0].message.content
        
        # Parse the JSON content
        import re
        import json
        
        # Find JSON array in the response
        match = re.search(r'\[.*\]', content, re.DOTALL)
        if match:
            json_content = match.group(0)
            questions = json.loads(json_content)
            print(f"✅ Successfully generated {len(questions)} questions")
            return questions
        else:
            # Try to parse the entire content as JSON
            questions = json.loads(content)
            if isinstance(questions, list):
                print(f"✅ Successfully generated {len(questions)} questions")
                return questions
            else:
                raise ValueError("Could not parse response as a list of questions")
    
    except Exception as e:
        print(f"❌ Error generating quiz questions: {e}")
        # If error, try simplified approach
        try:
            simple_prompt = f"Create {num_questions} multiple-choice {difficulty} questions about {topic}. Each question should have 4 options with one correct answer. Format as JSON array."
            
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": simple_prompt}],
                temperature=0.5,
                max_tokens=2000
            )
            
            content = response.choices[0].message.content
            match = re.search(r'\[.*\]', content, re.DOTALL)
            
            if match:
                json_content = match.group(0)
                questions = json.loads(json_content)
                print(f"✅ Generated {len(questions)} questions with simplified approach")
                return questions
            else:
                return []
        except:
            print("❌ Failed even with simplified approach")
            return []

def generate_quiz_id(topic, difficulty):
    """Generate a unique ID for this quiz"""
    timestamp = int(time.time())
    random_part = random.randint(1000, 9999)
    topic_part = ''.join(word[0] for word in topic.split() if word)
    return f"FQ-{topic_part}-{difficulty[0].upper()}-{timestamp}-{random_part}"

def score_quiz(user_answers, questions, topic, difficulty, completion_time):
    """Score a quiz based on user answers with passing criteria"""
    if len(user_answers) != len(questions):
        raise ValueError(f"Number of answers ({len(user_answers)}) does not match number of questions ({len(questions)})")
    
    results = []
    correct_count = 0
    
    for i, (question, answer) in enumerate(zip(questions, user_answers)):
        is_correct = answer == question['correctAnswerIndex']
        if is_correct:
            correct_count += 1
        
        results.append({
            'question_num': i+1,
            'question': question['question'],
            'user_answer': answer,
            'correct_answer': question['correctAnswerIndex'],
            'is_correct': is_correct,
            'explanation': question['explanation'],
            'topic': question.get('topic', topic),
            'difficulty': question.get('difficulty', difficulty)
        })
    
    score_percentage = (correct_count / len(questions)) * 100
    
    # Get pass criteria for this topic and difficulty
    pass_criteria = QUIZ_TOPICS.get(topic, {}).get("pass_criteria", {}).get(difficulty, 70)
    passed = score_percentage >= pass_criteria
    
    return {
        'total_questions': len(questions),
        'correct_answers': correct_count,
        'score_percentage': score_percentage,
        'detailed_results': results,
        'pass_criteria': pass_criteria,
        'passed': passed,
        'completion_time': completion_time,
        'topic': topic,
        'difficulty': difficulty
    }

async def generate_quiz_for_topic_difficulty(topic, difficulty, num_questions):
    """Generate a quiz for a specific topic and difficulty combination"""
    # Get the corresponding dataset for this topic
    dataset_key = QUIZ_TOPICS[topic]['dataset']
    
    # Get the dataset summary
    dataset_summary = DATASET_SUMMARIES.get(dataset_key, "Dataset information not available.")
    
    # If num_questions not provided, use default range for the difficulty
    if not isinstance(num_questions, int) or num_questions < 1:
        diff_settings = DIFFICULTY_LEVELS.get(difficulty, DIFFICULTY_LEVELS["medium"])
        min_questions = diff_settings["min_questions"]
        max_questions = diff_settings["max_questions"]
        
        # Randomly determine number of questions within range
        num_questions = random.randint(min_questions, max_questions)
    
    # Generate quiz questions
    questions = generate_quiz_questions(dataset_summary, topic, difficulty, num_questions)
    
    if not questions:
        return None
    
    # Generate unique quiz ID
    quiz_id = generate_quiz_id(topic, difficulty)
    
    return {
        "quiz_id": quiz_id,
        "topic": topic,
        "difficulty": difficulty,
        "questions": questions,
        "pass_criteria": QUIZ_TOPICS[topic]['pass_criteria'].get(difficulty, 70),
        "timestamp": int(time.time())
    }

# API Routes
@app.get("/")
async def root():
    return {"message": "Fintech Quiz Generator API is running"}

@app.get("/topics")
async def get_topics():
    """Get all available quiz topics"""
    topics = []
    for topic_name, topic_info in QUIZ_TOPICS.items():
        topics.append({
            "id": topic_name,
            "name": topic_name,
            "description": topic_info.get("description", ""),
            "difficulties": list(topic_info.get("pass_criteria", {}).keys())
        })
    return {"topics": topics}

@app.get("/difficulties")
async def get_difficulties():
    """Get all available difficulty levels"""
    difficulties = []
    for diff_name, diff_info in DIFFICULTY_LEVELS.items():
        difficulties.append({
            "id": diff_name,
            "name": diff_name.capitalize(),
            "description": diff_info.get("description", ""),
            "pass_percentage": diff_info.get("pass_percentage", 70)
        })
    return {"difficulties": difficulties}

@app.post("/generate-quiz")
async def generate_quiz(request: QuizRequest):
    """Generate a new quiz based on topic and difficulty"""
    topic = request.topic
    difficulty = request.difficulty
    num_questions = request.num_questions
    
    # Validate topic and difficulty
    if topic not in QUIZ_TOPICS:
        raise HTTPException(status_code=400, detail=f"Invalid topic: {topic}")
    
    if difficulty not in DIFFICULTY_LEVELS:
        raise HTTPException(status_code=400, detail=f"Invalid difficulty: {difficulty}")
    
    # Check if we have a cached quiz
    cache_key = f"{topic}_{difficulty}"
    if cache_key in QUIZ_CACHE:
        # Check if cache is less than 1 hour old
        cached_quiz = QUIZ_CACHE[cache_key]
        if (int(time.time()) - cached_quiz.get("timestamp", 0)) < 3600:
            return cached_quiz
    
    # Generate a new quiz
    quiz = await generate_quiz_for_topic_difficulty(topic, difficulty, num_questions)
    
    if not quiz:
        raise HTTPException(status_code=500, detail="Failed to generate quiz questions")
    
    # Cache the quiz
    QUIZ_CACHE[cache_key] = quiz
    
    return quiz

@app.post("/submit-quiz")
async def submit_quiz(submission: QuizSubmission):
    """Score a submitted quiz"""
    try:
        # Score the quiz
        results = score_quiz(
            submission.answers,
            submission.questions,
            submission.topic,
            submission.difficulty,
            submission.completion_time
        )
        
        # Return results
        return results
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/generate-all-quizzes")
async def generate_all_quizzes(request: BatchQuizRequest = Body(default=None)):
    """Generate quizzes for all topics and difficulties or specified combinations"""
    # If no specific topics requested, use all available topics
    topics = request.topics if request.topics else list(QUIZ_TOPICS.keys())
    
    # Filter to only valid topics
    topics = [t for t in topics if t in QUIZ_TOPICS]
    
    # If no specific difficulties requested, use all available difficulties
    difficulties = request.difficulties if request.difficulties else list(DIFFICULTY_LEVELS.keys())
    
    # Filter to only valid difficulties
    difficulties = [d for d in difficulties if d in DIFFICULTY_LEVELS]
    
    # Number of questions per quiz
    num_questions = request.num_questions
    
    # Generate quizzes for all combinations
    quizzes = {}
    
    print(f"Generating quizzes for {len(topics)} topics and {len(difficulties)} difficulty levels...")
    
    for topic in topics:
        topic_quizzes = {}
        for difficulty in difficulties:
            print(f"Generating quiz for {topic} ({difficulty})...")
            quiz = await generate_quiz_for_topic_difficulty(topic, difficulty, num_questions)
            if quiz:
                topic_quizzes[difficulty] = quiz
                
                # Also update the cache
                cache_key = f"{topic}_{difficulty}"
                QUIZ_CACHE[cache_key] = quiz
        
        if topic_quizzes:
            quizzes[topic] = topic_quizzes
    
    return {
        "message": f"Generated {len(quizzes)} topic quizzes with {len(difficulties)} difficulty levels each",
        "quizzes": quizzes
    }

@app.get("/get-quiz/{topic}/{difficulty}")
async def get_quiz(topic: str, difficulty: str):
    """Get a quiz for a specific topic and difficulty"""
    # Validate topic and difficulty
    if topic not in QUIZ_TOPICS:
        raise HTTPException(status_code=400, detail=f"Invalid topic: {topic}")
    
    if difficulty not in DIFFICULTY_LEVELS:
        raise HTTPException(status_code=400, detail=f"Invalid difficulty: {difficulty}")
    
    # Check if we have a cached quiz
    cache_key = f"{topic}_{difficulty}"
    if cache_key in QUIZ_CACHE:
        # Return the cached quiz
        return QUIZ_CACHE[cache_key]
    
    # No cached quiz, generate a new one
    quiz = await generate_quiz_for_topic_difficulty(topic, difficulty, None)
    
    if not quiz:
        raise HTTPException(status_code=500, detail="Failed to generate quiz questions")
    
    # Cache the quiz
    QUIZ_CACHE[cache_key] = quiz
    
    return quiz

@app.get("/list-quizzes")
async def list_quizzes():
    """List all available quizzes in the cache"""
    available_quizzes = []
    
    for key, quiz in QUIZ_CACHE.items():
        topic = quiz.get("topic")
        difficulty = quiz.get("difficulty")
        question_count = len(quiz.get("questions", []))
        timestamp = quiz.get("timestamp", 0)
        
        available_quizzes.append({
            "topic": topic,
            "difficulty": difficulty,
            "question_count": question_count,
            "quiz_id": quiz.get("quiz_id"),
            "generated_at": timestamp,
            "age_minutes": int((time.time() - timestamp) / 60)
        })
    
    return {"quizzes": available_quizzes}

# Run the application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)