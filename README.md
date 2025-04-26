# Fintech Quiz Generator API

A FastAPI service that generates unique quizzes on fintech topics using OpenAI's GPT models.

## Deployment Instructions

1. Create a Render web service using this repository
2. Set the following environment variables:
   - `OPENAI_API_KEY`: Your OpenAI API key
   - `PORT`: 8000 (or your preferred port)
3. Create a persistent disk and mount it to `/app/models`
4. Upload the model files to the persistent disk

## Available Endpoints

- `POST /generate-all-quizzes`: Generate quizzes for all topics and difficulties
- `GET /get-quiz/{topic}/{difficulty}`: Get a specific quiz
- `POST /submit-quiz`: Submit and score a completed quiz
- `GET /topics`: Get list of available topics
- `GET /difficulties`: Get list of difficulty levels
- `GET /list-quizzes`: List all cached quizzes