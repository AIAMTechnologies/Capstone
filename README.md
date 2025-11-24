# Lead Allocation & Installer ML Local Setup

This repo already contains everything needed to run the FastAPI backend and its machine-learning powered installer allocation locally. Use the steps below to expose the ML status/training endpoints from your local environment without depending on cloud configuration.

## 1. Create a local `.env` file
Copy the sample variables and adjust them to match your local database and secrets:

```
cp .env.example .env
```

Then edit `.env` and fill in the values:

- `DATABASE_URL` – Postgres connection string for your local DB
- `SECRET_KEY` – any long random string
- `ML_PUBLIC_API_KEY` – choose a key for accessing `/api/admin/ml/status` and `/api/admin/ml/train`

The backend now auto-loads this `.env` file on startup, so the variables become available without exporting them in your shell.

## 2. Install backend dependencies
```
cd backend
pip install -r requirements.txt
```

The list now includes `python-dotenv`, which is responsible for loading `.env` when you run `main.py` locally.

## 3. Run the FastAPI server locally
```
uvicorn main:app --reload
```

(Or continue using `python main.py` if you have an entrypoint that wraps FastAPI in uvicorn.)

## 4. Hit the ML endpoints from a browser or REST client
Once the server is up, call:

```
http://localhost:8000/api/admin/ml/status?api_key=<value_from_env>
http://localhost:8000/api/admin/ml/train?api_key=<value_from_env>&force=true
```

Because the API key lives in your local `.env`, you no longer need Azure or any other remote platform to retrieve it.
