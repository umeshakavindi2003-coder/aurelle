# Aurelle

Online clothing store: FastAPI + SQLAlchemy + SQLite, vanilla HTML/CSS/JS, and **Shoppie**, a TF-IDF (scikit-learn) assistant that needs no paid AI API.

## Run
```
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```
Open http://127.0.0.1:8000

## Test
```
pytest
```

## Notes
- Orders check and reduce stock in a single transaction (atomic `UPDATE ... WHERE stock >= qty`), so items are never oversold.
- Shoppie answers FAQs by TF-IDF cosine similarity, reads live price/stock from the DB, and declines when similarity is below `THRESHOLD`.
- PostgreSQL: set `DATABASE_URL=postgresql://user:pass@host/db` (and `pip install psycopg2-binary`).
