import os
os.environ["DATABASE_URL"] = "sqlite:///./test.db"

import pytest
from fastapi.testclient import TestClient
from app.main import Base, SessionLocal, app, engine, seed

client = TestClient(app)


@pytest.fixture(autouse=True)
def fresh_db():
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    with SessionLocal() as db:
        seed(db)


def order(items, email="a@b.com"):
    return client.post("/api/orders", json={"customer_name": "Ishini", "email": email, "items": items})


def stock(pid):
    return next(p["stock"] for p in client.get("/api/products").json() if p["id"] == pid)


def test_list_products():
    assert len(client.get("/api/products").json()) == 6


def test_search_and_filter():
    assert [p["name"] for p in client.get("/api/products?q=sweater").json()] == ["Waffle-Knit Sweater"]
    assert all(p["category"] == "Men Clothing" for p in client.get("/api/products?category=Men Clothing").json())


def test_order_reduces_stock():
    r = order([{"product_id": 1, "qty": 2}])
    assert r.status_code == 201 and r.json()["total"] == 258.0
    assert stock(1) == 4


def test_oversell_blocked_and_rolled_back():
    r = order([{"product_id": 4, "qty": 2}, {"product_id": 1, "qty": 7}])  # 7 > 6 in stock
    assert r.status_code == 409
    assert stock(4) == 12 and stock(1) == 6  # nothing changed


def test_empty_order_rejected():
    assert order([]).status_code == 400


def test_order_history():
    order([{"product_id": 2, "qty": 1}], email="x@y.com")
    assert len(client.get("/api/orders?email=x@y.com").json()) == 1


def test_chat_faq():
    assert "3 to 7" in client.post("/api/chat", json={"message": "how many days does delivery take"}).json()["reply"]


def test_chat_live_stock_and_sold_out():
    assert "sold out" in client.post("/api/chat", json={"message": "waffle knit sweater available?"}).json()["reply"]
    assert "6 left" in client.post("/api/chat", json={"message": "black leather bomber jacket price"}).json()["reply"]


def test_chat_declines_when_unsure():
    assert "not sure" in client.post("/api/chat", json={"message": "quantum physics homework"}).json()["reply"]
