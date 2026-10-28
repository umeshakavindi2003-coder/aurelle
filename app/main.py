import os
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sqlalchemy import ForeignKey, create_engine, select, update
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, relationship, sessionmaker

# Switch to PostgreSQL by setting DATABASE_URL=postgresql://user:pass@host/db
DB_URL = os.getenv("DATABASE_URL", "sqlite:///./aurelle.db")
engine = create_engine(DB_URL, connect_args={"check_same_thread": False} if DB_URL.startswith("sqlite") else {})
SessionLocal = sessionmaker(bind=engine)
STATIC = os.path.join(os.path.dirname(__file__), "static")


class Base(DeclarativeBase):
    pass


class Product(Base):
    __tablename__ = "products"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str]
    category: Mapped[str]
    description: Mapped[str]
    price: Mapped[float]
    stock: Mapped[int]
    image: Mapped[str] = mapped_column(default="")


class Order(Base):
    __tablename__ = "orders"
    id: Mapped[int] = mapped_column(primary_key=True)
    customer_name: Mapped[str]
    email: Mapped[str]
    total: Mapped[float] = mapped_column(default=0)
    items: Mapped[list["OrderItem"]] = relationship(cascade="all, delete-orphan")


class OrderItem(Base):
    __tablename__ = "order_items"
    id: Mapped[int] = mapped_column(primary_key=True)
    order_id: Mapped[int] = mapped_column(ForeignKey("orders.id"))
    product_id: Mapped[int] = mapped_column(ForeignKey("products.id"))
    qty: Mapped[int]
    unit_price: Mapped[float]


SEED = [
    ("Black Leather Bomber Jacket", "Women Clothing", "Relaxed-fit faux leather bomber with a zip front and elastic cuffs.", 129.00, 6, "bomber-jacket.jpg"),
    ("Rust Puff-Sleeve Smocked Dress", "Women Clothing", "Smocked bodice, puff sleeves and a tiered mini skirt.", 54.00, 10, "smocked-dress.jpg"),
    ("Waffle-Knit Sweater", "Women Clothing", "Chunky waffle knit in cream, rose and mocha.", 39.00, 0, "waffle-sweater.jpg"),
    ("Linen Wide-Leg Trousers", "Women Clothing", "Breathable linen trousers with an elastic back waist.", 45.00, 12, "linen-trousers.jpg"),
    ("Oversized Denim Shirt", "Men Clothing", "Washed denim shirt with a boxy fit and chest pockets.", 42.00, 8, "denim-shirt.jpg"),
    ("Ribbed Cotton Tee", "Men Clothing", "Soft ribbed cotton t-shirt in a slim fit.", 19.00, 25, "cotton-tee.jpg"),
]


def seed(db: Session):
    if db.scalar(select(Product.id).limit(1)) is None:
        db.add_all(Product(name=n, category=c, description=d, price=p, stock=s, image=i) for n, c, d, p, s, i in SEED)
        db.commit()


@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(engine)
    with SessionLocal() as db:
        seed(db)
    yield


app = FastAPI(title="Aurelle", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC), name="static")


def get_db():
    with SessionLocal() as db:
        yield db


def product_dict(p: Product):
    return {"id": p.id, "name": p.name, "category": p.category, "description": p.description,
            "price": p.price, "stock": p.stock, "image": p.image}


@app.get("/")
def home():
    return FileResponse(os.path.join(STATIC, "index.html"))


@app.get("/api/products")
def list_products(q: str = "", category: str = "", db: Session = Depends(get_db)):
    stmt = select(Product)
    if category:
        stmt = stmt.where(Product.category == category)
    if q:
        stmt = stmt.where(Product.name.ilike(f"%{q}%") | Product.description.ilike(f"%{q}%"))
    return [product_dict(p) for p in db.scalars(stmt.order_by(Product.id))]


class ItemIn(BaseModel):
    product_id: int
    qty: int = Field(gt=0, le=20)


class OrderIn(BaseModel):
    customer_name: str = Field(min_length=1)
    email: str = Field(min_length=3)
    items: list[ItemIn]


@app.post("/api/orders", status_code=201)
def create_order(body: OrderIn, db: Session = Depends(get_db)):
    """Check and reduce stock for every item in ONE transaction, so nothing is oversold."""
    if not body.items:
        raise HTTPException(400, "Your bag is empty")
    try:
        order = Order(customer_name=body.customer_name, email=body.email)
        db.add(order)
        total = 0.0
        for it in body.items:
            p = db.get(Product, it.product_id)
            if not p:
                raise HTTPException(404, f"Product {it.product_id} not found")
            # Atomic: only succeeds if enough stock is still there.
            res = db.execute(update(Product)
                             .where(Product.id == p.id, Product.stock >= it.qty)
                             .values(stock=Product.stock - it.qty))
            if res.rowcount != 1:
                raise HTTPException(409, f"Not enough stock for {p.name}")
            order.items.append(OrderItem(product_id=p.id, qty=it.qty, unit_price=p.price))
            total += p.price * it.qty
        order.total = round(total, 2)
        db.commit()  # all stock changes + the order are saved together
        return {"id": order.id, "total": order.total}
    except Exception:
        db.rollback()  # any failure undoes every stock change
        raise


@app.get("/api/orders")
def my_orders(email: str, db: Session = Depends(get_db)):
    orders = db.scalars(select(Order).where(Order.email == email).order_by(Order.id.desc()))
    return [{"id": o.id, "total": o.total,
             "items": [{"product_id": i.product_id, "qty": i.qty, "unit_price": i.unit_price} for i in o.items]}
            for o in orders]


# ---------- Shoppie: TF-IDF assistant (no paid AI API) ----------
FAQ = [
    ("shipping delivery how many days arrive takes", "Orders usually arrive within 3 to 7 business days, depending on your location."),
    ("payment methods pay visa mastercard paypal cash", "We accept Visa, Mastercard, PayPal, and cash on delivery in selected areas."),
    ("discount offer sale coupon promo deals", "New season offers appear on our homepage. Subscribe to our newsletter to get early access to sales."),
    ("return refund exchange send back", "You can return unworn items within 14 days for a refund or exchange."),
    ("size sizing fit measurements", "Sizes are listed on each product. If you are between sizes, we suggest sizing up."),
]
THRESHOLD = 0.2


class ChatIn(BaseModel):
    message: str


def shoppie_answer(db: Session, question: str) -> str:
    products = db.scalars(select(Product).order_by(Product.id)).all()
    docs = [f for f, _ in FAQ] + [f"{p.name} {p.category} {p.description}" for p in products]
    vec = TfidfVectorizer(stop_words="english").fit(docs)
    sims = cosine_similarity(vec.transform([question]), vec.transform(docs))[0]
    best = int(sims.argmax())
    if sims[best] < THRESHOLD:  # not confident -> don't guess
        return "I'm not sure about that. Could you rephrase, or ask about shipping, payment, sizes or a product?"
    if best < len(FAQ):
        return FAQ[best][1]
    p = products[best - len(FAQ)]  # live data straight from the DB
    if p.stock == 0:
        return f"Sorry, the {p.name} is sold out right now. Please check back soon."
    return f"The {p.name} costs ${p.price:.2f} and we have {p.stock} left in stock."


@app.post("/api/chat")
def chat(body: ChatIn, db: Session = Depends(get_db)):
    return {"reply": shoppie_answer(db, body.message)}
