from flask import Flask, render_template, request, abort
import pickle
import sqlite3
import numpy as np

app = Flask(__name__)

# load model
try:
    model = pickle.load(open("diabetes_model.pkl", "rb"))
except Exception as e:
    raise RuntimeError("Failed to load diabetes_model.pkl: " + str(e))

# db init (keeps your existing table schema)
def init_db():
    conn = sqlite3.connect("diawell.db")
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS patients (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT,
                    age INTEGER,
                    pregnancies INTEGER,
                    glucose INTEGER,
                    bloodpressure INTEGER,
                    skinthickness INTEGER,
                    insulin INTEGER,
                    bmi REAL,
                    dpf REAL,
                    probability REAL,
                    prediction TEXT)''')
    conn.commit()
    conn.close()

init_db()

# reasonable ranges for validation
RANGES = {
    "pregnancies": (0, 20),
    "age": (1, 120),
    "glucose": (20, 500),
    "bloodpressure": (20, 200),
    "skinthickness": (5, 100),
    "insulin": (0, 900),
    "bmi": (10.0, 70.0),
    "dpf": (0.0, 3.0),
}

# dataset-based default (you can tune these)
DEFAULTS = {
    "pregnancies": 0,     # very low risk
    "skinthickness": 10,  # typical low-fat skin fold
    "dpf": 0.2,           # low family history risk
}


def to_number(val, cast=int, default=None):
    if val == "" or val is None:
        return default
    try:
        return cast(val)
    except:
        return default

def clamp(name, val):
    if val is None:
        return val
    lo, hi = RANGES[name]
    return max(lo, min(hi, val))

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # required fields
    name = request.form.get("name", "Unknown")
    age = to_number(request.form.get("age"), int, None)
    glucose = to_number(request.form.get("glucose"), int, None)
    bloodpressure = to_number(request.form.get("bloodpressure"), int, None)
    insulin = to_number(request.form.get("insulin"), int, None)
    bmi = to_number(request.form.get("bmi"), float, None)

    # optional (allow blank in form)
    pregnancies = to_number(request.form.get("pregnancies"), int, DEFAULTS["pregnancies"])
    skinthickness = to_number(request.form.get("skinthickness"), int, DEFAULTS["skinthickness"])
    dpf = to_number(request.form.get("dpf"), float, DEFAULTS["dpf"])

    # basic validation: required numeric fields must exist
    required = {"age": age, "glucose": glucose, "bloodpressure": bloodpressure, "insulin": insulin, "bmi": bmi}
    missing = [k for k,v in required.items() if v is None]
    if missing:
        return f"Missing or invalid fields: {', '.join(missing)}. Please provide valid numeric values.", 400

    # clamp to reasonable ranges
    pregnancies = clamp("pregnancies", pregnancies or DEFAULTS["pregnancies"])
    age = clamp("age", age)
    glucose = clamp("glucose", glucose)
    bloodpressure = clamp("bloodpressure", bloodpressure)
    skinthickness = clamp("skinthickness", skinthickness or DEFAULTS["skinthickness"])
    insulin = clamp("insulin", insulin)
    bmi = clamp("bmi", bmi)
    dpf = clamp("dpf", dpf or DEFAULTS["dpf"])

    # ensure correct feature order used during training:
    features = np.array([[pregnancies, glucose, bloodpressure, skinthickness, insulin, bmi, dpf, age]], dtype=float)

    # get probability and prediction
    try:
        probs = model.predict_proba(features)[0]   # [prob_no, prob_yes]
        prob_pos = float(probs[1])
        pred = int(model.predict(features)[0])
    except Exception as e:
        return f"Model prediction error: {e}", 500

    result = "Diabetes Risk" if pred == 1 else "No Risk"

    # save to DB with probability
    conn = sqlite3.connect("diawell.db")
    c = conn.cursor()
    c.execute("""INSERT INTO patients (name, age, pregnancies, glucose, bloodpressure, skinthickness,
                 insulin, bmi, dpf, probability, prediction)
                 VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
              (name, age, pregnancies, glucose, bloodpressure, skinthickness, insulin, bmi, dpf, prob_pos, result))
    conn.commit()
    conn.close()

    # Render a result page that includes probability
    return render_template("result.html", name=name, result=result, probability=round(prob_pos, 4),
                           inputs={"pregnancies":pregnancies,"age":age,"glucose":glucose,
                                   "bloodpressure":bloodpressure,"skinthickness":skinthickness,
                                   "insulin":insulin,"bmi":bmi,"dpf":dpf})

@app.route("/records")
def records():
    conn = sqlite3.connect("diawell.db")
    c = conn.cursor()
    c.execute("SELECT * FROM patients ORDER BY id DESC")
    rows = c.fetchall()
    conn.close()
    return render_template("records.html", patients=rows)

# DEBUG route - shows model internals. only for local dev.
@app.route("/model_info")
def model_info():
    try:
        coef = getattr(model, "coef_", None)
        intercept = getattr(model, "intercept_", None)
        classes = getattr(model, "classes_", None)
        return {
            "classes": classes.tolist() if hasattr(classes, "tolist") else str(classes),
            "intercept": intercept.tolist() if hasattr(intercept, "tolist") else str(intercept),
            "coef": coef.tolist() if hasattr(coef, "tolist") else str(coef),
        }
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    app.run(debug=True)
