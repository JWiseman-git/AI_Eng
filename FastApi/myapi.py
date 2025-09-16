from fastapi import FastAPI, HTTPException, Path, Depends, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from typing import Dict
import json
import hashlib

# ------------------------------
# Mock DB setup (JSON file)
# ------------------------------
DB_FILE = "students.json"

def load_students() -> Dict[int, dict]:
    try:
        with open(DB_FILE, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}  # start empty if file doesn't exist

def save_students(students: Dict[int, dict]):
    with open(DB_FILE, "w") as f:
        json.dump(students, f, indent=4)

# ------------------------------
# FastAPI setup
# ------------------------------
app = FastAPI()

# ------------------------------
# OAuth2 setup
# ------------------------------
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# In-memory "users" for demo
users_db = {
    "admin": {
        "username": "admin",
        "hashed_password": hashlib.sha256("password123".encode()).hexdigest(),
    }
}

def verify_password(plain_password, hashed_password):
    return hashlib.sha256(plain_password.encode()).hexdigest() == hashed_password

def authenticate_user(username: str, password: str):
    user = users_db.get(username)
    if not user or not verify_password(password, user["hashed_password"]):
        return False
    return user

async def get_current_user(token: str = Depends(oauth2_scheme)):
    # For simplicity, token is the username
    user = users_db.get(token)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user

# ------------------------------
# Pydantic models
# ------------------------------
class Student(BaseModel):
    name: str
    age: str
    class_name: str  # 'class' is a reserved word

# ------------------------------
# Auth endpoint
# ------------------------------
@app.post("/token")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(status_code=400, detail="Incorrect username or password")
    # For simplicity, token = username
    return {"access_token": user["username"], "token_type": "bearer"}

# ------------------------------
# CRUD Endpoints
# ------------------------------
@app.get("/")
def index():
    return {"message": "Welcome to the Student API"}

@app.get("/students", dependencies=[Depends(get_current_user)])
def get_all_students():
    return load_students()

@app.get("/students/{student_id}", dependencies=[Depends(get_current_user)])
def get_student(student_id: int = Path(..., description="ID of the student")):
    students = load_students()
    student = students.get(str(student_id))
    if not student:
        raise HTTPException(status_code=404, detail="Student not found")
    return student

@app.post("/students", dependencies=[Depends(get_current_user)])
def create_student(student: Student):
    students = load_students()
    new_id = max([int(k) for k in students.keys()], default=0) + 1
    students[str(new_id)] = student.dict()
    save_students(students)
    return {"id": new_id, "student": student}

@app.put("/students/{student_id}", dependencies=[Depends(get_current_user)])
def update_student(student_id: int, student: Student):
    students = load_students()
    if str(student_id) not in students:
        raise HTTPException(status_code=404, detail="Student not found")
    students[str(student_id)] = student.dict()
    save_students(students)
    return {"id": student_id, "student": student}

@app.delete("/students/{student_id}", dependencies=[Depends(get_current_user)])
def delete_student(student_id: int):
    students = load_students()
    if str(student_id) not in students:
        raise HTTPException(status_code=404, detail="Student not found")
    removed = students.pop(str(student_id))
    save_students(students)
    return {"deleted": removed}
