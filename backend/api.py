from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import sqlite3
from datetime import datetime
import os

class EmployeeRequest(BaseModel):
    employeeCode: str

class VacancyResponse(BaseModel):
    name: str
    vacancyBalanceDays: int

class PayrollResponse(BaseModel):
    name: str
    YTDPayroll: float

app = FastAPI()

def init_db():
    if os.path.exists("employee.db"):
        os.remove("employee.db")
    conn = sqlite3.connect("employee.db")
    cursor = conn.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS employee (
            employee_code TEXT PRIMARY KEY,
            name TEXT
        )
        """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS earnings (
            payment_date TEXT,
            employee_code TEXT,
            amount REAL,
            FOREIGN KEY (employee_code) REFERENCES employee (employee_code)
        )
        """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS employee_vacancy (
            employee_code TEXT PRIMARY KEY,
            balance_days INTEGER,
            FOREIGN KEY (employee_code) REFERENCES employee (employee_code)
        )
        """
    )

    cursor.executemany(
        """
        INSERT OR IGNORE INTO employee (employee_code, name)
        VALUES (?, ?)
        """,
        [
            ("abc123", "Allan Ferreira"),
            ("def456", "Yan Ferreira"),
        ],
    )

    cursor.executemany(
        """
        INSERT OR IGNORE INTO employee_vacancy (employee_code, balance_days)
        VALUES (?, ?)
        """,
        [
            ("abc123", 10),
            ("def456", 20),
        ],
    )

    cursor.executemany(
        """
        INSERT OR IGNORE INTO earnings (payment_date, employee_code, amount)
        VALUES (?, ?, ?)
        """,
        [
            ("2025-01-01", "abc123", 50.00),
            ("2025-01-02", "abc123", 50.00),
            ("2025-01-01", "def456", 100.00),
            ("2025-01-02", "def456", 100.00),
        ],
    )

    conn.commit()
    conn.close()

init_db()

@app.post("/employee/vacancy", response_model=VacancyResponse)
async def get_employee_vacancy(request: EmployeeRequest):
    conn = sqlite3.connect("employee.db")
    cursor = conn.cursor()
    cursor.execute(
        "SELECT e.name, ev.balance_days FROM employee e  " \
        " LEFT JOIN employee_vacancy ev ON e.employee_code = ev.employee_code " \
        " WHERE e.employee_code = ?",
        (request.employeeCode,)
    )
    result = cursor.fetchone()
    conn.close()

    if result:
        return VacancyResponse(
            name=result[0],
            vacancyBalanceDays=result[1]
        )
    else:
        raise HTTPException(status_code=404, detail="Employee not found")

@app.post("/employee/payroll", response_model=PayrollResponse)
async def get_employee_payroll(request: EmployeeRequest):
    conn = sqlite3.connect("employee.db")
    cursor = conn.cursor()
    cursor.execute(
        "SELECT e.name, SUM(amount) AS ytd_payroll FROM employee e " \
        " LEFT JOIN earnings er ON e.employee_code = er.employee_code " \
        " WHERE e.employee_code = ? AND er.payment_date < ?",
        (request.employeeCode, datetime.now().strftime("%Y-%m-%d"))
    )
    result = cursor.fetchone()
    conn.close()

    if result:
        return PayrollResponse(
            name=result[0],
            YTDPayroll=result[1]
        )
    else:
        raise HTTPException(status_code=404, detail="Employee not found")
