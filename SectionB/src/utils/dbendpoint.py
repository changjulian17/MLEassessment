import sqlite3



















# custom package for employee
from employee import Employee
# Connect to the SQLite database
# conn = sqlite3.connect('../../data/db/employee.db')
conn = sqlite3.connect(':memory:')

# Create a cursor object
c = conn.cursor()

# Create a table named 'employees'
c.execute("""CREATE TABLE IF NOT EXISTS employees (
                first text,
                last text,
                pay integer
                )""")



def insert_emp(emp):
    with conn:
        c.execute("INSERT INTO employees VALUES (:first, :last, :pay)",
                  {'first': emp.first, 'last': emp.last, 'pay': emp.pay})


def get_emps_by_name(lastname):
    c.execute("SELECT * FROM employees WHERE last=:last",
              {'last': lastname})
    return c.fetchall()


def update_pay(emp, pay):
    with conn:
        c.execute("""UPDATE employees SET pay = :pay
                    WHERE first = :first AND last = :last""",
                  {'first': emp.first, 'last': emp.last, 'pay': pay})


def remove_emp(emp):
    with conn:
        c.execute("DELETE from employees WHERE first = :first AND last = :last",
                  {'first': emp.first, 'last': emp.last})



emp_1 = Employee('a', 'c', 1)
emp_2 = Employee('b', 'd', 2)

insert_emp(emp_1)
insert_emp(emp_2)

c.execute("SELECT * FROM employees")
print(c.fetchall())

# Commit the transaction
conn.commit()

# Close the connection
conn.close()
