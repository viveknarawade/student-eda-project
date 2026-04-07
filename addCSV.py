import pandas as pd
import random

names = ["Amit","Sneha","Priya","Rahul","Neha","Karan","Anjali","Rohit","Pooja","Ravi",
         "Simran","Arjun","Meena","Kunal","Riya","Sahil","Tanvi","Yash","Isha","Dev",
         "Nikita","Aman"]

colleges = ["COEP","MIT","VIT","SPPU","PCCOE"]

data = []

# Generate 200 rows
for i in range(1, 201):
    roll_no = f"R{i:03}"   # R001, R002 ...

    name = random.choice(names)
    age = random.choice([random.randint(18, 24), "", "twenty", "twenty one"])
    gender = random.choice(["Male","Female","female","FEMALE"])
    college = random.choice(colleges + [""])

    math = random.randint(40, 100)
    physics = random.randint(40, 100)
    chemistry = random.choice([random.randint(40, 100), None])
    english = random.choice([random.randint(40, 100), None])

    # Add outliers 🔥
    if random.random() < 0.05:
        math = random.randint(120, 200)
    if random.random() < 0.05:
        physics = random.randint(0, 20)

    data.append([roll_no, name, age, gender, college, math, physics, chemistry, english])


# Convert to DataFrame
df = pd.DataFrame(data, columns=[
    "roll_no","name","age","gender","college",
    "math","physics","chemistry","english"
])

# 🔥 ADD DUPLICATES INTENTIONALLY
duplicates = df.sample(20, random_state=42)   # pick 20 random rows
df = pd.concat([df, duplicates], ignore_index=True)

# Save
df.to_csv("StudentsPerformance_large.csv", index=False)

print("✅ Dataset generated with roll_no + duplicates")