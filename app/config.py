import os
from dotenv import load_dotenv

load_dotenv()

SECRET_KEY = os.getenv("SECRET_KEY", "etiylXz37f3lDmyNkLd8vWSTMJZrKxuu")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost/senior-project-2")
# DATABASE_URL = "postgresql://dandevko:dandevko@localhost/senior_project_2"
# DATABASE_URL = "postgresql://postgres:postgres@localhost/senior-project-2"
