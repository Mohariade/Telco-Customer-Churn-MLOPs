FROM python:3.9-slim

WORKDIR /app

# 1. Upgrade pip first (Newer pip handles network errors better)
RUN pip install --upgrade pip

# 2. Install heaviest libraries individually to create "Save Points" (Layers)
# If the network fails on scikit-learn, you won't lose the numpy/pandas progress.
RUN pip install --no-cache-dir --default-timeout=1000 numpy
RUN pip install --no-cache-dir --default-timeout=1000 pandas
RUN pip install --no-cache-dir --default-timeout=1000 scikit-learn

# 3. Install the rest from requirements
COPY requirements.txt .
RUN pip install --no-cache-dir --default-timeout=1000 -r requirements.txt

# 4. Copy code and Run
COPY . .
EXPOSE 5000
CMD ["python", "deployments/app.py"]
