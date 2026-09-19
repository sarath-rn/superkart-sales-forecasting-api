FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy all files from the current directory to the container's working directory
COPY . .

# Install dependencies from the requirements file without using cache to reduce image size
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Define the command to start the application using Gunicorn with 4 worker processes
# - `-w 4`: Uses 4 worker processes for handling requests
# - `-b 0.0.0.0:5000`: Binds the server to port 5000 on all network interfaces
# - `app:app`: Runs the Flask app (referencing the Flask instance `app` in `app.py`)
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app:app"]
