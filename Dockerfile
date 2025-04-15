# Use an official Python runtime as a parent image
FROM python:3.9

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages
RUN pip install --no-cache-dir -r requirements.txt

RUN pip install python-dotenv

# Make port 2024 available to the world outside this container
EXPOSE 2024

# Define environment variable
ENV FLASK_APP=app.py

# Run the application
#CMD ["python", "app.py"]
CMD ["gunicorn", "--workers=4", "--bind=0.0.0.0:2024", "app:app"]





