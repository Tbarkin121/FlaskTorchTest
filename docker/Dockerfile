# Start with a base image
FROM python:3.9

# Set the working directory
WORKDIR /var/app

# Copy just the requirements and install them
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose port
EXPOSE 5000

# Start the app with gunicorn
CMD ["gunicorn", "app:application", "--bind", "0.0.0.0:5000"]