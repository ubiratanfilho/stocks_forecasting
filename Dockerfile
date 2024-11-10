# Use the official Python image as the base image
FROM python:3.12.5-slim

# Set the working directory in the container
WORKDIR /

# Copy the requirements.txt file into the container
COPY requirements.txt .

# Install the required Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire app into the container
COPY . .

# Expose the port on which the app will run
EXPOSE 8502

# Command to run the Streamlit app
CMD ["streamlit", "run", "src/app.py", "--server.port=8502"]