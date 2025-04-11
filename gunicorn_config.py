import multiprocessing

# Gunicorn configuration
bind = "0.0.0.0:$PORT"
workers = multiprocessing.cpu_count() * 2 + 1
worker_class = 'uvicorn.workers.UvicornWorker'
timeout = 120
keepalive = 5