import requests
import time

time.sleep(1)
try:
    response = requests.get('http://localhost:8001/jobs/job_20250611-172933/status')
    print('Status Code:', response.status_code)
    if response.status_code == 200:
        job_status = response.json()
        print('Job Status Keys:', list(job_status.keys()))
        if 'current_task_id' in job_status:
            print('Current Task ID:', job_status['current_task_id'])
        else:
            print('No current_task_id found')
    else:
        print('Response:', response.text)
except Exception as e:
    print('Error:', e)
