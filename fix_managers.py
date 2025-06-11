import re

# Fix testing_manager_qt.py
with open('vestim/backend/src/managers/testing_manager_qt.py', 'r') as f:
    content = f.read()
content = re.sub(r'self\.job_manager', 'self.job_service', content)
with open('vestim/backend/src/managers/testing_manager_qt.py', 'w') as f:
    f.write(content)
print('Fixed testing_manager_qt.py')

# Fix training_setup_manager_qt.py  
with open('vestim/backend/src/managers/training_setup_manager_qt.py', 'r') as f:
    content = f.read()
content = re.sub(r'self\.job_manager', 'self.job_service', content)
with open('vestim/backend/src/managers/training_setup_manager_qt.py', 'w') as f:
    f.write(content)
print('Fixed training_setup_manager_qt.py')
