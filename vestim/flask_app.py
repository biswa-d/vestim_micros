from flask import Flask
from vestim.gateway.src.job_manager_qt_flask import job_manager_blueprint
from vestim.gateway.src.hyper_param_manager_qt_flask import hyper_param_manager_blueprint
from vestim.gateway.src.training_setup_manager_qt_flask import training_setup_blueprint
from vestim.gateway.src.training_task_manager_qt_flask import training_task_blueprint

# Initialize Flask app
app = Flask(__name__)

# Register all the blueprints in one place
app.register_blueprint(job_manager_blueprint, url_prefix='/job_manager')
app.register_blueprint(hyper_param_manager_blueprint, url_prefix='/hyper_param_manager')
app.register_blueprint(training_setup_blueprint, url_prefix='/training_setup')
app.register_blueprint(training_task_blueprint, url_prefix='/training_task')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
