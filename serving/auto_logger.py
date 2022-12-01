import logging
import sys
import traceback


class AutoLogger:
    def __init__(self, app, log_file, level=logging.WARN):
        logging.basicConfig(filename=log_file, level=level)
        self.app = app

    def auto_log(self, log, exception=None, is_print=False):
        if (is_print):
            print(log)
            if exception:
                print(f'Exception: {str(exception)}', file=sys.stderr)
                print(f'Stack trace: {traceback.format_exc()}', file=sys.stderr)

        response_data = {'log': log}

        if exception:
            response_data['exception'] = str(exception)
            response_data['stack_trace'] = traceback.format_exc()
            self.app.logger.error(response_data)
        else:
            self.app.logger.info(response_data)

        return response_data
