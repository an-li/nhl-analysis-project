import logging
import sys
import time
import traceback


class Logger:
    def __init__(self, log_file, name, level=logging.WARN):
        self.logger = logging.getLogger(name)
        logging.basicConfig(filename=log_file, level=level, format='%(asctime)s:%(name)s:%(levelname)s:%(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S')
        logging.Formatter.converter = time.gmtime

    def auto_log(self, log, exception=None, is_print=False):
        if is_print:
            print(log)
            if exception:
                print(f'Exception: {str(exception)}', file=sys.stderr)
                print(f'Stack trace: {traceback.format_exc()}', file=sys.stderr)

        response_data = {'log': log}

        if exception:
            response_data['exception'] = str(exception)
            response_data['stack_trace'] = traceback.format_exc()
            self.logger.error(log)
            self.logger.error(response_data['exception'])
            self.logger.error(response_data['stack_trace'])
        else:
            self.logger.info(log)

        return response_data
