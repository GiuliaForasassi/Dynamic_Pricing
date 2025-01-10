from logging import getLogger

from ml3_service_template.service.environment import EnvironmentService

if __name__ == '__main__':
    logger = getLogger(__name__)
    logger.info('Hello world!')

    es = EnvironmentService()
    logger.info(es.get_web_api_info())
