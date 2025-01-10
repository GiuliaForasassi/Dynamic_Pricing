"""
EnvironmentService class is the central unique point where you define
and store environment and configuration variable in your service.
That is important because everyone exactly knows where to find
information without creating different version of them.

The variables are divided according to their context and they can be
read from `os.environ` or by other sources like ML cube Secret Server.
"""

import logging
import os

from pydantic import BaseModel

from ml3_service_template.exception.environment import (
    EnvironmentInitializationError,
)


class WebApiInfo(BaseModel):
    """Class containing environment information about the web api
    key backend
    """

    api_key: str
    api_url: str


class OtherImportantInfo(BaseModel):
    """Class containing other environment variables"""

    something: int


class ServerEnvironmentVariables(BaseModel):
    """Set of environment variables to use in the service"""

    web_api_info: WebApiInfo
    other_info: OtherImportantInfo


class EnvironmentService:
    """
    This service is responsible for creating the environment variables.

    To work it needs three environment variables already set:
        - DEPLOY_ENVIRONMENT: deploy_environment name of the application
        - LOCAL: if local mode is active

    EnvironmentService is a singleton that when initialized saves all
    the variables in the class attributes _env
    """

    _logger = logging.getLogger('.'.join([__name__, 'EnvironmentService']))
    _deploy_environment: str
    _local: bool
    _env: ServerEnvironmentVariables
    # This is a list of additional required environment variables
    # that are checked at the start of the object creation
    _required_variables = tuple(
        ['DEPLOY_ENVIRONMENT', 'LOCAL', 'API_KEY', 'WEB_API_URL', 'SOMETHING']
    )
    _initialized = False

    def __new__(cls):  # noqa
        """
        Create the first class instance or returns it if it already
        exists
        """

        if not cls._initialized:
            try:
                cls._check_required_env_variables()
                cls._setup_environment()
                cls._initialized = True
            except Exception as e:
                raise EnvironmentInitializationError() from e
        return cls

    @classmethod
    def _check_required_env_variables(cls) -> None:
        """
        Check if the required env variables are present in order to work
        correctly
        """

        missing_env_variables = []
        for var in cls._required_variables:
            if var not in os.environ:
                missing_env_variables.append(var)
        if len(missing_env_variables) > 0:
            raise ValueError(
                f'Required environment variables are missing: '
                f'{missing_env_variables}'
            )

    @classmethod
    def _setup_environment(cls) -> None:
        """
        Set up the environment by adding all the variables
        """

        cls._deploy_environment = os.environ['DEPLOY_ENVIRONMENT']
        cls._local = os.environ['LOCAL'].lower() == 'true'
        cls._env = ServerEnvironmentVariables(
            web_api_info=cls._create_web_api_info(),
            other_info=cls._create_other_info(),
        )
        for handler in logging.root.handlers:
            handler.setLevel(level='debug' if cls._local else 'info')
        cls._logger.info(
            'Working with:'
            f'\n\tDEPLOY_ENVIRONMENT: {cls._deploy_environment}'
            f'\n\tLOCAL: {cls._local}'
        )

    @classmethod
    def _create_web_api_info(cls) -> WebApiInfo:
        """Add environment info for web application"""

        if cls._local:
            api_url = 'http://localhost:5050/internal'
            api_key = 'test_api_key'
        else:
            # optionally you can use the SecretServer to store secrets
            api_key = os.environ['API_KEY']
            api_url = os.environ['WEB_API_URL']

        return WebApiInfo(api_key=api_key, api_url=api_url)

    @classmethod
    def _create_other_info(cls) -> OtherImportantInfo:
        """Creates cache info retrieving information from secret server"""

        something = 42 if cls._local else int(os.environ['SOMETHING'])
        return OtherImportantInfo(something=something)

    @classmethod
    def local(cls) -> bool:
        """
        Returns the application is in local mode
        """

        return cls._local

    @classmethod
    def get_web_api_info(cls) -> WebApiInfo:
        """Returns web api info"""

        return cls._env.web_api_info

    @classmethod
    def get_other_info(cls) -> OtherImportantInfo:
        """Returns other info"""

        return cls._env.other_info
