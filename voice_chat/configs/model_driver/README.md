The driver files contains API settings for models hosted on the API server. 
If the models are running in docker then those settings will be contained in the docker-compose.yaml file.
These files will contain duplicate configuration keys and so you must manually ensure that they are consistent. 
Not doing so can lead to unexpected behaviours. 
@mtman