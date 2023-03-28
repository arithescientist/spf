# Stock Price Forecasting Flask Web App

### Dockerhub Repository: 
https://hub.docker.com/repository/docker/ikramkhan1/mlops_a2/general
# NOTE
## (Image upload speed to Dockerhub Repository is very slow. It has only uploaded 40 MB in 1 hour, and the total size of image is 3.53 GB)



## JENKINS PIPELINE
### Note: If anyone is interested in using Jenkins pipeline, replace:
registry = "ikramkhan1/mlops_a2" &#8594; &#8594

```
pipeline {
    agent any 
    environment {
        registry = "ikramkhan1/mlops_a2"
        registryCredential = 'dockerhub_id'
        dockerImage = ''
    }
    
    stages {
        stage('Cloning Git') {
            steps {
                checkout scmGit(branches: [[name: '*/main']], extensions: [], userRemoteConfigs: [[url: 'https://github.com/ikram554/spf']])
            }
        }
    
        stage('Building image') {
          steps{
            script {
              dockerImage = docker.build registry
            }
          }
        }
    
        stage('Upload Image') {
         steps{    
             script {
                docker.withRegistry( '', registryCredential ) {
                dockerImage.push()
                }
            }
          }
        }
    }
}

```

 ## Preview
  <img src='screenshots/home.PNG' width='50%'/>
  <img src='screenshots/results.PNG' width='50%'/>
  <img src='screenshots/results2.PNG' width='50%'/>
  <img src='screenshots/results3.PNG' width='50%'/>

  <img src='screenshots/trends.PNG' width='50%'/>
  <img src='screenshots/corr.PNG' width='50%'/>
  
    
  <img src='screenshots/autoarima.PNG' width='50%'/>
    <img src='screenshots/error.PNG' width='50%'/>
  <img src='screenshots/twitter.PNG' width='50%'/>


## Setup
- Install the requirements and setup the development environment.

	`pip3 install -r requirements.txt`
	`make install && make dev`

- Run the application.

		`python3 main.py`

- Navigate to `localhost:5000`.
