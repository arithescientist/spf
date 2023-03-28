# Stock Price Forecasting Flask Web App

### Dockerhub Repository: 
https://hub.docker.com/repository/docker/ikramkhan1/mlops_a2/general
# NOTE:
#### (Image upload speed to Dockerhub Repository is considerably slower than expected. It has only uploaded 40 MB in 1 hour, and the total size of image is 3.53 GB. This may cause some delays in the upload process or in worse case I might not be able to not upload the docker image to dockerhub.)



## JENKINS PIPELINE

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
### Note:
If you're considering using the Jenkins pipeline, please keep in mind the following important instructions for customization:
- To add your DockerHub credentials, navigate to Manage Jenkins → Manage Credentials in Jenkins, and update the appropriate field.
- To replace the GitHub URL in the checkout stage, simply update it to reflect your own repository's URL.
- Finally, ensure that you update the "registry" field to match the URL for your own DockerHub repository.
By following these steps, you can easily customize the Jenkins pipeline for your own purposes.


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


## For Manual Setup
- Install the requirements and setup the development environment.

	`pip3 install -r requirements.txt`
	`make install && make dev`

- Run the application.

		`python3 main.py`

- Navigate to `localhost:5000`.
