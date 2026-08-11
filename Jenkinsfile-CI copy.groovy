#!groovy
def util_scrips
properties([
  parameters([
        [
            $class: 'CascadeChoiceParameter', 
            name: 'VERSIONS', 
            choiceType: 'PT_CHECKBOX', 
            description: 'Select at least one app',
            filterable: false, 
            script: [
             $class: 'GroovyScript', 
             script: [
                 sandbox: true, 
                 script: "return ['3090','4090','5090','gcp']"
             ]
         ]
        ]
 ])
])
pipeline {
    agent {
        label ""
    }
    options {
        timeout(time: 120, unit: 'MINUTES')
        buildDiscarder(logRotator(numToKeepStr: '20', artifactNumToKeepStr: '20', daysToKeepStr: '90', artifactDaysToKeepStr: '4'))
    }

    parameters {
        string( name: 'branch', defaultValue: 'main', description: 'repo branch? [ie: main]')
        string( name: 'SCRIPTS_REPO_BRANCH', defaultValue: 'kube/base', description: 'pipeline scripts repo branch ? [ie: kube/base]')
        booleanParam(name: 'HELM_DRY_RUN_DEBUG', defaultValue: false, description: 'If checked, do not deploy with helm, just show what would be deployed/installed')

    }

    environment {
        APP_NAME = "ml-tagger-python"
        GROOVY_SCRIPTS_DIR = "scripts/groovy"
        IMAGE_TAG = sh (returnStdout: true, script: "git log -n 1 --pretty=format:'%h'").trim()
        GCE_ENVIRONMENT = "dev-dc-03"
    }

    stages {
        stage('show-env') {
            steps {
                sh "ls -all"
                sh "env | sort"
                sh "echo ${params.VERSIONS}"
                sh "git clean -fdx"
            }
        }

        stage ('set-jenkins-build-id') {
            steps {
                script {
                    currentBuild.displayName = "$BUILD_NUMBER-$branch-$IMAGE_TAG"
                }
            }
        }

        // stage('Git-checkout') {
        //     steps {
        //         checkout([$class                           : 'GitSCM',
        //                   branches                         : [[name: "${branch}"]],
        //                   doGenerateSubmoduleConfigurations: false,
        //                   extensions                       : [[$class: 'SubmoduleOption',
        //                                                        disableSubmodules: false,
        //                                                        parentCredentials: true,
        //                                                        recursiveSubmodules: true,
        //                                                        reference: '',
        //                                                        trackingSubmodules: false]],                           
        //                   submoduleCfg                     : [],
        //                   userRemoteConfigs                : [[credentialsId: 'git-ssh-ad', url: "${env.GIT_URL}" ]]])
        //     }
        // }
        
        stage('Clone-scripts-repo') {
            steps {
                sh 'mkdir -p cloned_repo'
                sh 'mkdir -p k8s'
                dir('cloned_repo') {
                    git(
                        changelog: false,
                        poll: false,
                        credentialsId: "git-cloud-infrastructure",
                        url: "git@github.com:cloudastructure/cloud-infrastructure.git",
                        branch: "${SCRIPTS_REPO_BRANCH}"
                    )
                }
                sh 'cp -rf ./cloned_repo/scripts .'
                sh 'cp -rf ./cloned_repo/helms/deploy-app-helm k8s/'
                sh 'cp -rf ./cloned_repo/helms/values/*.yaml k8s/'
                sh 'rm -rf cloned_repo'
                stash includes: 'scripts/**/*', name: 'scripts'
                stash includes: 'k8s/**/*', name: 'k8s'
            }
        }

        stage('Init-Envs') {
            steps {
                unstash 'scripts'
                script{
                    util_scrips = load "${GROOVY_SCRIPTS_DIR}/Utils.groovy"
                }
            }
        }

        stage('Slack-notification-job-started') {
            steps {
                script {
                    util_scrips.sendSlackNotification(APP_NAME, "STARTED", "BUILD", IMAGE_TAG )
                }
            }
        }


        stage('Dynamic docker-build-and-push Stages') {
            steps {
                script {
            def VERSIONS = params.VERSIONS.split(',')
            for (int i = 0; i < VERSIONS.length; i++) {
                def version = VERSIONS[i].trim()
                stage("docker-build-and-push-${VERSIONS[i]}") {
                    sh 'chmod +x scripts/*'
                    if (version == "gcp") {
                        sh "echo 'Running special commands for GCP version'"
                        sh "scripts/custom-docker-build-path.sh ${APP_NAME} ${IMAGE_TAG} Dockerfiles Dockerfile.gcp"
                        sh "scripts/custom-docker-push.sh ${APP_NAME} ${GCE_ENVIRONMENT} ${IMAGE_TAG}"

                    } else {
                        sh "scripts/custom-docker-build-path.sh ${APP_NAME}-${VERSIONS[i]} ${IMAGE_TAG} Dockerfiles Dockerfile.${VERSIONS[i]}"
                        sh "scripts/custom-docker-push.sh ${APP_NAME}-${VERSIONS[i]} ${GCE_ENVIRONMENT} ${IMAGE_TAG}"

                    }
                }
            }
                }
            }
        }

        // stage('Dynamic eploy-to-dev-gke Stages') {
        //     steps {
        //         script {
        //     def VERSIONS = params.VERSIONS.split(',')

        //         for (int i = 0; i < VERSIONS.length; i++) {
        //             stage("deploy-to-dev-gke-${VERSIONS[i]}") {
        //                 unstash 'k8s'
        //                 sh "scripts/custom-deploy-helm.sh ${APP_NAME}-${VERSIONS[i]} ${GCE_ENVIRONMENT} ${IMAGE_TAG} "
        //             }
        //         }
        //         }
        //     }
        // }

        
    }

    post {
        always {
            sh 'echo [JENKINS_RESULT] ALWAYS...'
        }
        success {
            sh 'echo [JENKINS_RESULT] SUCCESS'
            script {
                util_scrips.sendSlackNotification(APP_NAME, "SUCCESS", "BUILD", IMAGE_TAG )
            }
            script {
                util_scrips.sendSlackNotification(APP_NAME, "SUCCESS", "DEPLOYED", IMAGE_TAG, "dev" )
            }
        }
        failure {
            sh 'echo [JENKINS_RESULT] FAILURE'
            script {
                util_scrips.sendSlackNotification(APP_NAME, "FAILED", "BUILD", IMAGE_TAG )
            }
        }
    }
}
