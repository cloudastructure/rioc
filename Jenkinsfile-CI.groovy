#!groovy
def util_scrips

pipeline {
    agent {
        label ""
    }

    options {
        timeout(time: 60, unit: 'MINUTES')
        buildDiscarder(logRotator(numToKeepStr: '20', artifactNumToKeepStr: '20', daysToKeepStr: '90', artifactDaysToKeepStr: '4'))
    }

    parameters {
        string( name: 'branch', defaultValue: 'main', description: 'repo branch? [ie: main]')
        string( name: 'SCRIPTS_REPO_BRANCH', defaultValue: 'kube/base', description: 'pipeline scripts repo branch ? [ie: kube/base]')
        booleanParam(name: 'HELM_DRY_RUN_DEBUG', defaultValue: false, description: 'If checked, do not deploy with helm, just show what would be deployed/installed')
    }

    environment {
        APP_NAME = "rioc"
        GROOVY_SCRIPTS_DIR = "scripts/groovy"
        GCE_ENVIRONMENT = "dev-dc-03"
    }

    stages {
        stage('checkout-app') {
            steps {
                // Override the branch from job-config SCM with the `branch` parameter
                // so manual builds can target any branch. IMAGE_TAG is computed after
                // this so it reflects the requested branch, not the implicit checkout.
                checkout([
                    $class: 'GitSCM',
                    branches: [[name: "*/${params.branch}"]],
                    extensions: scm.extensions,
                    userRemoteConfigs: scm.userRemoteConfigs
                ])
                script {
                    env.IMAGE_TAG = sh(returnStdout: true, script: "git log -n 1 --pretty=format:'%h'").trim()
                }
            }
        }

        stage('show-env') {
            steps {
                sh "ls -all"
                sh "env | sort"
                sh "git clean -fdx"
            }
        }

        stage('set-jenkins-build-id') {
            steps {
                script {
                    currentBuild.displayName = "$BUILD_NUMBER-$branch-$IMAGE_TAG"
                }
            }
        }

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
                // The repo has its own scripts/ dir (videodb_rtsp.sh). Pull the shared
                // CI scripts into ci-scripts/ to avoid clobbering app scripts that ship
                // in the Docker image.
                sh 'rm -rf ci-scripts'
                sh 'cp -rf ./cloned_repo/scripts ci-scripts'
                sh 'cp -rf ./cloned_repo/helms/deploy-app-helm k8s/'
                sh 'cp -rf ./cloned_repo/helms/values/*.yaml k8s/'
                sh 'rm -rf cloned_repo'
                stash includes: 'ci-scripts/**/*', name: 'ci-scripts'
                stash includes: 'k8s/**/*', name: 'k8s'
            }
        }

        stage('Init-Envs') {
            steps {
                unstash 'ci-scripts'
                script {
                    util_scrips = load "ci-scripts/groovy/Utils.groovy"
                }
            }
        }

        stage('Slack-notification-job-started') {
            steps {
                script {
                    util_scrips.sendSlackNotification(APP_NAME, "STARTED", "BUILD", IMAGE_TAG)
                }
            }
        }

        stage('docker-build-and-push') {
            steps {
                sh 'chmod +x ci-scripts/*'
                // Single Dockerfile at repo root — no GPU variants. The runtime image
                // is CPU-only; vision inference is offloaded to a remote vLLM server.
                sh "ci-scripts/custom-docker-build-path.sh ${APP_NAME} ${IMAGE_TAG} . Dockerfile"
                sh "ci-scripts/custom-docker-push.sh ${APP_NAME} ${GCE_ENVIRONMENT} ${IMAGE_TAG}"
            }
        }

        // Deployment is handled by a separate CD pipeline. Uncomment to deploy from CI:
        //
        // stage('deploy-to-dev-gke') {
        //     steps {
        //         unstash 'k8s'
        //         sh "ci-scripts/custom-deploy-helm.sh ${APP_NAME} ${GCE_ENVIRONMENT} ${IMAGE_TAG}"
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
                util_scrips.sendSlackNotification(APP_NAME, "SUCCESS", "BUILD", IMAGE_TAG)
            }
        }
        failure {
            sh 'echo [JENKINS_RESULT] FAILURE'
            script {
                util_scrips.sendSlackNotification(APP_NAME, "FAILED", "BUILD", IMAGE_TAG)
            }
        }
    }
}
