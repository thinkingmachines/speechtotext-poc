# Deployment Procedure
## Dev Deployment
### Pre-requisites

- Access to the project's GCP development environment
- GitHub Actions CI/CD workflow set up for dev deployment
- Necessary environment variables configured in GitHub Secrets:

  - `GCP_PROJECT_ID_DEV`
  - `GCP_SA_KEY_DEV` (Base64 encoded service account key)
  - `GCP_REGION_DEV`
  - `GCP_BUCKET_NAME_DEV`

### How-to-Guide

1. Ensure your changes are merged into the `dev` branch.
2. The CI/CD pipeline will automatically trigger when changes are pushed to the `dev` branch.
3. To manually trigger the dev deployment:

   - Go to the GitHub Actions tab in the project repository
   - Select the "Dev Deployment" workflow
   - Click "Run workflow" and select the `dev` branch

4. The deployment process includes:

   - Running unit tests
   - Building the Docker image
   - Pushing the image to Google Container Registry
   - Deploying the application to Cloud Run

5. Monitor the deployment progress in the GitHub Actions tab.
6. Once completed, verify the deployment:

   - Check the Cloud Run service in the GCP Console
   - Test the API endpoints using the provided dev URL

## Production Deployment
### Pre-requisites

- Access to the project's GCP production environment
- GitHub Actions CI/CD workflow set up for production deployment
- Necessary environment variables configured in GitHub Secrets:
  - `GCP_PROJECT_ID_PROD`
  - `GCP_SA_KEY_PROD` (Base64 encoded service account key)
  - `GCP_REGION_PROD`
  - `GCP_BUCKET_NAME_PROD`

- Approval from the project manager for production releases

### How-to-Guide

1. Create a new release branch from `master`:
```zsh
git checkout master
git pull origin master
git checkout -b release/v1.x.x
git push origin release/v1.x.x
```

2. Create a Pull Request to merge the release branch into `master`.

3. After approval and merging, tag the release:
```zsh
git checkout master
git pull origin master
git tag -a v1.x.x -m "Release v1.x.x"
git push origin v1.x.x
```

4. The production deployment CI/CD pipeline will automatically trigger when a new tag is pushed.
5. To manually trigger the production deployment:

- Go to the GitHub Actions tab in the project repository
- Select the "Production Deployment" workflow
- Click "Run workflow" and select the release tag

6. The production deployment process includes:

- Running unit and integration tests
- Building the Docker image with production configurations
- Pushing the image to Google Container Registry
- Deploying the application to Cloud Run in the production environment
- Updating the production database schema (if applicable)

7. Monitor the deployment progress in the GitHub Actions tab.
8. Once completed, perform the following verifications:

- Check the Cloud Run service in the GCP Production Console
- Run a series of smoke tests against the production API
- Monitor application logs and metrics for any anomalies

9. If issues are detected, be prepared to rollback:

- In the GCP Console, navigate to the Cloud Run service
- Select the previous revision and click "Roll back"

10. After successful deployment, update the project documentation with any relevant changes or new features.

Remember to always follow the principle of least privilege when setting up service accounts and IAM roles for deployments.

Regularly rotate secrets and review access permissions to maintain security.
