# $\lambda$ ML & CI/CD Functions Repository

This repository houses the source code for **AWS Lambda functions** used in our **Machine Learning (ML)** workflows and manages their deployment using an automated **Continuous Integration/Continuous Deployment (CI/CD)** pipeline.

-----

## 🎯 Purpose and Scope

  * **ML Functions:** Contains code for Lambda functions responsible for tasks like real-time model inference, scheduled data pre-processing, feature store interaction, and model retraining triggers.
  * **CI/CD:** Enforces a controlled deployment process, ensuring all code is thoroughly tested in a staging environment before reaching production.

-----


-----

## 🤝 Contribution and Merge Guidelines

  * **Target Branch:** All development PRs **must** target the **`UAT`** branch.
  * **Testing:** Ensure comprehensive unit tests are included for all new features or logic changes.
  * **Production Merge:** The PR from `UAT` to `Main` require and confirmation that UAT testing is complete and successful.
  * **Direct Push:** Direct pushes to the **`Main`** branch are blocked by branch protection rules.