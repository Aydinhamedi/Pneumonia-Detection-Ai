@echo off
set /p branchName="Enter the branch name to archive: "

:: Check if the branch exists locally
git rev-parse --verify %branchName% >nul 2>&1
if errorlevel 1 (
    echo The branch '%branchName%' does not exist locally.
    echo Fetching the branch from the remote repository...
    git fetch origin %branchName%:%branchName%
)

:: Check if the fetch was successful
git rev-parse --verify %branchName% >nul 2>&1
if errorlevel 1 (
    echo Failed to fetch the branch '%branchName%'. Please check the branch name and try again.
    exit /b
)

:: Archive the branch
set tagName=archive/%branchName%
git tag %tagName% %branchName%
git push origin %tagName%
echo Branch %branchName% has been archived as %tagName%.

:: Ask the user if they want to delete the branch
set /p deleteBranch="Do you want to delete the branch locally and remotely? (yes/no): "
if /i "%deleteBranch%"=="yes" (
    git branch -d %branchName%
    git push origin --delete %branchName%
    echo Branch %branchName% has been deleted.
)

:: Suggest a filename for the batch file
set filename=archive_git_branch_%branchName%_%date:/=-%.bat
echo Suggested filename for this batch file is: %filename%
:: Filename: archive_git_branch.bat

@echo off
set /p branchName="Enter the branch name to archive: "

:: Check if the branch exists locally
git rev-parse --verify %branchName% >nul 2>&1
if errorlevel 1 (
    echo The branch '%branchName%' does not exist locally.
    echo Fetching the branch from the remote repository...
    git fetch origin %branchName%:%branchName%
)

:: Check if the fetch was successful
git rev-parse --verify %branchName% >nul 2>&1
if errorlevel 1 (
    echo Failed to fetch the branch '%branchName%'. Please check the branch name and try again.
    exit /b
)

:: Archive the branch
set tagName=archive/%branchName%
git tag %tagName% %branchName%
git push origin %tagName%
echo Branch %branchName% has been archived as %tagName%.

:: Ask the user if they want to delete the branch
set /p deleteBranch="Do you want to delete the branch locally and remotely? (yes/no): "
if /i "%deleteBranch%"=="yes" (
    git branch -d %branchName%
    git push origin --delete %branchName%
    echo Branch %branchName% has been deleted.
)
