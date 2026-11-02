@echo off
set PATH=%~dp0mingit\cmd;%~dp0;%PATH%

echo ========================================================
echo STEP 1: Log in to GitHub
echo ========================================================
echo You will see an 8-digit code. Please press ENTER when 
echo prompted, which will open your browser. Paste the code 
echo to securely log in.
echo.
gh auth login -w -p https
gh auth setup-git

echo.
echo ========================================================
echo STEP 2: Pushing code...
echo ========================================================
git push -u origin main -f

echo.
echo ALL DONE! You can close this window now.
pause
