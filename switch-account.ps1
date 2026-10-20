# ==============================================================================
# GitHub Multi-Account Switcher for Story-SLM
# ==============================================================================
# Usage:
#   .\switch-account.ps1
#   .\switch-account.ps1 -Account 1  (Switch to VarunHujan1403)
#   .\switch-account.ps1 -Account 2  (Switch to Second Account)
# ==============================================================================

param (
    [string]$Account = ""
)

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "         GitHub Multi-Account Switcher (Local Repo)       " -ForegroundColor Yellow
Write-Host "==========================================================" -ForegroundColor Cyan

# Ensure Git Credential Manager tracks individual repository URLs & usernames
git config --local credential.useHttpPath true

$configDir = "$PSScriptRoot\.antigravity"
if (-not (Test-Path $configDir)) {
    New-Item -ItemType Directory -Path $configDir -Force | Out-Null
}
$accountsFile = "$configDir\github_accounts.json"

# Load saved profiles if existing
$profiles = @{}
if (Test-Path $accountsFile) {
    try {
        $profiles = Get-Content $accountsFile -Raw | ConvertFrom-Json
    } catch {
        $profiles = @{}
    }
}

# Default Account 1 (Current repo origin)
$account1 = @{
    Username = "VarunHujan1403"
    Email    = "varunanss@gmail.com"
    RepoUrl  = "https://github.com/VarunHujan/story-slm.git"
}

# Determine action
if ($Account -eq "1") {
    $targetUsername = $account1.Username
    $targetEmail    = $account1.Email
    $targetRepo     = $account1.RepoUrl
} elseif ($Account -eq "2" -and $profiles.Account2) {
    $targetUsername = $profiles.Account2.Username
    $targetEmail    = $profiles.Account2.Email
    $targetRepo     = $profiles.Account2.RepoUrl
} else {
    Write-Host "`nChoose an option:" -ForegroundColor White
    Write-Host "  [1] Switch to Account 1: VarunHujan1403 (varunanss@gmail.com)" -ForegroundColor Green
    if ($profiles.Account2) {
        Write-Host "  [2] Switch to Saved Account 2: $($profiles.Account2.Username) ($($profiles.Account2.Email))" -ForegroundColor Cyan
        Write-Host "  [3] Set up / Connect a New GitHub Account" -ForegroundColor Magenta
    } else {
        Write-Host "  [2] Set up & Connect New GitHub Account (Interactive Login)" -ForegroundColor Magenta
    }
    
    $choice = Read-Host "`nEnter option number (1, 2, or 3)"
    
    if ($choice -eq "1") {
        $targetUsername = $account1.Username
        $targetEmail    = $account1.Email
        $targetRepo     = $account1.RepoUrl
    } elseif ($choice -eq "2" -and $profiles.Account2) {
        $targetUsername = $profiles.Account2.Username
        $targetEmail    = $profiles.Account2.Email
        $targetRepo     = $profiles.Account2.RepoUrl
    } else {
        Write-Host "`n--- Setup New GitHub Account ---" -ForegroundColor Yellow
        $targetUsername = Read-Host "Enter Second GitHub Username"
        $targetEmail    = Read-Host "Enter Second GitHub Email"
        $targetRepo     = Read-Host "Enter Target Repository URL (e.g. https://github.com/Owner/repo.git)"

        if ([string]::IsNullOrWhiteSpace($targetUsername) -or [string]::IsNullOrWhiteSpace($targetRepo)) {
            Write-Host "Username and Repository URL cannot be empty!" -ForegroundColor Red
            exit 1
        }

        # Save profile for easy 1-key switching in the future
        $saveData = [PSCustomObject]@{
            Account2 = [PSCustomObject]@{
                Username = $targetUsername
                Email    = $targetEmail
                RepoUrl  = $targetRepo
            }
        }
        $saveData | ConvertTo-Json | Set-Content $accountsFile
        Write-Host "Profile saved to local configuration." -ForegroundColor DarkGray
    }
}

# Format authenticated URL with username embedded
# Example: https://myuser@github.com/owner/repo.git
$cleanUrl = $targetRepo -replace "https://[^@]+@", "https://"
$authUrl  = $cleanUrl -replace "https://", "https://$($targetUsername)@"

Write-Host "`nConfiguring repository settings..." -ForegroundColor Cyan

# 1. Update remote URL
git remote set-url origin $authUrl

# 2. Update local committer identity
git config --local user.name "$targetUsername"
if (-not [string]::IsNullOrWhiteSpace($targetEmail)) {
    git config --local user.email "$targetEmail"
}

Write-Host "`nActive Git Configuration:" -ForegroundColor Green
Write-Host "  Username : $(git config user.name)"
Write-Host "  Email    : $(git config user.email)"
Write-Host "  Remote   : $(git remote get-url origin)"

Write-Host "`n----------------------------------------------------------" -ForegroundColor Yellow
Write-Host "Opening GitHub Browser Authentication Window for '$targetUsername'..." -ForegroundColor Cyan
Write-Host "Your browser will open. Just click 'Sign in' / 'Authorize'!" -ForegroundColor Yellow
Write-Host "----------------------------------------------------------`n" -ForegroundColor Yellow

# Trigger Git Credential Manager authentication popup
git fetch origin
if ($LASTEXITCODE -eq 0) {
    Write-Host "`n SUCCESS! Connected to $targetRepo as '$targetUsername'!" -ForegroundColor Green
} else {
    Write-Host "`nFetch finished. You can now run 'git push' or 'git pull' seamlessly." -ForegroundColor Cyan
}
