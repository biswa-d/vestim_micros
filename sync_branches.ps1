# Sync commits from pybattml_new to other branches
# Usage: .\sync_branches.ps1 [commit_hash]
# If no commit hash provided, uses the latest commit on current branch

param(
    [string]$CommitHash = ""
)

$ErrorActionPreference = "Stop"

# Function to display colored messages
function Write-Status {
    param([string]$Message, [string]$Color = "Cyan")
    Write-Host $Message -ForegroundColor $Color
}

function Write-Success {
    param([string]$Message)
    Write-Host "✓ $Message" -ForegroundColor Green
}

function Write-Error {
    param([string]$Message)
    Write-Host "✗ $Message" -ForegroundColor Red
}

# Get current branch
$currentBranch = git rev-parse --abbrev-ref HEAD
Write-Status "Current branch: $currentBranch"

# If no commit hash provided, use the latest commit
if ([string]::IsNullOrWhiteSpace($CommitHash)) {
    $CommitHash = git rev-parse HEAD
    Write-Status "No commit hash provided, using latest commit: $CommitHash"
} else {
    Write-Status "Using specified commit: $CommitHash"
}

# Get commit message for display
$commitMessage = git log -1 --pretty=format:"%s" $CommitHash
Write-Status "Commit message: $commitMessage" "Yellow"

# Ensure we're on pybattml_new
if ($currentBranch -ne "pybattml_new") {
    Write-Status "Switching to pybattml_new branch..."
    git checkout pybattml_new
}

# Push pybattml_new first
Write-Status "`nPushing pybattml_new to remote..." "Cyan"
git push origin pybattml_new
Write-Success "pybattml_new pushed"

# Stash any local changes
$stashOutput = git stash
$hasStash = $stashOutput -ne "No local changes to save"

# Define target branches
$targetBranches = @("main", "pybattml_old", "optimize_speed")

foreach ($branch in $targetBranches) {
    Write-Status "`n========================================" "Magenta"
    Write-Status "Processing branch: $branch" "Magenta"
    Write-Status "========================================" "Magenta"
    
    try {
        # Checkout the target branch
        Write-Status "Checking out $branch..."
        git checkout $branch 2>&1 | Out-Null
        
        # For main, use reset --hard instead of cherry-pick
        if ($branch -eq "main") {
            Write-Status "Resetting main to match pybattml_new..."
            git reset --hard pybattml_new
            Write-Status "Force pushing main..."
            git push origin main --force
            Write-Success "$branch synced via reset"
        } else {
            # For other branches, try cherry-pick
            Write-Status "Cherry-picking commit $CommitHash..."
            $cherryPickOutput = git cherry-pick $CommitHash 2>&1
            
            if ($LASTEXITCODE -eq 0) {
                Write-Success "Cherry-pick successful"
                
                # Try to push, handle conflicts
                Write-Status "Pushing $branch..."
                $pushOutput = git push origin $branch 2>&1
                
                if ($LASTEXITCODE -ne 0) {
                    # Push failed, try pull with rebase
                    Write-Status "Push failed, attempting pull --rebase..."
                    git pull --rebase origin $branch
                    Write-Status "Retrying push..."
                    git push origin $branch
                }
                
                Write-Success "$branch updated and pushed"
            } else {
                Write-Error "Cherry-pick failed for $branch"
                Write-Status "Output: $cherryPickOutput" "Red"
                
                # Abort cherry-pick
                git cherry-pick --abort 2>&1 | Out-Null
                Write-Status "Cherry-pick aborted, skipping $branch" "Yellow"
            }
        }
    } catch {
        Write-Error "Error processing $branch : $_"
        # Try to recover
        git cherry-pick --abort 2>&1 | Out-Null
        git merge --abort 2>&1 | Out-Null
    }
}

# Return to pybattml_new
Write-Status "`n========================================" "Magenta"
Write-Status "Returning to pybattml_new..." "Cyan"
git checkout pybattml_new

# Restore stash if we created one
if ($hasStash) {
    Write-Status "Restoring stashed changes..."
    git stash pop
}

Write-Status "`n========================================" "Green"
Write-Success "Sync complete!"
Write-Status "========================================" "Green"
Write-Status "`nSummary:"
Write-Status "  • Commit: $CommitHash"
Write-Status "  • Message: $commitMessage"
Write-Status "  • Branches synced: $($targetBranches -join ', ')"
