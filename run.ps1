# PowerShell script

param (
    [string]$arg1
)

if ($arg1 -eq "prod") {
    # Run jekyll with local configuration
    
    jekyll serve
} else {
    # Run jekyll without specifying a config
    jekyll serve --config _config-local.yml
}
