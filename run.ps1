# PowerShell script

param (
    [string]$arg1
)

if ($arg1 -eq "local") {
    # Run jekyll with local configuration
    jekyll serve --config _config-local.yml
} else {
    # Run jekyll without specifying a config
    jekyll serve
}
