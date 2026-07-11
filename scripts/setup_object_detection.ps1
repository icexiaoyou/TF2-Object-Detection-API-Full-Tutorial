[CmdletBinding()]
param(
    [string]$ModelsDirectory = "third_party/models",
    [string]$Ref = "master",
    [switch]$SkipTests
)

$ErrorActionPreference = "Stop"
if (-not (Get-Command protoc -ErrorAction SilentlyContinue)) {
    throw "protoc was not found. Run: conda install -c conda-forge protobuf"
}

$modelsPath = Join-Path (Get-Location) $ModelsDirectory
if (-not (Test-Path $modelsPath)) {
    git clone https://github.com/tensorflow/models.git $modelsPath
}
git -C $modelsPath fetch --tags --prune
git -C $modelsPath checkout $Ref
git -C $modelsPath pull --ff-only

Push-Location (Join-Path $modelsPath "research")
try {
    $protos = Get-ChildItem -Path "object_detection/protos" -Filter "*.proto"
    & protoc $protos.FullName --python_out=.
    Copy-Item "object_detection/packages/tf2/setup.py" "setup.py" -Force
    python -m pip install --upgrade pip
    python -m pip install .
    if (-not $SkipTests) {
        python object_detection/builders/model_builder_tf2_test.py
    }
}
finally {
    Pop-Location
}
