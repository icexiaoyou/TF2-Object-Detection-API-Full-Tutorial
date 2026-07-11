# TensorFlow 2 Object Detection API Tutorial

A maintained Windows and Conda workflow for fine-tuning TensorFlow 2 Object Detection API models on a Pascal VOC dataset. It replaces the pinned 2022 environment, global `PYTHONPATH`, and manual system-wide Protobuf installation.

The tracked `models/` directory is an archived snapshot retained for historical reference. New work uses a fresh clone of the official TensorFlow Model Garden under `third_party/models`, installed inside the Conda environment.

## 1. Create the environment

TensorFlow is installed from its official pip wheels inside Conda. This project defaults to CPU execution; no CUDA, cuDNN, or GPU setup is required.

```powershell
conda env create -f environment.yml
conda activate tf2-object-detection
```

## 2. Install Object Detection API

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup_object_detection.ps1
```

The script clones the latest Model Garden, compiles protobuf definitions, installs the API into the active environment, and runs the official model-builder test. Use `-SkipTests` only when the installation has already been verified.

## 3. Prepare data

Place each image beside its Pascal VOC XML annotation in `workspace/training_demo/images`. Split and create records with explicit paths:

```powershell
cd workspace/training_demo
python scripts/pre-process/partition_dataset.py --image-dir images --test-ratio 0.2 --copy-xml
python scripts/pre-process/generate_tfrecord.py --xml-dir images/train --image-dir images/train --labels-path annotations/label_map.pbtxt --output-path annotations/train.record
python scripts/pre-process/generate_tfrecord.py --xml-dir images/test --image-dir images/test --labels-path annotations/label_map.pbtxt --output-path annotations/test.record
```

## 4. Train and export

Download a model and matching `pipeline.config` from the official [TF2 Detection Model Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md). Set the model path, TFRecord paths, label-map path, batch size, class count, and fine-tune checkpoint in that configuration.

Then run the installed Model Garden commands from the repository root:

```powershell
python third_party/models/research/object_detection/model_main_tf2.py --pipeline_config_path=workspace/training_demo/models/my_model/pipeline.config --model_dir=workspace/training_demo/models/my_model --alsologtostderr
python third_party/models/research/object_detection/exporter_main_v2.py --input_type=image_tensor --pipeline_config_path=workspace/training_demo/models/my_model/pipeline.config --trained_checkpoint_dir=workspace/training_demo/models/my_model --output_directory=workspace/training_demo/exported-models/my_model
```

The deployment helper remains in `workspace/training_demo/scripts/process`. All project scripts support `--help`.
