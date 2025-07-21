<h1 align="center">InteractVLM: 3D Interaction Reasoning from 2D Foundational Models</h1>

<p align="center">
  <img src="https://img.s<sup>†</sup> *We use Azure OpenAI GPT-4o image model to generate HOI-VQA for DAMON, LEMON and PIAD images. The script for calling Azure OpenAI API, raw data and preprocessing scripts are [here](https://github.com/saidwivedi/InteractVLM/tree/main/preprocess_data/generate_gpt4o_response).*ields.io/badge/CVPR-2025-blue" alt="CVPR 2025">
</p>

<div align="center"> 
    <a href="https://saidwivedi.in/">Sai Kumar Dwivedi</a><sup>1</sup>,
    <a href="https://anticdimi.github.io/">Dimitrije Antić</a><sup>2</sup>,
    <a href="https://sha2nkt.github.io//">Shashank Tripathi</a><sup>1</sup>,
    <a href="https://otaheri.github.io/">Omid Taheri</a><sup>1</sup>,<br/>
    <a href="https://thoth.inrialpes.fr/people/schmid/">Cordelia Schmid</a><sup>3</sup>,
    <a href="https://ps.is.mpg.de/person/black">Michael J. Black</a><sup>1</sup>,
    <a href="https://dtzionas.com/">Dimitrios Tzionas</a><sup>2</sup>
</div>

<br />

<div align="center"> 
<p style="text-align: center;"><span role="presentation" dir="ltr"><sup><small>1</small></sup>Max Planck Institute for Intelligent Systems, Tübingen<br /><sup><small>2</small></sup>University of Amsterdam&nbsp; &nbsp; &nbsp;<sup><small>3</small></sup>Inria, France</span></p>
<p style="text-align: center;"><span role="presentation" dir="ltr"></span></p>
<p></p>
</div>

<h5 align="center">
    <a href="https://interactvlm.is.tue.mpg.de">
        <img src="https://img.shields.io/website?url=http%3A//interactvlm.is.tue.mpg.de" alt="Website shields.io">
    </a>
    <a href="https://www.youtube.com/watch?v=brxygxM1nRk">
        <img src="https://img.shields.io/badge/YouTube-Watch-red?style=flat-square&logo=youtube" alt="YouTube Badge">
    </a>
    <a href="https://arxiv.org/abs/2504.05303">
        <img src="https://img.shields.io/badge/arXiv-2504.05303-b31b1b" alt="arXiv">
    </a>
    <img src="https://img.shields.io/badge/Status-Code%20Released-green" alt="Status: Code Released">
</h5><br />

<div style="display:flex;">
    <img src="assets/teaser.png">
</div>
<br />
<p style="text-align: justify;">
    <span style="color:#007acc; font-weight:bold;">InteractVLM</span> estimates 3D contact points on both human bodies and objects from single in-the-wild images, enabling accurate human-object joint reconstruction in 3D.
    We introduce a novel task, <span style="color:#cc6600; font-weight:bold; ">Semantic Human Contact</span>, which goes beyond the traditional Binary Human Contact to infer object-specific contacts on bodies. 
    By leveraging the rich visual knowledge of large <span style="color:#cc6600; font-weight:bold;">Vision-Language Models</span>, we address the limited availability of ground-truth 3D interaction data for training, resulting in better generalization to diverse real-world interactions.
</p>

### Joint Human-Object Reconstruction
<p align="center">
  <img src="assets/results/human_object/img1.png" width="22%">
  <img src="assets/results/human_object/img1_result.gif" width="22%">
  <img src="assets/results/human_object/img2.png" width="22%">
  <img src="assets/results/human_object/img2_result.gif" width="22%">
</p>

### Semantic Human Contact
<p align="center">
  <img src="assets/results/human_contact/img1.jpg" width="30%" alt="Input Image">
  <img src="assets/results/human_contact/img1_bench.gif" width="30%" alt="Contact Prediction">
  <img src="assets/results/human_contact/img1_bottle.gif" width="30%" alt="Joint Reconstruction">
</p>

## 🎯 Model Zoo

<div align="center">
<table>
  <thead>
    <tr>
      <th>#</th>
      <th>Model</th>
      <th>Type</th>
      <th>Training Datasets</th>
      <th>Comment</th>
      <th>Status</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>1</td>
      <td><a href="https://download.is.tue.mpg.de/download.php?domain=interactvlm&sfile=interactvlm-3d-hcontact-damon.zip"><code>interactvlm-3d-hcontact-damon</code></a></td>
      <td><img src="https://img.shields.io/badge/hcontact-blue" alt="hcontact"></td>
      <td><sub><a href="https://deco.is.tue.mpg.de">DAMON</a></sub></td>
      <td>Winner of RHOBIN Human Contact Challenge (CVPR 2025)</td>
      <td>
        <img src="https://img.shields.io/badge/✅_Available-green" alt="Available">
      </td>
    </tr>
    <tr>
      <td>2</td>
      <td><a href="https://download.is.tue.mpg.de/download.php?domain=interactvlm&sfile=interactvlm-3d-hcontact-wScene-damon-lemon-rich.zip"><code>interactvlm-3d-hcontact-wScene</code></a></td>
      <td><img src="https://img.shields.io/badge/hcontact-blue" alt="hcontact"></td>
      <td><sub><a href="https://deco.is.tue.mpg.de">DAMON</a> + <a href="https://yyvhang.github.io/LEMON/">LEMON-HU</a> + <a href="https://rich.is.tue.mpg.de">RICH</a></sub></td>
      <td>Best in-the-wild 3D Human Contact Estimation (with foot ground contact)</td>
      <td>
        <img src="https://img.shields.io/badge/✅_Available-green" alt="Available">
      </td>
    </tr>
    <tr>
      <td>3</td>
      <td><a href="https://download.is.tue.mpg.de/download.php?domain=interactvlm&sfile=interactvlm-3d-oafford-lemon-piad.zip"><code>interactvlm-3d-oafford-lemon-piad</code></a></td>
      <td><img src="https://img.shields.io/badge/oafford-purple" alt="oafford"></td>
      <td><sub><a href="https://yyvhang.github.io/LEMON/">LEMON-OBJ</a> + <a href="https://yyvhang.github.io/publications/IAG/index.html">PIAD</a></sub></td>
      <td>Estimates Object Affordance</td>
      <td>
        <img src="https://img.shields.io/badge/✅_Available-green" alt="Available">
      </td>
    </tr>
    <tr>
      <td>4</td>
      <td><a href="https://download.is.tue.mpg.de/download.php?domain=interactvlm&sfile=interactvlm-2d-hcontact.zip"><code>interactvlm-2d-hcontact</code></a></td>
      <td><img src="https://img.shields.io/badge/h2dcontact-teal" alt="h2dcontact"></td>
      <td><sub>Extended <a href="https://github.com/dvlab-research/LISA">LISA</a> by projecting <a href="https://deco.is.tue.mpg.de">DAMON</a> contact on images</sub></td>
      <td>2D Human Contact Segmentation via Referring Segmentation</td>
      <td>
        <img src="https://img.shields.io/badge/✅_Available-green" alt="Available">
      </td>
    </tr>
    <tr>
      <td>5</td>
      <td><a href="https://download.is.tue.mpg.de/download.php?domain=interactvlm&sfile=interactvlm-joint-reconstruction.zip"><code>interactvlm-joint-reconstruction</code></a></td>
      <td>
        <img src="https://img.shields.io/badge/hcontact-blue" alt="hcontact"> <img src="https://img.shields.io/badge/ocontact-yellow" alt="ocontact">
      </td>
      <td>
        <sub>
        <a href="https://deco.is.tue.mpg.de">DAMON</a> + 
        <a href="https://yyvhang.github.io/LEMON/">LEMON-HU</a> +
        <a href="https://rich.is.tue.mpg.de">RICH</a> +  
        <a href="https://yyvhang.github.io/LEMON/">LEMON-OBJ</a> + 
        <a href="https://yyvhang.github.io/publications/IAG/index.html">PIAD</a> + 
        <a href="https://pico.is.tue.mpg.de">PICO</a> + 
        <a href="https://github.com/saidwivedi/InteractVLM/tree/main/preprocess_data/generate_gpt4o_response/README.md">HOI-VQA</a><sup>#</sup>
        </sub>
      </td>
      <td>Single Model for Joint 3D Human Object Contact Estimation</td>
      <td>
        <img src="https://img.shields.io/badge/✅_Available-green" alt="Available">
      </td>
    </tr>
  </tbody>
</table>
</div>

<sup>*</sup> *The `interactvlm-joint-reconstruction` model is trained with our new **[PICO Dataset (CVPR 2025)](https://pico.is.tue.mpg.de)**, which enables accurate 3D object contact estimation unlike object affordance using **LEMON-OBJ** and **PIAD** dataset.*

<sup>#</sup> *We use GPT-4o image model to generate HOI-VQA for DAMON, LEMON and PIAD images. The script for calling OpenAI API, raw data and preprocessing scripts are [here](https://github.com/saidwivedi/InteractVLM/tree/main/preprocess_data/generate_gpt4o_response/).*

---

## ⚙️ Installation

### 🛠️ Setup Environment

1. **Install Micromamba** (if not already installed):
   ```bash
   curl -Ls https://micro.mamba.pm/api/download/linux-64/latest | tar -xvj bin/micromamba
   sudo mv bin/micromamba /usr/local/bin/
   ```

2. **Create and activate environment**:
   ```bash
   micromamba create -n interactvlm python=3.10 -c conda-forge
   micromamba activate interactvlm
   ```

3. **Install PyTorch with CUDA 12.1**:
   ```bash
   pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
   ```

4. **Clone the repository**:
   ```bash
   git clone https://github.com/saidwivedi/InteractVLM.git
   cd InteractVLM
   ```

5. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   pip install flash-attn --no-build-isolation
   DS_BUILD_FUSED_ADAM=1 pip install deepspeed==0.15.1
   ```

6. **Setup Environment**:
   ```bash
   # Before running demo, training or evaluation scripts, ensure CUDA is properly configured
   export CUDA_HOME=/usr/local/cuda  # or your CUDA installation path
   export PATH=$CUDA_HOME/bin:$PATH
   export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
   ```

## 📁 Code Structure

```
InteractVLM/
├── 📁 model/                         # Core model implementation
├── 📁 datasets/                      # Data loading and processing
├── 📁 utils/                         # Utility functions
├── 📁 preprocess_data/               # Data preprocessing scripts
├── 📁 scripts/                       # Execution scripts
├── 📁 data/                          # Dataset folders, Body models, Demo samples
├── 📁 trained_models/                # Trained models
├── 📄 train.py                       # Main training script
├── 📄 evaluate.py                    # Main evaluation script
├── 📄 run_demo.py                    # Run Demo
└── 📄 requirements.txt               # Python dependencies
```

## 📦 Data and Model Downloads

### 📁 Essential Data Files

To run InteractVLM, you need to download essential data files and pre-trained models. We provide a convenient script to handle this process.

### 🚀 Download Script Usage

1. **Register for access** at [https://interactvlm.is.tue.mpg.de/login.php](https://interactvlm.is.tue.mpg.de/login.php) to get your credentials

2. **Run the download script**:
   ```bash
   bash fetch_data.sh
   ```

### 🎮 Demo

Run the demo on your own images with either human or object interaction estimation modes:

```bash
# For 3D human contact estimation
bash scripts/run_demo.sh hcontact data/demo_samples folder

# For 2D human contact segmentation
bash scripts/run_demo.sh h2dcontact data/demo_samples file

# For 3D object affordance estimation  
bash scripts/run_demo.sh oafford data/demo_samples folder
```

**Demo Requirements:**

- **Human Contact Demo**: The canonical human mesh and rendered input are already provided. Simply run the script to estimate 3D contact points on human bodies. We now also support human contact estimation with scene (e.g. ground or undefined objects) with the latest released model. Download the latest model using `hcontact-wScene` argument in `fetch_data.sh` and use the same argument while running the demo script. The object name in the image filename serves as the query object for contact estimation (e.g., "bottle" or "chair"). To estimate contact with the scene or ground, use "scene" as the query or prefix the filename with "scene".

- **2D Human Contact Demo**: Performs 2D contact segmentation directly on the input image using referring segmentation. This extends LISA's capabilities for human-object contact detection in 2D space. The object name in the image filename serves as the query object for contact estimation.

- **Object Affordance Demo**: The code expects an object mesh as input. The script will automatically render multiple views of the object for affordance prediction.

**Input Modes:**

The demo supports two input structures:

1. **Folder-based mode** (default): Each sample in its own folder (required for 3D human contact and object affordance)
2. **File-based mode**: All samples as files in a single folder. Supported for:
   - **2D Human Contact** (`h2dcontact`): Direct segmentation on input images
   - **3D Human Contact** (`hcontact`): Estimating human contact for video frames
   
**Sample Data**: The `data/demo_samples/` directory contains ready-to-use samples for testing both human contact and object affordance estimation. One should get the following results:

<p align="center">
  <img src="assets/demo_results/tennis_racket__000000041045.png" width="22%" style="border:2px solid #444; padding:2px;">
  <img src="assets/demo_results/interactvlm-3d-hcontact-damon_tennis_racket__000000041045_hcontact_concat.jpg" width="22%" style="border:2px solid #444; padding:2px;">
  <img src="assets/demo_results/interactvlm-3d-oafford-lemon-piad_tennis_racket__000000041045_oafford_concat.jpg" width="22%" style="border:2px solid #444; padding:2px;">
</p>


## 🏋️ Training and Evaluation

### 🔧 Data Generation

To generate the data needed for training, run the following script. For now, we provide preprocessed dataset for DAMON. We will soon release for LEMON, PIAD and PICO.

```bash
# Generate preprocessed data
bash scripts/run_datagen.sh
```

### 🚀 Training

To train 3D Human Contact Estimation using DAMON dataset, download the preprocessed dataset using the following command and place it under `data/damon`. Then run the training script.

```bash
# Download preprocessed DAMON dataset
bash fetch_data.sh damon-dataset

# Train human contact with DAMON dataset
bash scripts/run_train.sh hcontact-damon
```

### 📊 Evaluation

#### **Model Weight Preparation**
If you have trained a new model, prepare the weights for evaluation:

```bash
# Prepare weights for model 0 (adjust number as needed)
bash scripts/run_prepare_weights.sh 0
```

#### **Run Evaluation on Pre-trained Models**
```bash
# Evaluate the model on either DAMON or PIAD. Adjust the congfiguration accordingly
bash scripts/run_eval.sh
```

## 📋 Code Release Status

### ✅ **Released**
- **3D Human Contact Estimation** - Training, evaluation, and demo code available
- **3D Object Contact/Affordance Estimation** - Training, evaluation, and demo code available

### 📅 **Pending**
- **Object Shape Retrieval from Single Image** - Code release pending
- **Optimization Pipeline for Joint Reconstruction** - Code release pending

## 🙏 Acknowledgements

We thank
<a href="https://is.mpg.de/person/acseke">Alpár Cseke</a> 
for his assistance with evaluating joint human-object reconstruction.
We also thank
<a href="https://ps.is.mpg.de/person/talexiadis">Tsvetelina Alexiadis</a> and
<a href="https://ps.is.mpg.de/person/tmcconnell">Taylor Obersat</a>
for MTurk evaluation,
<a href="https://yfeng95.github.io">Yao Feng</a>,
<a href="https://kulits.github.io">Peter Kulits</a>, and
<a href="https://ps.is.mpg.de/person/mdiomataris">Markos Diomataris</a> 
for their valuable feedback and
<a href="https://is.mpg.de/~bpellkofer">Benjamin Pellkofer</a>
for IT support.
SKD is supported by the International Max Planck Research School for Intelligent Systems (IMPRS-IS). 
The UvA part of the team is supported by an ERC Starting Grant (STRIPES, 101165317, PI: D. Tzionas).

### Code and Datasets
InteractVLM builds upon several excellent open-source projects and datasets:

- **[LISA](https://github.com/dvlab-research/LISA)** - InteractVLM is built on top of this foundational framework
- **[LEMON](https://yyvhang.github.io/LEMON/)**, **[DECO](https://deco.is.tue.mpg.de)**, **[PIAD](https://github.com/yyvhang/IAGNet)**, **[PICO](https://pico.is.tue.mpg.de)** and **[RICH](https://rich.is.tue.mpg.de)** - For human contact and object affordance data
- **[Blendify](https://github.com/ptrvilya/blendify/)** - For rendering

### Optimization Pipeline
Our optimization pipeline integrates the following repositories:

- **[OpenShape](https://github.com/Colin97/OpenShape_code)** - For object shape retrieval
- **[OSX](https://github.com/IDEA-Research/OSX)** - For SMPLX human pose estimation  
- **[Grounded-SAM](https://github.com/IDEA-Research/Grounded-Segment-Anything)** - For object detection and segmentation
- **[Depth Pro](https://github.com/apple/ml-depth-pro)** - For depth estimation



## 📝 Citation
If you find this code useful for your research, please consider citing the following paper:

```bibtex
@inproceedings{dwivedi_interactvlm_2025,
    title     = {{InteractVLM}: {3D} Interaction Reasoning from {2D} Foundational Models},
    author    = {Dwivedi, Sai Kumar and Antić, Dimitrije and Tripathi, Shashank and Taheri, Omid and Schmid, Cordelia and Black, Michael J. and Tzionas, Dimitrios},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2025},
}
```
## ⚖️ License

This code is available for **non-commercial scientific research purposes** as defined in the [LICENSE file](LICENSE). By downloading and using this code you agree to the terms in the [LICENSE](LICENSE). Third-party datasets and software are subject to their respective licenses.

## 📧 Contact

For code related questions, please contact sai.dwivedi@tuebingen.mpg.de

For commercial licensing (and all related questions for business applications), please contact ps-licensing@tue.mpg.de.

