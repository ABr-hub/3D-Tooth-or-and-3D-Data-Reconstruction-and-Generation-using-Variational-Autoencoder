# 🦷 3D Tooth (or any 3D Data) Reconstruction and Generation using Variational-Autoencoder (VAE) 
This project utilizes a Variational Autoencoder (VAE) to reconstruct and generate synthetic 3D tooth models from point clouds. The model creates new tooth shapes for quantitative and visual analysis. It includes details on training, optimization, and performance evaluation using metrics like Chamfer Distance.

**Repository content**
  1. Project description
  2. Installation
  3. Usage
  4. Contributing

( :exclamation: ***This repository is under construction, please be patient***)

## Project Description
### Motivation

This project addresses the need for advanced modeling techniques in dentistry, allowing for the reconstruction and generation of detailed 3D tooth models. By employing a VAE, the project leverages the power of machine learning to enhance dental applications, improving diagnostics, treatment planning, and overall patient care.

**Possible Applications**
1. **Automated Diagnosis**: A VAE can analyze 3D dental scans to identify abnormalities or conditions such as cavities, misalignment, or periodontal disease. By learning from a variety of healthy and unhealthy tooth structures, it can assist dentists in making accurate diagnoses.
2. **Custom Prosthetics and Implants**: The model can generate highly personalized dental implants and prosthetics based on a patient's unique dental morphology. This results in improved fit and functionality, enhancing the overall patient experience.
3. **Predictive Analytics for Treatment Outcomes**: Utilizing historical treatment data, a VAE can help predict the outcomes of various dental procedures. This assists dental professionals in choosing the most effective treatment options for their patients.

*While this project focuses on 3D data of teeth, it can be adapted for various applications involving 3D data in other fields.*

### Content
This project includes the following key components:

1. **Synthetic Tooth Model Creation:** Generate synthetic tooth models that serve as placeholders for real 3D data, providing a foundation for training and evaluation of the model. Here we start with a simple 3D elipsoid.

     <img src="assets\3D_Toothmodel.gif" width="300"/>

3. **Data Preprocessing:** Implement preprocessing techniques to convert 3D models into point clouds, ensuring they are suitable for deep learning applications.

   <img src="assets\3D_Toothmodel_Pointcloud.gif" width="300"/>

4. **Variational Autoencoder (VAE) Design:** Develop a VAE that learns the statistical distribution of the 3D data, enabling effective reconstruction and generation of new tooth models. The basic working principle is depicted below
   *For more details see the respective folder*
   
     <img src="assets\VAE_.png" width="500"/>

5. **Model Application:** Apply the trained VAE to reconstruct existing 3D models (e.g to eliminate noise or incomplete areas in order to obtain higher-quality data) and generate new synthetic tooth data, facilitating quantitative and qualitative analysis. *(left: reconstruction, right: generated)*

     <img src="assets\Plot_recon_&_generated.png" width="500"/>

     Reconstruction Results:
     * The reconstructed point clouds demonstrate that the VAE captures the underlying features of the input data (synthetic 3D teeth) reasonably well.
     * While small differences between the original and reconstructed point clouds are noticeable, the overall shape and basic structures are preserved.
     * This indicates that the VAE is capable of learning the statistical distributions of the input data and reconstructing them with sufficient accuracy.

     Generation Results:
     * The generated point clouds show larger deviations and lack of detail compared to the original data.
     * The generated tooth shapes appear "blurry" or not precise enough for real-world applications.
     * This suggests that the complexity of the generated 3D data is not sufficient to capture finer structures typically found in real tooth data.

     Interpretation:
     * The VAE performs well in reconstructing previously seen data but struggles with generating completely new data.
     * For real-world applications (e.g., dental prosthetics, 3D modeling), further optimization is required to improve the quality and diversity of the generated data.
     * Possible improvements could involve using a larger dataset and a deeper VAE model to capture more complex patterns.
     

## Installation Instructions
To set up the project, follow these steps:
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/3d-tooth-reconstruction.git
   cd 3d-tooth-reconstruction
   ```
2. Install the required packages (e.g.)
   ```bash
   pip install trimesh
   ```

## Usage
To use the model for reconstruction or generation of 3D tooth models, run the following command:
```bash
   python main.py
```

Make sure to adjust the configuration parameters in the script as needed.

### Contributing
Contributions are welcome! If you would like to contribute, please follow these steps:

  1. Fork the repository.
  2. Create a new branch for your feature or bug fix.
  3. Commit your changes and push the branch.
  4. Create a pull request with a detailed description of your changes.




