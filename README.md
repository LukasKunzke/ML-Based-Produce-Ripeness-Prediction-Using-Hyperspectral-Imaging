# Hyperspectral Imaging: Unlocking Economic and Environmental Potential in Produce Supply Chains  

---

## Economic and Environmental Opportunities  
The produce industry, in Germany and globally, can reap significant benefits from hyperspectral imaging:  

- **Economic Potential**:  
  - Reducing waste by even 1% could save millions of euros annually.  
  - Premium pricing opportunities arise from delivering consistently high-quality, optimally ripe produce.  
  - Dynamic pricing and improved logistics streamline supply chain efficiency.  

- **Environmental Impact**:  
  - Reducing spoilage minimizes waste and lowers carbon emissions associated with food loss.  
  - Non-destructive testing supports sustainability goals, preserving resources and enabling better food utilization.  

---

## Current Ripeness Detection Challenges  
Traditional ripeness detection methods, such as visual inspection, taste testing, and destructive sampling, face notable limitations:  

- **Inconsistency**: Results are subjective, leading to variability.  
- **Inefficiency**: Manual, time-consuming processes slow operations.  
- **Waste**: Destructive methods result in unusable samples, contributing to food loss.  
- **Heuristic Approach**: The quality of an entire batch is often assessed based on a few destructive samples instead of evaluating produce individually.  

---

## Ripeness Detection with Hyperspectral Imaging  
Hyperspectral imaging overcomes these challenges by analyzing the chemical and physical properties of produce. Key capabilities include:  

- **Non-Destructive Testing**: Preserves produce integrity for sale or further testing.  
- **Spatial Analysis**: Detects localized ripeness variations within individual pieces of produce.  
- **Early Detection**: Identifies internal ripeness or defects before external signs appear.  

---

## Benefits and Disadvantages of Hyperspectral Imaging  

### Benefits  
- High-speed analysis enables efficient processing of large volumes.  
- Objective, consistent results reduce human error.  
- Automation-ready for seamless integration into existing workflows.  
- Adaptable across multiple produce types with customized wavelength calibration.  
- Long-term usability due to minimal changes in produce spectral properties over time.  
- **Rich Data Analysis**: Can predict attributes such as sugar content, chlorophyll, carotenoids, and ripeness stage.  

### Disadvantages  
- **Cost**: Initial investment in equipment and calibration may be significant, though costs can be reduced by integrating into existing systems like conveyor belts.  
- **Complexity**: Requires specialized expertise for setup and ongoing maintenance.  
- **Integration Time**: Supply chain modifications may take time and require phased implementation.  

---

## Costs of Integration for Produce Sellers  
- **Hardware Costs**: Includes hyperspectral cameras and scanners.  
- **Implementation**: Adapting these systems to existing workflows may involve infrastructure upgrades, though economies of scale can lower costs in the long run.  

---

## Costs of Model Creation  
- **Software and Calibration**: Tailoring systems for specific produce types, workflows, and equipment.  
- **Data Collection**: Gathering and labeling data for different types of produce can often be integrated into existing workflows at minimal cost.  
- **Training**: Educating staff on system operation and maintenance.  

---

## Summary: Economic and Environmental Benefits  
By adopting hyperspectral imaging, the produce industry can achieve:  

- **Economic Gains**: Reduced waste, improved logistics, and increased revenue from premium-quality produce.  
- **Sustainability**: Reduced environmental impact through minimized food loss and better resource utilization.  

Hyperspectral imaging is a transformative technology for modernizing ripeness detection, enhancing quality control, and optimizing supply chains in Germany and beyond.  

---

## Dataset and Comparative Performance  

To demonstrate the efficacy of hyperspectral imaging for ripeness detection, I developed a **machine learning pipeline** using the [DeepHS Fruit v2 dataset](https://paperswithcode.com/dataset/deephs-fruit-v2).  

### Dataset  
The DeepHS Fruit v2 dataset was introduced alongside the paper **[Measuring the Ripeness of Fruit with Hyperspectral Imaging and Deep Learning](https://arxiv.org/pdf/2104.09808v1)** by Varga, Makowski, and Zell (2021). It contains hyperspectral images of various fruits, captured using advanced imaging technology, including the INNOSPEC RedEye and SpecimFX 10 cameras.  

This project focuses specifically on **kiwis** photographed with the SpecimFX 10 camera.  

### Paper  
The original paper employed a custom **32k-parameter convolutional neural network (CNN)** to assess kiwi ripeness, achieving:  
- **Accuracy (INNOSPEC RedEye)**: 77.8%  
- **Accuracy (SpecimFX 10)**: 66.7%  

---

## Performance  
Using an optimized approach, I implemented a pipeline that:  
- **Improves accuracy with the SpecimFX10 from 66% to 73%** compared to the best-performing model in the reference paper through better feature selection and model tuning.  
- **Translates to an MAE of approximately 0.347**. Based on ripeness labels ranging from 1 to 3, and an average kiwi shelf life of 10 days, this suggests that the model predicts kiwi ripeness to within about one day on average.  
- **Decreases computational load, training, and inference time**, reducing resource demands.  

---

## Methodology  
### Key Steps
- Treating the problem as a regression task instead of categorization.  
- Using only a vertical slice of the images.  
- Computing a number of statistics based on the selected slice:  
  - `np.min(two_d_slice)`
  - `np.percentile(two_d_slice, 3)`
  - `np.percentile(two_d_slice, 5)`
  - `np.percentile(two_d_slice, 10)`
  - `np.percentile(two_d_slice, 15)`
  - `np.percentile(two_d_slice, 20)`
  - `np.percentile(two_d_slice, 25)`
  - `np.percentile(two_d_slice, 30)`
  - `np.percentile(two_d_slice, 35)`
  - `np.percentile(two_d_slice, 40)`
  - `np.percentile(two_d_slice, 45)`
  - `np.median(two_d_slice)`
  - `np.mean(two_d_slice)`
  - `np.percentile(two_d_slice, 55)`
  - `np.percentile(two_d_slice, 60)`
  - `np.percentile(two_d_slice, 65)`
  - `np.percentile(two_d_slice, 70)`
  - `np.percentile(two_d_slice, 75)`
  - `np.percentile(two_d_slice, 80)`
  - `np.percentile(two_d_slice, 85)`
  - `np.percentile(two_d_slice, 90)`
  - `np.percentile(two_d_slice, 95)`
  - `np.percentile(two_d_slice, 97)`
  - `np.max(two_d_slice)`
  - `np.ptp(two_d_slice)`
  - `np.var(two_d_slice)`
  - `np.std(two_d_slice)`
  - `np.sum(two_d_slice)`
  - `np.mean(np.abs(two_d_slice - np.mean(two_d_slice)))`
  - `scipy.stats.skew(two_d_slice.flatten())`
  - `scipy.stats.kurtosis(two_d_slice.flatten())`
- Reducing the number of components with PCA and standardizing the statistics.  




### 1. Prepare DataFrame  
- Load and join JSON files containing labels into DataFrames.  
- Concatenate the DataFrames.  
- Drop irrelevant columns and convert columns to appropriate data types. 


### 2. Handle Missing Values  
- Check for missing values.  
- Identify rows with missing values.  
- Impute missing values using KNN Imputer.  

### 3. Label Refinement  
- Create new columns (e.g., 'weight loss %').  
- Train SVR models and predict target variables.  
- Add the predictions to the original DataFrame.  
- Standardize values in target label columns for comparison with the reference paper.  

### 4. Verify Label Distributions  
- Define a function to check the distribution of labels in the specified column.  
- Calculate and display the frequency of each label.  
- Plot the frequency of each label.  

### 5. Load, Preprocess, and Save Images  
- Load images with paths from the CSV file.  
- Define parameters for loading spectral data.  
- Load spectral data using the defined parameters.  
- Save preprocessed images to a file in the 'data/processed' directory.  

### 6. Model Fitting  
- Set target and reference labels.  
- Split the data and labels into training and testing sets.  
- Evaluate model configurations.  
- Optimize features using backward elimination.  
- Estimate accuracy with the optimized features.  



---

## Limitations  
The two most pressing concerns are:  
- **Ripeness Labels**: Labels are not entirely objective, introducing some noise.  
- **Dataset Size**: The relatively small sample size limits generalizability.  

---

### Acknowledgment  
I would like to thank **Steffen Meinert** for his valuable input and guidance during the development of the initial pipeline during my time with the **Semantic Information Systems Group at Universität Osnabrück**.  
