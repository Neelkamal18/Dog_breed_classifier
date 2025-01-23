# City Dog Show Image Classification Project

## Project Overview
The project involves using a provided classifier function to analyze images submitted during a citywide dog show registration. The main objective is to identify whether submitted images depict dogs and, if so, determine their breed using various convolutional neural network (CNN) architectures. The classifier will help distinguish dogs from other animals and verify the breed of the registered dogs.

## Objectives
1. **Dog Identification**: Correctly identify images of dogs and distinguish them from images of other animals or objects.
2. **Breed Classification**: Accurately classify the breed of the dogs.
3. **Algorithm Comparison**: Evaluate three CNN architectures—AlexNet, VGG, and ResNet—to determine which performs best for objectives 1 and 2.
4. **Performance Analysis**: Consider the runtime and computational efficiency of the algorithms while balancing accuracy.

## Project Instructions
The main program file, `check_images.py`, contains the necessary functions to achieve the above objectives. The following steps outline the key tasks:

### 1. Timing the Program
- Use Python's `time` module to compute the total runtime of the program.

### 2. User Inputs
- Collect command-line arguments for inputs such as the folder containing pet images, the desired CNN architecture, and output options.

### 3. Creating Pet Image Labels
- Use the filenames in the `pet_images` folder to create labels for each image.
- Store the pet image labels in a dictionary with the filename as the key and the label as the value.

### 4. Classifier Labels and Comparison
- Use the provided classifier function to classify images and generate labels.
- Compare the classifier-generated labels to the pet image labels.
- Store the comparison results in a complex data structure (e.g., a dictionary of lists).

### 5. Dog/Not Dog Classification
- Use the `dognames.txt` file to determine whether labels correspond to dog breeds.
- Update the data structure with classifications for “Dog” or “Not Dog.”

### 6. Calculating Results
- Analyze the stored labels and classifications to calculate metrics such as:
  - Percentage of correctly classified dogs.
  - Percentage of correctly classified breeds.
  - Accuracy of the classifier in distinguishing between “Dog” and “Not Dog”.

### 7. Printing Results
- Format and display the results for each CNN architecture.
- Include accuracy metrics and runtime for each architecture.

### 8. Iterating Over Architectures
- Repeat the above tasks for AlexNet, VGG, and ResNet.
- Compare the results to determine the best-performing architecture.

## Provided Files
1. `classifier.py`: Contains the classifier function for image classification.
2. `test_classifier.py`: Demonstrates how to use the classifier function.
3. `pet_images`: Folder containing images for classification.
4. `dognames.txt`: File containing a list of recognized dog breeds.
5. `print_functions_for_lab_checks.py`: Helper functions for debugging and checking intermediate results.

## Expected Outputs
- For each CNN architecture:
  - Percentage of images correctly classified as dogs.
  - Percentage of dog breeds correctly classified.
  - Runtime for the classification process.
- A summary of the best-performing CNN architecture based on accuracy and runtime.

## Installation and Usage
1. Clone the repository:
   ```bash
   git clone <repository_url>
   ```
2. Navigate to the project directory:
   ```bash
   cd Dog_breed_classifier
   ```
3. Install dependencies (if any).
4. Run the program with the desired CNN architecture:
   ```bash
   python check_images.py --dir pet_images --arch resnet --dogfile dognames.txt
   ```

## Results and Analysis
The program outputs classification results for each CNN architecture, including:
- Accuracy of identifying dog images.
- Accuracy of breed classification.
- Total runtime.
- A recommendation of the best architecture based on the results.

## Notes
- Ensure all required files are in the project workspace.
- Use the `print_functions_for_lab_checks.py` program to verify intermediate results.

