#include <iostream>
#include <fstream>
#include <math.h>



namespace matrixOperations{
    float computeMean(float **input, int rowCount, int colCount){
        float mean = 0.0f;
        for(int i = 0; i < rowCount; ++i){
            for(int j = 0; j < colCount; ++j){
                mean += input[i][j];
            }
        }
        return (mean / (rowCount*colCount));
    }

    void subtractMean(float **input, int rowCount, int colCount, float mean){
        for(int i = 0; i < rowCount; ++i){
            for(int j = 0; j < colCount; ++j){
                input[i][j] = input[i][j] - mean;
            }
        }
    }
}

namespace vectorOperations{
    float eculideanProduct(float *vector1, float *vector2, int length){
        float result = 0.0f;
        for(int i = 0; i < length; ++i){
            result += (vector1[i] * vector2[i]);
        }
        return result;
    }

    float *scalarVectorMultiplication(float *vector, float scalar, int length){
        float *result = new float[length];
        for(int i = 0; i < length; ++i){
            result[i] = scalar * vector[i];
        }
        return result;
    }

    float *scalarVectorDivision(float *vector, float scalar, int length){
        float *result = new float[length];
        for(int i = 0; i < length; ++i){
            result[i] = vector[i] / scalar;
        }
        return result;
    }

    float *addVectors(float *vector1, float *vector2, int length){
        float *result = new float[length];
        for(int i = 0; i < length; ++i){
            result[i] = vector1[i] + vector2[i];
        }
        return result;
    }

    float *subtractVectors(float *vector1, float *vector2, int length){
        float *result = new float[length];
        for(int i = 0; i < length; ++i){
            result[i] = vector1[i] - vector2[i];
        }
        return result;
    }

    float vectorNorm(float *vector, int length){
        float result = 0.0f;
        result = eculideanProduct(vector, vector, length);
        return sqrt(result);
    }

    void printVector(float * vec, int length){
        for(int i = 0; i < length; ++i){
            std::cout << vec[i] << std::endl;
        }
    }
}

namespace pca{
    float * getNextPC(float **input, int rowCount, int colCount, float epsilon){
        float *result = new float[colCount];
        srand (time(NULL));
        for(int i = 0; i < colCount; ++i){
            result[i] = (rand() % 100) / 100.f;
            std::cout << result[i] << std::endl;
        }
        float difference = epsilon + 1.0f;
        while(difference < epsilon){
            float *s = new float[colCount];
            for(int i = 0; i < colCount; ++i){
                s[i] = 0;
            }
            for(int j = 0; j < rowCount; ++j){
                s = vectorOperations::addVectors(s, vectorOperations::scalarVectorMultiplication(input[j], vectorOperations::eculideanProduct(input[j], result, colCount), colCount), colCount);
            }
            difference = std::abs(result - vectorOperations::scalarVectorDivision(s, vectorOperations::vectorNorm(s, colCount), colCount));
        }
        return result;
    }

    float **projectData(float **input, float *component, int rowCount, int colCount){
        for(int k = 0; k  < rowCount; ++k){
            input[k] = vectorOperations::subtractVectors(input[k], vectorOperations::scalarVectorMultiplication(input[k], vectorOperations::eculideanProduct(input[k], component, colCount), colCount), colCount);
        }
        return input;
    }

    // Using the power method
    float** performPCA(float **input, int rowCount, int colCount, unsigned short dimension){
        
        // Step 1: Scale data to the center
        float mean = matrixOperations::computeMean(input, rowCount, colCount);
        matrixOperations::subtractMean(input, rowCount, colCount, mean);
        
        // Step 2: Use power method to calculate the first principal component
        float * pc1 = getNextPC(input, rowCount, colCount, 1E-06);
        std::cout << "first pc: " << std::endl;
        vectorOperations::printVector(pc1, colCount);
        // Step 3: Use power method to calculate the first principal component
        input = projectData(input, pc1, rowCount, colCount);
        
        // Step 4: Recurse
        float * pc2 = getNextPC(input, rowCount, colCount, 1E-06);
        std::cout << "second pc: " << std::endl;
        vectorOperations::printVector(pc2, colCount);
        input = projectData(input, pc2, rowCount, colCount);

        return input;
    }
}

float ** loadData(std::string path, int rowCount, int colCount){
    float **matrix = new float*[rowCount];
    for(int i = 0; i < rowCount; ++i)
        matrix[i] = new float[colCount];
    for(int i = 0; i < rowCount; ++i)
        for(int j = 0; j < colCount; ++j)
            matrix[i][j] = (rand() % 100);
    return matrix;
}

void writeData(std::string path, float **matrix, int rowCount, int colCount){
    std::ofstream f;
    f.open(path);
    for(int i = 0; i < rowCount; ++i){
        for(int j = 0; j < colCount-1; ++j){
            f << matrix[i][j];
            f << ",";
        }
        f << matrix[i][colCount-1];
        f << "\n";
    }
    f.close();
}

int main(){
    
    srand (time(NULL));
    
    // rows are data
    // cols are dims
    int rowCount = 100;
    int colCount = 3;
    
    std::cout << "Load data from filesystem..." << std::endl;
    float **input = loadData("obsolet", rowCount, colCount);
    writeData("file.txt", input, rowCount, colCount);
    std::cout << "Perform PCA..." << std::endl;
    float **output = pca::performPCA(input, rowCount, colCount, 2);
    std::cout << "Write reduced data to filesystem..." << std::endl;
    writeData("output.txt", input, rowCount, colCount);
    std::cout << "Done!" << std::endl;
    return 0;
}
