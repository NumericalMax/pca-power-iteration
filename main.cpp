#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
#include <limits>
#include <vector>

namespace matrixOperations {
    float *computeColumnMeans(float **input, int rowCount, int colCount) {
        float *means = new float[colCount];
        for (int j = 0; j < colCount; ++j) {
            float sum = 0.0f;
            for (int i = 0; i < rowCount; ++i) {
                sum += input[i][j];
            }
            means[j] = sum / static_cast<float>(rowCount);
        }
        return means;
    }

    void centerData(float **input, int rowCount, int colCount, const float *mean) {
        for (int i = 0; i < rowCount; ++i) {
            for (int j = 0; j < colCount; ++j) {
                input[i][j] -= mean[j];
            }
        }
    }

    float **cloneMatrix(float **input, int rowCount, int colCount) {
        float **copy = new float *[rowCount];
        for (int i = 0; i < rowCount; ++i) {
            copy[i] = new float[colCount];
            for (int j = 0; j < colCount; ++j) {
                copy[i][j] = input[i][j];
            }
        }
        return copy;
    }

    void deleteMatrix(float **matrix, int rowCount) {
        for (int i = 0; i < rowCount; ++i) {
            delete[] matrix[i];
        }
        delete[] matrix;
    }
}

namespace vectorOperations {
    float dotProduct(float *vector1, float *vector2, int length) {
        float result = 0.0f;
        for (int i = 0; i < length; ++i) {
            result += vector1[i] * vector2[i];
        }
        return result;
    }

    float vectorNorm(float *vector, int length) {
        float result = 0.0f;
        for (int i = 0; i < length; ++i) {
            result += vector[i] * vector[i];
        }
        return std::sqrt(result);
    }

    void normalize(float *vector, int length) {
        float norm = vectorNorm(vector, length);
        if (norm == 0.0f) {
            return;
        }
        for (int i = 0; i < length; ++i) {
            vector[i] /= norm;
        }
    }

    float *randomVector(int length) {
        float *vector = new float[length];
        for (int i = 0; i < length; ++i) {
            vector[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX) - 0.5f;
        }
        normalize(vector, length);
        return vector;
    }

    void printVector(float *vec, int length) {
        for (int i = 0; i < length; ++i) {
            std::cout << vec[i] << " ";
        }
        std::cout << std::endl;
    }
}

namespace pca {
    float *getNextPC(float **input, int rowCount, int colCount, float epsilon) {
        float *result = vectorOperations::randomVector(colCount);
        float *next = new float[colCount];

        float difference = std::numeric_limits<float>::max();
        while (difference > epsilon) {
            for (int j = 0; j < colCount; ++j) {
                next[j] = 0.0f;
            }

            for (int i = 0; i < rowCount; ++i) {
                float project = vectorOperations::dotProduct(input[i], result, colCount);
                for (int j = 0; j < colCount; ++j) {
                    next[j] += input[i][j] * project;
                }
            }

            vectorOperations::normalize(next, colCount);
            difference = 0.0f;
            for (int j = 0; j < colCount; ++j) {
                difference += std::abs(next[j] - result[j]);
            }

            for (int j = 0; j < colCount; ++j) {
                result[j] = next[j];
            }
        }

        delete[] next;
        return result;
    }

    float **performPCA(float **input, int rowCount, int colCount, unsigned short dimension) {
        int effectiveDimension = std::min(static_cast<int>(dimension), colCount);
        float **centered = matrixOperations::cloneMatrix(input, rowCount, colCount);

        float *mean = matrixOperations::computeColumnMeans(centered, rowCount, colCount);
        matrixOperations::centerData(centered, rowCount, colCount, mean);

        float **projected = new float *[rowCount];
        for (int i = 0; i < rowCount; ++i) {
            projected[i] = new float[effectiveDimension];
        }

        for (int k = 0; k < effectiveDimension; ++k) {
            float *component = getNextPC(centered, rowCount, colCount, 1e-6f);
            std::cout << "PC " << (k + 1) << ": ";
            vectorOperations::printVector(component, colCount);

            for (int i = 0; i < rowCount; ++i) {
                float projection = vectorOperations::dotProduct(centered[i], component, colCount);
                projected[i][k] = projection;

                for (int j = 0; j < colCount; ++j) {
                    centered[i][j] -= projection * component[j];
                }
            }

            delete[] component;
        }

        delete[] mean;
        matrixOperations::deleteMatrix(centered, rowCount);

        return projected;
    }
}

float **loadData(int rowCount, int colCount) {
    float **matrix = new float *[rowCount];
    for (int i = 0; i < rowCount; ++i) {
        matrix[i] = new float[colCount];
        for (int j = 0; j < colCount; ++j) {
            matrix[i][j] = static_cast<float>((rand() % 100)) / 10.0f;
        }
    }
    return matrix;
}

void writeData(std::string path, float **matrix, int rowCount, int colCount) {
    std::ofstream f(path);
    for (int i = 0; i < rowCount; ++i) {
        for (int j = 0; j < colCount - 1; ++j) {
            f << matrix[i][j] << ",";
        }
        f << matrix[i][colCount - 1] << "\n";
    }
    f.close();
}

int main() {
    srand(static_cast<unsigned int>(time(nullptr)));

    int rowCount = 100;
    int colCount = 3;

    std::cout << "Load data from filesystem..." << std::endl;
    float **input = loadData(rowCount, colCount);
    writeData("file.txt", input, rowCount, colCount);

    std::cout << "Perform PCA..." << std::endl;
    float **output = pca::performPCA(input, rowCount, colCount, 2);

    std::cout << "Write reduced data to filesystem..." << std::endl;
    writeData("output.txt", output, rowCount, 2);

    matrixOperations::deleteMatrix(input, rowCount);
    matrixOperations::deleteMatrix(output, rowCount);

    std::cout << "Done!" << std::endl;
    return 0;
}
