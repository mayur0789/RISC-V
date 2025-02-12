# VSDSquadron-Mini-Internship
---

Details 
## Task 1

### Step 1: Create the Program file 
Created a file named `sum1ton.c` in the home folder, typed a C code to compute the sum of numbers from 1 to n using a text editor, and saved it.
    ![c_code](https://github.com/user-attachments/assets/bfd3de06-efef-4f54-ac66-f96ebd0ed2ba)

### Step 2: Compile the C code using GCC compiler
The code was compiled using GCC Compiler and the output was printed on the terminal. 
  ```
  $ gcc sum1ton.c
  $ ./a.out
  ```
   ![c_output](https://github.com/user-attachments/assets/3320fbf7-f0c2-4508-b675-3b98b8653a20)

### Step 3: Compile the C code using the RISC-V compiler (-O1) 
The program was compiled with the `riscv64-unknown-elf-gcc` compiler using the `-O1` optimization level.
  ```
  $ riscv64-unknown-elf-gcc -O1 -mabi=lp64 -march=rv64i -o sum1ton.o sum1ton.c
  $ ls -ltr sum1ton.o
  ```
   ![sum1ton o](https://github.com/user-attachments/assets/dd69cb0b-974f-4fb6-9c33-a559b6942f88)

### Step 4: Inspect the Assembly Code for the Main Function (-O1)
Opened a new terminal and entered the below command. The disassembled assembly code for the main function was inspected to observe how the compiler optimized the program with the `-O1` optimization level.
   ```
   $ riscv64-unknown-elf-objdump -d sum1ton.o | less
   ```
   ![main](https://github.com/user-attachments/assets/4319d61e-678e-467c-a5ad-de6c4dce23ad)

### Step 5: Compile the C code using the RISC-V compiler (-Ofast) 
Gone back to the old terminal and again compiled the code  using the `riscv64-unknown-elf-gcc` compiler with the `-Ofast` optimization level to enable aggressive optimizations for performance.
  ```
  $ riscv64-unknown-elf-gcc -Ofast -mabi=lp64 -march=rv64i -o sum1ton.o sum1ton.c
  ```
  ![ofast](https://github.com/user-attachments/assets/4c9d90d4-0593-4c65-8364-8fc473597969)

### Step 6: Inspect the Assembly Code for the Main Function (-Ofast)
Again gone to the the new terminal and entered the below command.The disassembled assembly code for the main function was inspected again to analyze the effects of the `-Ofast` optimization level.
  ```
  $ riscv64-unknown-elf-objdump -d sum1ton.o | less
  ```
  ![main2](https://github.com/user-attachments/assets/5d08ed72-ee14-458f-8a51-3c0a881a4369)

---
## Task 2

### Branch Prediction Using a Neural Network

#### Introduction

Branch prediction is a critical component of modern CPUs to improve instruction pipeline efficiency. By predicting the outcome of a branch instruction (taken or not taken), processors can minimize delays. This project demonstrates a neural network-based approach for branch prediction, where a simple feedforward neural network is trained to predict branch behavior based on historical patterns.

---

#### Data Set

The table below represents the training data used for the neural network. Each row corresponds to a historical branch pattern (input) and the associated branch outcome (target).

| **Input (Branch History)** | **Target (Branch Outcome)** |
|-----------------------------|-----------------------------|
| `{1, 0, 1, 1, 0}`           | `1` (Taken)                |
| `{0, 1, 1, 0, 1}`           | `0` (Not Taken)            |
| `{1, 1, 0, 1, 0}`           | `1` (Taken)                |
| `{0, 0, 0, 1, 1}`           | `0` (Not Taken)            |

---

#### Specification

- **Neural Network Architecture**: 
  - Input layer: 5 neurons (representing 5 historical branch outcomes).
  - Hidden layer: 3 neurons (with activation function).
  - Output layer: 1 neuron (producing a value between 0 and 1).
  
- **Objective**: Predict whether a branch will be taken (output near 1) or not taken (output near 0).

---

#### implementation

Below is the implementation of the neural network-based branch predictor in C:

```c
#include <stdio.h>
#include <stdlib.h>

#define INPUT_SIZE 5
#define HIDDEN_SIZE 3
#define OUTPUT_SIZE 1
#define EPOCHS 10000
#define LEARNING_RATE 0.1

// Custom approximation of the exponential function
double custom_exp(double x) {
    double result = 1.0;
    double term = 1.0;
    for (int i = 1; i < 20; i++) {  // Use 20 terms for better approximation
        term *= x / i;
        result += term;
    }
    return result;
}

// Activation function (Sigmoid)
double sigmoid(double x) {
    return 1.0 / (1.0 + custom_exp(-x));
}

// Derivative of Sigmoid
double sigmoid_derivative(double x) {
    return x * (1.0 - x);
}

// Initialize weights and biases randomly
void initialize(double weights_in[HIDDEN_SIZE][INPUT_SIZE], double weights_out[OUTPUT_SIZE][HIDDEN_SIZE], 
                double bias_hidden[HIDDEN_SIZE], double bias_output[OUTPUT_SIZE]) {
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        for (int j = 0; j < INPUT_SIZE; j++) {
            weights_in[i][j] = (double)rand() / RAND_MAX;
        }
        bias_hidden[i] = (double)rand() / RAND_MAX;
    }

    for (int i = 0; i < OUTPUT_SIZE; i++) {
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            weights_out[i][j] = (double)rand() / RAND_MAX;
        }
        bias_output[i] = (double)rand() / RAND_MAX;
    }
}

// Forward pass
void forward(double input[INPUT_SIZE], double weights_in[HIDDEN_SIZE][INPUT_SIZE], double bias_hidden[HIDDEN_SIZE],
             double hidden[HIDDEN_SIZE], double weights_out[OUTPUT_SIZE][HIDDEN_SIZE], 
             double bias_output[OUTPUT_SIZE], double output[OUTPUT_SIZE]) {
    // Hidden layer computation
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        hidden[i] = bias_hidden[i];
        for (int j = 0; j < INPUT_SIZE; j++) {
            hidden[i] += input[j] * weights_in[i][j];
        }
        hidden[i] = sigmoid(hidden[i]);
    }

    // Output layer computation
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        output[i] = bias_output[i];
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            output[i] += hidden[j] * weights_out[i][j];
        }
        output[i] = sigmoid(output[i]);
    }
}

// Backward pass (Training)
void backward(double input[INPUT_SIZE], double weights_in[HIDDEN_SIZE][INPUT_SIZE], double bias_hidden[HIDDEN_SIZE],
              double hidden[HIDDEN_SIZE], double weights_out[OUTPUT_SIZE][HIDDEN_SIZE], double bias_output[OUTPUT_SIZE], 
              double output[OUTPUT_SIZE], double target[OUTPUT_SIZE]) {
    double output_error[OUTPUT_SIZE], hidden_error[HIDDEN_SIZE];

    // Calculate output error
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        output_error[i] = (target[i] - output[i]) * sigmoid_derivative(output[i]);
    }

    // Calculate hidden layer error
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        hidden_error[i] = 0.0;
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            hidden_error[i] += output_error[j] * weights_out[j][i];
        }
        hidden_error[i] *= sigmoid_derivative(hidden[i]);
    }

    // Update weights and biases (Output layer)
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            weights_out[i][j] += LEARNING_RATE * output_error[i] * hidden[j];
        }
        bias_output[i] += LEARNING_RATE * output_error[i];
    }

    // Update weights and biases (Input to Hidden layer)
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        for (int j = 0; j < INPUT_SIZE; j++) {
            weights_in[i][j] += LEARNING_RATE * hidden_error[i] * input[j];
        }
        bias_hidden[i] += LEARNING_RATE * hidden_error[i];
    }
}

// Main function
int main() {
    // Define network parameters
    double weights_in[HIDDEN_SIZE][INPUT_SIZE], weights_out[OUTPUT_SIZE][HIDDEN_SIZE];
    double bias_hidden[HIDDEN_SIZE], bias_output[OUTPUT_SIZE];
    double hidden[HIDDEN_SIZE], output[OUTPUT_SIZE];

    // Initialize network
    initialize(weights_in, weights_out, bias_hidden, bias_output);

    // Define training data
    double inputs[4][INPUT_SIZE] = {
        {1, 0, 1, 1, 0},  // Example historical branch patterns
        {0, 1, 1, 0, 1},
        {1, 1, 0, 1, 0},
        {0, 0, 0, 1, 1}
    };
    double targets[4][OUTPUT_SIZE] = {
        {1},  // Branch taken
        {0},  // Branch not taken
        {1},  // Branch taken
        {0}   // Branch not taken
    };

    // Train the network
    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        for (int i = 0; i < 4; i++) {
            forward(inputs[i], weights_in, bias_hidden, hidden, weights_out, bias_output, output);
            backward(inputs[i], weights_in, bias_hidden, hidden, weights_out, bias_output, output, targets[i]);
        }
    }

    // Test the network
    double test_input[INPUT_SIZE] = {0, 1, 0, 1, 1};
    //double test_input[INPUT_SIZE] ={1, 0, 1, 1, 0};
    forward(test_input, weights_in, bias_hidden, hidden, weights_out, bias_output, output);
    printf("Prediction for input {0, 1, 0, 1, 1}: %.2f\n", output[0]);
    //printf("Prediction for input {1, 0, 1, 1, 0}: %.2f\n", output[0]);
    //printf("Prediction for input : %.2f\n", output[0]);
    printf("%s\n", output[0] >= 0.5 ? "Branch Taken" : "Branch Not Taken");
    
    return 0;
}
```

---

####  Compile the C code using GCC compiler
  ```
  $ gcc bpnn.c
  $ ./a.out
  ```
![task1pic1](https://github.com/user-attachments/assets/e29b6be4-fcc6-43b6-873a-242241e0b11a)

---

####  Compile the C code using the RISC-V compiler ( -O1)
  ```
  $ riscv64-unknown-elf-gcc -O1 -mabi=lp64 -march=rv64i -o bpnn.o bpnn.c
  $ ls -ltr bpnn.o
  ```
![task2pic2](https://github.com/user-attachments/assets/dd4e62ac-7fee-4034-aeb3-0a3d69c166f9)

---

####  Inspect the Assembly Code for the Main Function (-O1)
   ```
   $ riscv64-unknown-elf-objdump -d bpnn.o | less
   ```
![task2pic3](https://github.com/user-attachments/assets/3b9cdfc3-c2bb-4891-9cd7-66ae7b0a06c2)

---

####  Compile the C code using the RISC-V compiler (-Ofast) 
  ```
  $ riscv64-unknown-elf-gcc -Ofast -mabi=lp64 -march=rv64i -o bpnn.o bpnn.c
  ```
![task2pic4](https://github.com/user-attachments/assets/44512c5a-b8b8-4c19-8bf3-d862352252be)

---

####  Inspect the Assembly Code for the Main Function (-Ofast)
  ```
  $ riscv64-unknown-elf-objdump -d bpnn.o | less
  ```
![task2pic5](https://github.com/user-attachments/assets/fc8e6b90-40fe-4d98-bd46-318af0838088)

---

####  Observe the ouput given by RISC V Compiler
  ```
  $ spike pk bpnn.o
  ```
![task2pic6](https://github.com/user-attachments/assets/0786e303-e551-4deb-af24-81f7e83c8e4c)

---

####  Inspect the Stack pointer in the  Assembly Code of the Main Function (-Ofast)
  ```
  $ riscv64-unknown-elf-objdump -d bpnn.o | less
  ```
![task2pic7](https://github.com/user-attachments/assets/9d91196d-f9fa-4fbb-bb1f-9347ad47a02e)

---

####  Debug the C code compiled by RISC V Compiler using spike command by inspecting the stack pointer
  ```
  $ spike -d pk bpnn.o
  ```
![task2pic8](https://github.com/user-attachments/assets/20470995-d3b2-415b-88a7-8e810bb4af90)

---

## Task3

## RISC-V ISA

The RV32I instruction set architecture (ISA) in RISC-V is made up of several types of instructions, which can be classified based on their functionalities and encoding formats. Below is a summary and classification of the various instructions, their groupings by bits, and the combinations defined for each function in the recent RV32I specification (May 2024).

### Types of Instructions in RV32I:
- **R-Type Instructions**
- **I-Type Instructions**
- **S-Type Instructions**
- **B-Type Instructions**
- **U-Type Instructions**
- **J-Type Instructions**

![WhatsApp Image 2024-12-06 at 15 45 43_7140aa82](https://github.com/user-attachments/assets/02cee157-e459-47ee-a6b9-d199ba388c78)

Ref: The RISC-V Instruction Set Manual Volume I | © RISC-V

---

## R-Type Instruction Format

| **Bit**  | 31-25      | 24-20     | 19-15     | 14-12     | 11-7      | 6-0       |
|----------|------------|-----------|-----------|-----------|-----------|-----------|
| **Field**| funct7     | rs2       | rs1       | funct3    | rd        | opcode    |
| **Description** | Function code (extended operation) | Source Register 2  | Source Register 1  | Function code (defines operation) | Destination Register  | Operation code  |

![WhatsApp Image 2024-12-06 at 16 52 58_88180d42](https://github.com/user-attachments/assets/62d18328-66d8-461d-91f5-af4d9991ec49)

---

## I-Type Instruction Format

| **Bit**  | 31-20      | 19-15     | 14-12     | 11-7      | 6-0       |
|----------|------------|-----------|-----------|-----------|-----------|
| **Field**| imm[11:0]        | rs1       | funct3    | rd        | opcode    |
| **Description** | Immediate bits [11:0] | Source Register 1 | Function code (defines operation) | Destination Register | Operation code |

![WhatsApp Image 2024-12-06 at 16 57 30_40e1abd6](https://github.com/user-attachments/assets/3e567cf2-00eb-4711-b30e-c760c5402fc4)

---

## S-Type Instruction Format

| **Bit**  | 31-25      | 24-20     | 19-15     | 14-12     | 11-7      | 6-0       |
|----------|------------|-----------|-----------|-----------|-----------|-----------|
| **Field**| imm[11:5]  | rs2       | rs1       | funct3    | imm[4:0]  | opcode    |
| **Description** | Immediate bits [11:5]  | Source Register 2 | Source Register 1 | Function code (defines operation) | Immediate bits [4:0]  | Operation code |

![WhatsApp Image 2024-12-06 at 17 45 13_a0fbc9ab](https://github.com/user-attachments/assets/c330e2d7-7bb2-4de2-adf2-5df40744aa8a)

---

## B-Type Instruction Format

| **Bit**  |    31     | 30-25     | 24-20     | 19-15     | 14-12      | 11-8       |    7     | 6-0       |
|----------|------------|-----------|-----------|-----------|-----------|-----------|---------|---------|
| **Field**| imm[12]    | imm[10:5]       | rs2       | rs1    | funct3 | imm[4:1]  | imm[11] | opcode |
| **Description** | Immediate bit [12] | Immediate bits [10:5] |  Source Register 2 | Source Register 1 | Function code (defines operation) | Immediate bits [4:1] | Immediate bit [11] | Operation code |

![WhatsApp Image 2024-12-06 at 17 07 25_e8674253](https://github.com/user-attachments/assets/4c8c579e-7866-460b-bf88-f5c4b0ed3e33)


---

## U-Type Instruction Format

| **Bit**  | 31-12      | 11-7      | 6-0       |
|----------|------------|-----------|-----------|
| **Field**| imm[31-12] | rd        | opcode    |
| **Description** | Immediate bits [31-12] | Destination Register | Operation code |

![WhatsApp Image 2024-12-06 at 17 54 28_8d65a2a1](https://github.com/user-attachments/assets/27904b7f-6048-4e82-855d-d7ed5ec24260)

---

## J-Type Instruction Format

| **Bit**  | 31         | 30-21     | 20        | 19-12      |  11-7     |  6-0      |
|----------|------------|-----------|-----------|------------|-----------|-----------|
| **Field**| imm[20]    | imm[10:1] |  imm[11]  | imm[19:12] | rd        | opcode    |
| **Description** | Immediate bit [20] | Immediate bits [10:1] |Immediate bit [11] |Immediate bits [19:12] | Destination Register | Operation code |

![WhatsApp Image 2024-12-06 at 17 54 28_19a3ba51](https://github.com/user-attachments/assets/7a90011d-b24e-452c-9063-2a1d51181c5d)

---


## Encoding Branch Prediction Using a Neural Network Application Instructions

### **1. addi sp, sp, -496**

For the instruction `addi sp, sp, -496`:

| **Bit**       | **31-20**       | **19-15** | **14-12** | **11-7**  | **6-0**   |
|---------------|-----------------|-----------|-----------|-----------|-----------|
| **Field**     | imm[11:0]       | rs1 (sp)  | funct3    | rd (sp)   | opcode    |
| **Value**     | 111111111000    | 00010     | 000       | 00010     | 0010011   |

#### **Explanation of Fields:**
- **imm[11:0]**: `-496` is represented as `111111111000` (12-bit sign-extended immediate).  
- **rs1**: `sp` is register `x2`, encoded as `00010`.  
- **funct3**: `000` indicates an `addi` operation.  
- **rd**: `sp` is register `x2`, encoded as `00010`.  
- **opcode**: `0010011` is the opcode for immediate arithmetic instructions.

#### **32-bit Representation:**
`111111111000 00010 000 00010 0010011`  
**Hexadecimal Representation:** `0xFFF10213`

---

### **2. sd ra, 488(sp)**

For the instruction `sd ra, 488(sp)`:

| **Bit**       | **31-25**       | **24-20** | **19-15** | **14-12** | **11-7**  | **6-0**   |
|---------------|-----------------|-----------|-----------|-----------|-----------|-----------|
| **Field**     | imm[11:5]       | rs2 (ra)  | rs1 (sp)  | funct3    | imm[4:0]  | opcode    |
| **Value**     | 0111100         | 00001     | 00010     | 011       | 01000     | 0100011   |

#### **Explanation of Fields:**
- **imm[11:5]**: Upper 7 bits of `488` (`1111000` in binary), encoded as `0111100`.  
- **rs2**: `ra` is register `x1`, encoded as `00001`.  
- **rs1**: `sp` is register `x2`, encoded as `00010`.  
- **funct3**: `011` indicates a `sd` operation.  
- **imm[4:0]**: Lower 5 bits of `488` (`1111000` in binary), encoded as `01000`.  
- **opcode**: `0100011` is the opcode for store instructions.

#### **32-bit Representation:**
`0111100 00001 00010 011 01000 0100011`  
**Hexadecimal Representation:** `0x3E825023`

---

### **3. sd s0, 480(sp)**

For the instruction `sd s0, 480(sp)`:

| **Bit**       | **31-25**       | **24-20** | **19-15** | **14-12** | **11-7**  | **6-0**   |
|---------------|-----------------|-----------|-----------|-----------|-----------|-----------|
| **Field**     | imm[11:5]       | rs2 (s0)  | rs1 (sp)  | funct3    | imm[4:0]  | opcode    |
| **Value**     | 0111000         | 00000     | 00010     | 011       | 00000     | 0100011   |

#### **Explanation of Fields:**
- **imm[11:5]**: Upper 7 bits of `480` (`1111000` in binary), encoded as `0111000`.  
- **rs2**: `s0` is register `x8`, encoded as `01000`.  
- **rs1**: `sp` is register `x2`, encoded as `00010`.  
- **funct3**: `011` indicates a `sd` operation.  
- **imm[4:0]**: Lower 5 bits of `480` (`1110000` in binary), encoded as `00000`.  
- **opcode**: `0100011` is the opcode for store instructions.

#### **32-bit Representation:**
`0111000 01000 00010 011 00000 0100011`  
**Hexadecimal Representation:** `0x3A802023`

---

### **4. sd s1, 472(sp)**

For the instruction `sd s1, 472(sp)`:

| **Bit**       | **31-25**       | **24-20** | **19-15** | **14-12** | **11-7**  | **6-0**   |
|---------------|-----------------|-----------|-----------|-----------|-----------|-----------|
| **Field**     | imm[11:5]       | rs2 (s1)  | rs1 (sp)  | funct3    | imm[4:0]  | opcode    |
| **Value**     | 0111000         | 00001     | 00010     | 011       | 00000     | 0100011   |

#### **Explanation of Fields:**
- **imm[11:5]**: Upper 7 bits of `472` (`1110100` in binary), encoded as `0111000`.  
- **rs2**: `s1` is register `x9`, encoded as `01001`.  
- **rs1**: `sp` is register `x2`, encoded as `00010`.  
- **funct3**: `011` indicates a `sd` operation.  
- **imm[4:0]**: Lower 5 bits of `472` (`1110000` in binary), encoded as `00000`.  
- **opcode**: `0100011` is the opcode for store instructions.

#### **32-bit Representation:**
`0111000 01001 00010 011 00000 0100011`  
**Hexadecimal Representation:** `0x3A902023`

---

### **5. sd s2, 464(sp)**

For the instruction `sd s2, 464(sp)`:

| **Bit**       | **31-25**       | **24-20** | **19-15** | **14-12** | **11-7**  | **6-0**   |
|---------------|-----------------|-----------|-----------|-----------|-----------|-----------|
| **Field**     | imm[11:5]       | rs2 (s2)  | rs1 (sp)  | funct3    | imm[4:0]  | opcode    |
| **Value**     | 0111000         | 00010     | 00010     | 011       | 00000     | 0100011   |

#### **Explanation of Fields:**
- **imm[11:5]**: Upper 7 bits of `464` (`1110100` in binary), encoded as `0111000`.  
- **rs2**: `s2` is register `x18`, encoded as `10010`.  
- **rs1**: `sp` is register `x2`, encoded as `00010`.  
- **funct3**: `011` indicates a `sd` operation.  
- **imm[4:0]**: Lower 5 bits of `464` (`1110000` in binary), encoded as `00000`.  
- **opcode**: `0100011` is the opcode for store instructions.

#### **32-bit Representation:**
`0111000 10010 00010 011 00000 0100011`  
**Hexadecimal Representation:** `0x3A928023`

---

### **6. sd s3, 456(sp)**

For the instruction `sd s3, 456(sp)`:

| **Bit**       | **31-25**       | **24-20** | **19-15** | **14-12** | **11-7**  | **6-0**   |
|---------------|-----------------|-----------|-----------|-----------|-----------|-----------|
| **Field**     | imm[11:5]       | rs2 (s3)  | rs1 (sp)  | funct3    | imm[4:0]  | opcode    |
| **Value**     | 0111000         | 00011     | 00010     | 011       | 00000     | 0100011   |

#### **Explanation of Fields:**
- **imm[11:5]**: Upper 7 bits of `456` (`1110010` in binary), encoded as `0111000`.  
- **rs2**: `s3` is register `x19`, encoded as `10011`.  
- **rs1**: `sp` is register `x2`, encoded as `00010`.  
- **funct3**: `011` indicates a `sd` operation.  
- **imm[4:0]**: Lower 5 bits of `456` (`1110010` in binary), encoded as `00000`.  
- **opcode**: `0100011` is the opcode for store instructions.

#### **32-bit Representation:**
`0111000 10011 00010 011 00000 0100011`  
**Hexadecimal Representation:** `0x3A938023`

---

### **7. addi a3, sp, 272**

For the instruction `addi a3, sp, 272`:

| **Bit**       | **31-20**       | **19-15** | **14-12** | **11-7**  | **6-0**   |
|---------------|-----------------|-----------|-----------|-----------|-----------|
| **Field**     | imm[31:20]      | rs1 (sp)  | funct3    | rd (a3)   | opcode    |
| **Value**     | 000000010000     | 00010     | 000       | 01100     | 0010011   |

#### **Explanation of Fields:**
- **imm[31:20]**: Upper 12 bits of `272` (`000000010000` in binary).  
- **rs1**: `sp` is register `x2`, encoded as `00010`.  
- **funct3**: `000` indicates an `addi` operation.  
- **rd**: `a3` is register `x15`, encoded as `01100`.  
- **opcode**: `0010011` is the opcode for `addi` instructions.

#### **32-bit Representation:**
`000000010000 00010 000 01100 0010011`  
**Hexadecimal Representation:** `0x00430313`

---

### **8. addi a2, sp, 280**

For the instruction `addi a2, sp, 280`:

| **Bit**       | **31-20**       | **19-15** | **14-12** | **11-7**  | **6-0**   |
|---------------|-----------------|-----------|-----------|-----------|-----------|
| **Field**     | imm[31:20]      | rs1 (sp)  | funct3    | rd (a2)   | opcode    |
| **Value**     | 000000010100     | 00010     | 000       | 01010     | 0010011   |

#### **Explanation of Fields:**
- **imm[31:20]**: Upper 12 bits of `280` (`000000010100` in binary).  
- **rs1**: `sp` is register `x2`, encoded as `00010`.  
- **funct3**: `000` indicates an `addi` operation.  
- **rd**: `a2` is register `x10`, encoded as `01010`.  
- **opcode**: `0010011` is the opcode for `addi` instructions.

#### **32-bit Representation:**
`000000010100 00010 000 01010 0010011`  
**Hexadecimal Representation:** `0x00530293`

---

### **9. addi a1, sp, 304**

For the instruction `addi a1, sp, 304`:

| **Bit**       | **31-20**       | **19-15** | **14-12** | **11-7**  | **6-0**   |
|---------------|-----------------|-----------|-----------|-----------|-----------|
| **Field**     | imm[31:20]      | rs1 (sp)  | funct3    | rd (a1)   | opcode    |
| **Value**     | 000000010110     | 00010     | 000       | 01001     | 0010011   |

#### **Explanation of Fields:**
- **imm[31:20]**: Upper 12 bits of `304` (`000000010110` in binary).  
- **rs1**: `sp` is register `x2`, encoded as `00010`.  
- **funct3**: `000` indicates an `addi` operation.  
- **rd**: `a1` is register `x11`, encoded as `01001`.  
- **opcode**: `0010011` is the opcode for `addi` instructions.

#### **32-bit Representation:**
`000000010110 00010 000 01001 0010011`  
**Hexadecimal Representation:** `0x00530293`

---

### **10. addi a0, sp, 328**

For the instruction `addi a0, sp, 328`:

| **Bit**       | **31-20**       | **19-15** | **14-12** | **11-7**  | **6-0**   |
|---------------|-----------------|-----------|-----------|-----------|-----------|
| **Field**     | imm[31:20]      | rs1 (sp)  | funct3    | rd (a0)   | opcode    |
| **Value**     | 000000011000     | 00010     | 000       | 01010     | 0010011   |

#### **Explanation of Fields:**
- **imm[31:20]**: Upper 12 bits of `328` (`000000011000` in binary).  
- **rs1**: `sp` is register `x2`, encoded as `00010`.  
- **funct3**: `000` indicates an `addi` operation.  
- **rd**: `a0` is register `x10`, encoded as `01010`.  
- **opcode**: `0010011` is the opcode for `addi` instructions.

#### **32-bit Representation:**
`000000011000 00010 000 01010 0010011`  
**Hexadecimal Representation:** `0x00C30293`

---

### **11. jal ra, 10284 <initialize>**

For the instruction `jal ra, 10284 <initialize>`:

| **Bit**       | **31**   | **30-21**     | **20**   | **19-12**     | **11-7** | **6-0**   |
|---------------|----------|---------------|----------|---------------|----------|-----------|
| **Field**     | imm[20]  | imm[10:1]     | imm[11]  | imm[19:12]    | rd (ra)  | opcode    |
| **Value**     | 0        | 0000001011    | 1        | 000010010100  | 00000    | 1101111   |

#### **Explanation of Fields:**
- **imm[20]**: Immediate bit 20 is `0` (from the offset `10284` in binary).
- **imm[10:1]**: Bits 10 to 1 of the immediate value `10284`, which are `0000001011`.
- **imm[11]**: Bit 11 of the immediate value `10284` is `1`.
- **imm[19:12]**: Bits 19 to 12 of the immediate value `10284`, which are `000010010100`.
- **rd**: The destination register `ra` is `x1`, encoded as `00000` (since `ra` is used as the link register in the `jal` instruction).
- **opcode**: `1101111` is the opcode for the `jal` instruction.

#### **32-bit Representation:**
`0 0000001011 1 000010010100 00000 1101111`  
**Hexadecimal Representation:** `0x000ACF6F`

---

### **12. lui a5, 0x22**

For the instruction `lui a5, 0x22`:

| **Bit**       | **31-12**   | **11-7**   | **6-0**   |
|---------------|-------------|------------|-----------|
| **Field**     | imm[31:12]  | rd (a5)    | opcode    |
| **Value**     | 000000000010 | 00101      | 0110111   |

#### **Explanation of Fields:**
- **imm[31:12]**: The immediate value `0x22` is extended to fit the 20-bit field. This results in `000000000010`.
- **rd**: The destination register `a5` corresponds to register `x15`, encoded as `00101`.
- **opcode**: `0110111` is the opcode for the `lui` instruction.

#### **32-bit Representation:**
`000000000010 00101 0110111`  
**Hexadecimal Representation:** `0x00022037`

---

### **13. addi a5, a5, 960**

For the instruction `addi a5, a5, 960`:

| **Bit**       | **31-20**      | **19-15** | **14-12** | **11-7** | **6-0**   |
|---------------|----------------|-----------|-----------|----------|-----------|
| **Field**     | imm[31:20]     | rs1 (a5)  | funct3    | rd (a5)  | opcode    |
| **Value**     | 000000111100    | 00101     | 000       | 00101    | 0010011   |

#### **Explanation of Fields:**
- **imm[31:20]**: The immediate value `960` is encoded in binary as `000000111100`.
- **rs1**: The source register `a5` corresponds to register `x15`, encoded as `00101`.
- **funct3**: The function code `000` indicates an addition operation.
- **rd**: The destination register is `a5`, encoded as `00101`.
- **opcode**: `0010011` is the opcode for the `addi` instruction.

#### **32-bit Representation:**
`000000111100 00101 000 00101 0010011`  
**Hexadecimal Representation:** `0x0f502933`

---

### **14. addi a4, sp, 80**

For the instruction `addi a4, sp, 80`:

| **Bit**       | **31-20**      | **19-15** | **14-12** | **11-7** | **6-0**   |
|---------------|----------------|-----------|-----------|----------|-----------|
| **Field**     | imm[31:20]     | rs1 (sp)  | funct3    | rd (a4)  | opcode    |
| **Value**     | 000000001010    | 00010     | 000       | 00100    | 0010011   |

#### **Explanation of Fields:**
- **imm[31:20]**: The immediate value `80` is encoded in binary as `000000001010`.
- **rs1**: The source register `sp` corresponds to register `x2`, encoded as `00010`.
- **funct3**: The function code `000` indicates an addition operation.
- **rd**: The destination register is `a4`, encoded as `00100`.
- **opcode**: `0010011` is the opcode for the `addi` instruction.

#### **32-bit Representation:**
`000000001010 00010 000 00100 0010011`  
**Hexadecimal Representation:** `0x00a10113`

---

### **15. addi a6, a5, 160**

For the instruction `addi a6, a5, 160`:

| **Bit**       | **31-20**      | **19-15** | **14-12** | **11-7** | **6-0**   |
|---------------|----------------|-----------|-----------|----------|-----------|
| **Field**     | imm[31:20]     | rs1 (a5)  | funct3    | rd (a6)  | opcode    |
| **Value**     | 000000010100    | 00101     | 000       | 00110    | 0010011   |

#### **Explanation of Fields:**
- **imm[31:20]**: The immediate value `160` is encoded in binary as `000000010100`.
- **rs1**: The source register `a5` corresponds to register `x5`, encoded as `00101`.
- **funct3**: The function code `000` indicates an addition operation.
- **rd**: The destination register is `a6`, encoded as `00110`.
- **opcode**: `0010011` is the opcode for the `addi` instruction.

#### **32-bit Representation:**
`000000010100 00101 000 00110 0010011`  
**Hexadecimal Representation:** `0x00530293`

---

### **16. ld a0, 0(a5)**

For the instruction `ld a0, 0(a5)`:

| **Bit**       | **31-20**      | **19-15** | **14-12** | **11-7** | **6-0**   |
|---------------|----------------|-----------|-----------|----------|-----------|
| **Field**     | imm[31:20]     | rs1 (a5)  | funct3    | rd (a0)  | opcode    |
| **Value**     | 000000000000    | 00101     | 010       | 00000    | 0000011   |

#### **Explanation of Fields:**
- **imm[31:20]**: The immediate value `0` is encoded in binary as `000000000000`.
- **rs1**: The source register `a5` corresponds to register `x5`, encoded as `00101`.
- **funct3**: The function code `010` indicates a load instruction (`LD`).
- **rd**: The destination register is `a0`, encoded as `00000`.
- **opcode**: `0000011` is the opcode for load instructions.

#### **32-bit Representation:**
`000000000000 00101 010 00000 0000011`  
**Hexadecimal Representation:** `0x0002a003`

---

### **17. ld a1, 8(a5)**

For the instruction `ld a1, 8(a5)`:

| **Bit**       | **31-20**      | **19-15** | **14-12** | **11-7** | **6-0**   |
|---------------|----------------|-----------|-----------|----------|-----------|
| **Field**     | imm[31:20]     | rs1 (a5)  | funct3    | rd (a1)  | opcode    |
| **Value**     | 000000000010    | 00101     | 010       | 00001    | 0000011   |

#### **Explanation of Fields:**
- **imm[31:20]**: The immediate value `8` is encoded in binary as `000000000010`.
- **rs1**: The source register `a5` corresponds to register `x5`, encoded as `00101`.
- **funct3**: The function code `010` indicates a load instruction (`LD`).
- **rd**: The destination register is `a1`, encoded as `00001`.
- **opcode**: `0000011` is the opcode for load instructions.

#### **32-bit Representation:**
`000000000010 00101 010 00001 0000011`  
**Hexadecimal Representation:** `0x0002a023`

---

### **18. ld a2, 16(a5)**

For the instruction `ld a2, 16(a5)`:

| **Bit**       | **31-20**      | **19-15** | **14-12** | **11-7** | **6-0**   |
|---------------|----------------|-----------|-----------|----------|-----------|
| **Field**     | imm[31:20]     | rs1 (a5)  | funct3    | rd (a2)  | opcode    |
| **Value**     | 000000000100    | 00101     | 010       | 00010    | 0000011   |

#### **Explanation of Fields:**
- **imm[31:20]**: The immediate value `16` is encoded in binary as `000000000100`.
- **rs1**: The source register `a5` corresponds to register `x5`, encoded as `00101`.
- **funct3**: The function code `010` indicates a load instruction (`LD`).
- **rd**: The destination register is `a2`, encoded as `00010`.
- **opcode**: `0000011` is the opcode for load instructions.

#### **32-bit Representation:**
`000000000100 00101 010 00010 0000011`  
**Hexadecimal Representation:** `0x0002a023`

---

### **19. ld a3, 24(a5)**

For the instruction `ld a3, 24(a5)`:

| **Bit**       | **31-20**      | **19-15** | **14-12** | **11-7** | **6-0**   |
|---------------|----------------|-----------|-----------|----------|-----------|
| **Field**     | imm[31:20]     | rs1 (a5)  | funct3    | rd (a3)  | opcode    |
| **Value**     | 000000000110    | 00101     | 010       | 00011    | 0000011   |

#### **Explanation of Fields:**
- **imm[31:20]**: The immediate value `24` is encoded in binary as `000000000110`.
- **rs1**: The source register `a5` corresponds to register `x5`, encoded as `00101`.
- **funct3**: The function code `010` indicates a load instruction (`LD`).
- **rd**: The destination register is `a3`, encoded as `00011`.
- **opcode**: `0000011` is the opcode for load instructions.

#### **32-bit Representation:**
`000000000110 00101 010 00011 0000011`  
**Hexadecimal Representation:** `0x0002a023`

---

### **20. sd a0, 0(a4)**

For the instruction `sd a0, 0(a4)`:

| **Bit**       | **31-25**      | **24-20** | **19-15** | **14-12** | **11-7** | **6-0**   |
|---------------|----------------|-----------|-----------|-----------|----------|-----------|
| **Field**     | imm[31:25]     | rs2 (a0)  | rs1 (a4)  | funct3    | imm[4:0] | opcode    |
| **Value**     | 0000000        | 00000     | 00100     | 011       | 00000    | 0100011   |

#### **Explanation of Fields:**
- **imm[31:25]**: The immediate value `0` is encoded as `0000000` for the higher 7 bits.
- **rs2**: The source register `a0` corresponds to register `x10`, encoded as `00000`.
- **rs1**: The base register `a4` corresponds to register `x4`, encoded as `00100`.
- **funct3**: The function code `011` indicates a store double word instruction (`SD`).
- **imm[4:0]**: The immediate value `0` is encoded in binary as `00000` for the lower 5 bits.
- **opcode**: `0100011` is the opcode for store instructions.

#### **32-bit Representation:**
`0000000 00000 00100 011 00000 0100011`  
**Hexadecimal Representation:** `0x00024223`

---


### **21. sd a1, 8(a4)**

For the instruction `sd a1, 8(a4)`:

| **Bit**       | **31-25**      | **24-20** | **19-15** | **14-12** | **11-7** | **6-0**   |
|---------------|----------------|-----------|-----------|-----------|----------|-----------|
| **Field**     | imm[31:25]     | rs2 (a1)  | rs1 (a4)  | funct3    | imm[4:0] | opcode    |
| **Value**     | 0000000        | 00011     | 00100     | 011       | 01000    | 0100011   |

#### **Explanation of Fields:**
- **imm[31:25]**: The immediate value `8` is encoded as `0000000` for the higher 7 bits.
- **rs2**: The source register `a1` corresponds to register `x11`, encoded as `00011`.
- **rs1**: The base register `a4` corresponds to register `x4`, encoded as `00100`.
- **funct3**: The function code `011` indicates a store double word instruction (`SD`).
- **imm[4:0]**: The immediate value `8` is encoded in binary as `01000` for the lower 5 bits.
- **opcode**: `0100011` is the opcode for store instructions.

#### **32-bit Representation:**
`0000000 00011 00100 011 01000 0100011`  
**Hexadecimal Representation:** `0x00324223`

---

## Task 4

### Functional simulation of the given design code of pipelined RISC V 32I Processor

#### Initial steps

- Both design and testbench codes are saved in a separate folder
- To simulate the verilog code
```
$ iverilog -o iiitb_rv32i iiitb_rv32i.v iiitb_rv32i_tb.v
$ ./iiitb_rv32i
```

- To open the dumped vcd file
```
$ gtkwave iiitb_rv32i.vcd
```

![task4pic1](https://github.com/user-attachments/assets/c1dba9ef-1fdd-4c76-8024-1f3dc78382f5)

---

### Analayzing the hex code given by the designer in the instruction memory

| **Program**                  | **Hex Code**     | **Assembly Code**       |
|------------------------------|------------------|--------------------------|
| MEM[0] <= 32'h02208300;      | 02208300         | add r6, r1, r2          |
| MEM[1] <= 32'h02209380;      | 02209380         | sub r7, r1, r2          |
| MEM[2] <= 32'h0230a400;      | 0230a400         | and r8, r1, r3          |
| MEM[3] <= 32'h02513480;      | 02513480         | or r9, r2, r5           |
| MEM[4] <= 32'h0240c500;      | 0240c500         | xor r10, r1, r4         |
| MEM[5] <= 32'h02415580;      | 02415580         | slt r11, r2, r4         |
| MEM[6] <= 32'h00520600;      | 00520600         | addi r12, r4, 5         |
| MEM[7] <= 32'h00209181;      | 00209181         | sw r3, 2(r1)            |
| MEM[8] <= 32'h00208681;      | 00208681         | lw r13, 2(r1)           |
| MEM[9] <= 32'h00f00002;      | 00f00002         | beq r0, r0, 15          |
| MEM[25] <= 32'h00210700;     | 00210700         | add r14, r2, r2         |

### Verifying each instructions using the waveform

Value of general purpose registers before running the program (As per the design code)

| **Register** | **Value (Hex)** | **Value (Decimal)** |
|--------------|------------------|----------------------|
| REG[0]       | 0x00000000       | 0                   |
| REG[1]       | 0x00000001       | 1                   |
| REG[2]       | 0x00000002       | 2                   |
| REG[3]       | 0x00000003       | 3                   |
| REG[4]       | 0x00000004       | 4                   |
| REG[5]       | 0x00000005       | 5                   |
| REG[6]       | 0x00000006       | 6                   | 

## Instruction 1: add r6, r1, r2  

![in1](https://github.com/user-attachments/assets/0362eedf-8249-4b89-b252-fa5d375548f7)

- REG[6] = REG[1] + REG[2] = 1 + 2 = 3
- 
| **Register** | **Value (Hex)** | **Value (Decimal)** |
|--------------|------------------|----------------------|
| REG[0]       | 0x00000000       | 0                   |
| REG[1]       | 0x00000001       | 1                   |
| REG[2]       | 0x00000002       | 2                   |
| REG[3]       | 0x00000003       | 3                   |
| REG[4]       | 0x00000004       | 4                   |
| REG[5]       | 0x00000005       | 5                   |
| REG[6]       | 0x00000003       | 3                   |


## Instruction 2: sub r7, r1, r2

![in2](https://github.com/user-attachments/assets/8cbfe1b1-4933-411c-948d-1f1b7e20c60e)

- REG[7] = REG[1] - REG[2] = 1 - 2 = -1
- 
| **Register** | **Value (Hex)** | **Value (Decimal)** |
|--------------|------------------|----------------------|
| REG[0]       | 0x00000000       | 0                   |
| REG[1]       | 0x00000001       | 1                   |
| REG[2]       | 0x00000002       | 2                   |
| REG[3]       | 0x00000003       | 3                   |
| REG[4]       | 0x00000004       | 4                   |
| REG[5]       | 0x00000005       | 5                   |
| REG[6]       | 0x00000003       | 3                   |
| REG[7]       | 0xFFFFFFFF       | -1                  |


## Instruction 3: and r8, r1, r3

![in3](https://github.com/user-attachments/assets/aa25ca12-7163-44de-ad97-6939daee59d3)

- REG[8] = REG[1] AND REG[3] = 1 AND 3 = 01 AND 11 = 01 = 1 (decimal)  

| **Register** | **Value (Hex)** | **Value (Decimal)** |
|--------------|------------------|----------------------|
| REG[0]       | 0x00000000       | 0                   |
| REG[1]       | 0x00000001       | 1                   |
| REG[2]       | 0x00000002       | 2                   |
| REG[3]       | 0x00000003       | 3                   |
| REG[4]       | 0x00000004       | 4                   |
| REG[5]       | 0x00000005       | 5                   |
| REG[6]       | 0x00000003       | 3                   |
| REG[7]       | 0xFFFFFFFF       | -1                  |
| REG[8]       | 0x00000001       | 1                   |


## Instruction 4: or r9, r2, r5

![in4](https://github.com/user-attachments/assets/718af532-02c0-45b6-8d9e-a4901b59260d)

- REG[9] = REG[2] OR REG[5] = 2 OR 5 = 010 OR 101 = 111 = 7 (decimal)
  
| **Register** | **Value (Hex)** | **Value (Decimal)** |
|--------------|------------------|----------------------|
| REG[0]       | 0x00000000       | 0                   |
| REG[1]       | 0x00000001       | 1                   |
| REG[2]       | 0x00000002       | 2                   |
| REG[3]       | 0x00000003       | 3                   |
| REG[4]       | 0x00000004       | 4                   |
| REG[5]       | 0x00000005       | 5                   |
| REG[6]       | 0x00000003       | 3                   |
| REG[7]       | 0xFFFFFFFF       | -1                  |
| REG[8]       | 0x00000001       | 1                   |
| REG[9]       | 0x00000007       | 7                   |


## Instruction 5: xor r10, r1, r4 

![in5](https://github.com/user-attachments/assets/284d60d7-5552-4977-b6dc-551714604ad8)

- REG[10] = REG[1] XOR REG[4] = 1 XOR 4 = 001 OR 100 = 101 = 5 (decimal) 

| **Register** | **Value (Hex)** | **Value (Decimal)** |
|--------------|------------------|----------------------|
| REG[0]       | 0x00000000       | 0                   |
| REG[1]       | 0x00000001       | 1                   |
| REG[2]       | 0x00000002       | 2                   |
| REG[3]       | 0x00000003       | 3                   |
| REG[4]       | 0x00000004       | 4                   |
| REG[5]       | 0x00000005       | 5                   |
| REG[6]       | 0x00000003       | 3                   |
| REG[7]       | 0xFFFFFFFF       | -1                  |
| REG[8]       | 0x00000001       | 1                   |
| REG[9]       | 0x00000007       | 7                   |
| REG[10]      | 0x00000005       | 5                   |


## Instruction 6: slt r11, r2, r4  

![in6](https://github.com/user-attachments/assets/604450bb-8b33-414b-a59a-611ca579c12c)

- REG[2] = 2 (0x00000002)
- REG[4] = 4 (0x00000004)
- Since 2 < 4, the slt instruction will set REG[11] to 1

| **Register** | **Value (Hex)** | **Value (Decimal)** |
|--------------|------------------|----------------------|
| REG[0]       | 0x00000000       | 0                   |
| REG[1]       | 0x00000001       | 1                   |
| REG[2]       | 0x00000002       | 2                   |
| REG[3]       | 0x00000003       | 3                   |
| REG[4]       | 0x00000004       | 4                   |
| REG[5]       | 0x00000005       | 5                   |
| REG[6]       | 0x00000003       | 3                   |
| REG[7]       | 0xFFFFFFFF       | -1                  |
| REG[8]       | 0x00000001       | 1                   |
| REG[9]       | 0x00000007       | 7                   |
| REG[10]      | 0x00000005       | 5                   |
| REG[11]      | 0x00000001       | 1                   |


## Instruction 7: addi r12, r4, 5 

![in7](https://github.com/user-attachments/assets/7e90ff88-e93d-4a0b-bf1c-50880de82dc1)

- REG[4] = 4 (0x00000004)
- Immediate value = 5
- REG[12] = REG[4] + 5 = 4 + 5 = 9

| **Register** | **Value (Hex)** | **Value (Decimal)** |
|--------------|------------------|---------------------|
| REG[0]       | 0x00000000       | 0                   |
| REG[1]       | 0x00000001       | 1                   |
| REG[2]       | 0x00000002       | 2                   |
| REG[3]       | 0x00000003       | 3                   |
| REG[4]       | 0x00000004       | 4                   |
| REG[5]       | 0x00000005       | 5                   |
| REG[6]       | 0x00000003       | 3                   |
| REG[7]       | 0xFFFFFFFF       | -1                  |
| REG[8]       | 0x00000001       | 1                   |
| REG[9]       | 0x00000007       | 7                   |
| REG[10]      | 0x00000005       | 5                   |
| REG[11]      | 0x00000001       | 1                   |
| REG[12]      | 0x00000009       | 9                   |

## Instruction 8: sw r3, 2(r1)

![in8](https://github.com/user-attachments/assets/6b34f0ad-173f-4e8d-8fad-76a93a729531)

- REG[3] = 3 (0x00000003)
- REG[1] = 1 (0x00000001)
- The instruction performs: address = REG[1] + 2 = 1 + 2 = 3
- So, the value REG[3] = 3 will be stored at memory location MEM[3] (address 3)

| **Register** | **Value (Hex)** | **Value (Decimal)** |
|--------------|------------------|---------------------|
| REG[0]       | 0x00000000       | 0                   |
| REG[1]       | 0x00000001       | 1                   |
| REG[2]       | 0x00000002       | 2                   |
| REG[3]       | 0x00000003       | 3                   |
| REG[4]       | 0x00000004       | 4                   |
| REG[5]       | 0x00000005       | 5                   |
| REG[6]       | 0x00000003       | 3                   |
| REG[7]       | 0xFFFFFFFF       | -1                  |
| REG[8]       | 0x00000001       | 1                   |
| REG[9]       | 0x00000007       | 7                   |
| REG[10]      | 0x00000005       | 5                   |
| REG[11]      | 0x00000001       | 1                   |
| REG[12]      | 0x00000009       | 9                   |
| **Memory**   | **Value (Hex)**  | **Value (Decimal)**  |
| MEM[0]       | 0x00000000       | 0                   |
| MEM[1]       | 0x00000001       | 1                   |
| MEM[2]       | 0x00000002       | 2                   |
| MEM[3]       | 0x00000003       | 3                   |


## Instruction 9: lw r13, 2(r1)

![in9](https://github.com/user-attachments/assets/bbedf59d-02c6-44ef-b1da-5d34533c300a)

- R1 = 1 (0x00000001)
- Memory at address R1 + 2 = 1 + 2 = 3, which is MEM[3]
- From the previous sw instruction, MEM[3] = 0x00000003
- So, the value at MEM[3] (which is 0x00000003) will be loaded into REG[13].

| **Register** | **Value (Hex)** | **Value (Decimal)** |
|--------------|------------------|---------------------|
| REG[0]       | 0x00000000       | 0                   |
| REG[1]       | 0x00000001       | 1                   |
| REG[2]       | 0x00000002       | 2                   |
| REG[3]       | 0x00000003       | 3                   |
| REG[4]       | 0x00000004       | 4                   |
| REG[5]       | 0x00000005       | 5                   |
| REG[6]       | 0x00000003       | 3                   |
| REG[7]       | 0xFFFFFFFF       | -1                  |
| REG[8]       | 0x00000001       | 1                   |
| REG[9]       | 0x00000007       | 7                   |
| REG[10]      | 0x00000005       | 5                   |
| REG[11]      | 0x00000001       | 1                   |
| REG[12]      | 0x00000009       | 9                   |
| REG[13]      | 0x00000003       | 3                   |
| **Memory**   | **Value (Hex)**  | **Value (Decimal)**  |
| MEM[0]       | 0x00000000       | 0                   |
| MEM[1]       | 0x00000001       | 1                   |
| MEM[2]       | 0x00000002       | 2                   |
| MEM[3]       | 0x00000003       | 3                   |


## Instruction 10: beq r0, r0, 15 

![new](https://github.com/user-attachments/assets/8711fe44-649b-4a13-a525-e4b72e78d57e)


- REG[0] = 0 (0x00000000)
- The beq instruction compares r0 with r0, and since both are equal (both are 0), the branch is taken.
- The instruction specifies a branch offset of 15, so the program will jump to address 15 in the instruction memory.
- No changes in the table

| **Register** | **Value (Hex)** | **Value (Decimal)** |
|--------------|------------------|---------------------|
| REG[0]       | 0x00000000       | 0                   |
| REG[1]       | 0x00000001       | 1                   |
| REG[2]       | 0x00000002       | 2                   |
| REG[3]       | 0x00000003       | 3                   |
| REG[4]       | 0x00000004       | 4                   |
| REG[5]       | 0x00000005       | 5                   |
| REG[6]       | 0x00000003       | 3                   |
| REG[7]       | 0xFFFFFFFF       | -1                  |
| REG[8]       | 0x00000001       | 1                   |
| REG[9]       | 0x00000007       | 7                   |
| REG[10]      | 0x00000005       | 5                   |
| REG[11]      | 0x00000001       | 1                   |
| REG[12]      | 0x00000009       | 9                   |
| REG[13]      | 0x00000003       | 3                   |
| **Memory**   | **Value (Hex)**  | **Value (Decimal)**  |
| MEM[0]       | 0x00000000       | 0                   |
| MEM[1]       | 0x00000001       | 1                   |
| MEM[2]       | 0x00000002       | 2                   |
| MEM[3]       | 0x00000003       | 3                   |


## Instruction 11: add r14, r2, r2 

![in11](https://github.com/user-attachments/assets/5466a591-39ba-4513-8174-04bf51d8fc8e)


- REG[2] = 0x00000002 (2 in decimal)
- So, r2 + r2 = 2 + 2 = 4
- After executing the instruction, REG[14] will be updated with the value 4 (0x00000004 in hex).


| **Register** | **Value (Hex)** | **Value (Decimal)** |
|--------------|------------------|---------------------|
| REG[0]       | 0x00000000       | 0                   |
| REG[1]       | 0x00000001       | 1                   |
| REG[2]       | 0x00000002       | 2                   |
| REG[3]       | 0x00000003       | 3                   |
| REG[4]       | 0x00000004       | 4                   |
| REG[5]       | 0x00000005       | 5                   |
| REG[6]       | 0x00000003       | 3                   |
| REG[7]       | 0xFFFFFFFF       | -1                  |
| REG[8]       | 0x00000001       | 1                   |
| REG[9]       | 0x00000007       | 7                   |
| REG[10]      | 0x00000005       | 5                   |
| REG[11]      | 0x00000001       | 1                   |
| REG[12]      | 0x00000009       | 9                   |
| REG[13]      | 0x00000003       | 3                   |
| REG[14]      | 0x00000004       | 4                   |
| **Memory**   | **Value (Hex)**  | **Value (Decimal)**  |
| MEM[0]       | 0x00000000       | 0                   |
| MEM[1]       | 0x00000001       | 1                   |
| MEM[2]       | 0x00000002       | 2                   |
| MEM[3]       | 0x00000003       | 3                   |

---

### VIDEO

https://drive.google.com/file/d/1DmFYPMZrU0df0-gx2Q0ekcFe76f23znp/view?usp=sharing

---


