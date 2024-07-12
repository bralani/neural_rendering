#include <torch/script.h> // One-stop header.
#include <vector>
#include "torch_runner.h"
#include <iostream>
std::string global_model_path;

typedef enum {
    normal = 1,
    neural = 2
} render_type;
void set_model_path(const char* model_path)
{
    // check if the model exists
    std::ifstream
        file(model_path);
    if (!file)
    {
        std::cout << "Model file does not exist" << std::endl;
        throw;
    }

    global_model_path = std::string(model_path);
}

torch::Device getDevice()
{
    // build a tensor and move it to cuda
    try {
        torch::tensor({ 1 }).to(at::kCUDA);
        std::cout << "CUDA is available." << std::endl;
        return torch::Device(torch::kCUDA);
    } catch (const c10::Error& e) {
        try {
            torch::tensor({ 1 }).to(at::kMPS);
            std::cout << "Metal is available." << std::endl;
            return torch::Device(torch::kMPS);
        } catch (const c10::Error& e) {
            std::cout << "CUDA or Metal support are not available -> switch to CPU" << std::endl;
            return torch::Device(torch::kCPU);
        }
    }
}

auto device_torch = getDevice();

static torch::jit::script::Module& get_model() {

    static torch::jit::script::Module module;
    static bool is_loaded = false;
    if (!is_loaded) {
        try {
            module = torch::jit::load(global_model_path);
            module.to(device_torch);
            module.eval();
            is_loaded = true;
        }
        catch (const c10::Error& e) {
            std::cerr << "error loading the model\n";
            throw;
        }
    }
    return module;
}
int run_torch(double* para)
{
    torch::jit::script::Module& module = get_model();
    std::vector<float> input_vec;
    std::vector<torch::jit::IValue> inputs;
    torch::Tensor input_tensor;
    para++;
    for (int i = 0; i < 5; i++)
    {
        input_vec.push_back(*para);
        para++;
    }
    inputs.push_back(torch::tensor({ {input_vec[0], input_vec[1], input_vec[2], input_vec[3]} }, torch::kFloat).to(device_torch));
    
    // inputs.push_back(input_tensor);
    at::Tensor output = module.forward(inputs).toTensor().to(device_torch);
    auto tmp = output.data_ptr<float>();
    std::vector<double> output_vector(output.data_ptr<float>(), output.data_ptr<float>() + output.numel());
    float res = float(output_vector[0]);
    
    return res > 0.5 ? 1 : 0;
}