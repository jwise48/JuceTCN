#include <string>


inline std::string BASE_DIR = "/Users/wisjhn/Plugins/AIBookRepo/JuceTCN/model";


// Add this struct definition before the JuceTCNAudioProcessor class
struct ModelConfig {
    std::string name;
    std::string model_type;
    int nblocks;
    int dilation_growth;
    int kernel_size;
    bool causal;
    double train_fraction;
    int max_epochs;
    int channel_width;
    
    // Constructor with default values for optional parameters
    ModelConfig(const std::string& name_, 
                const std::string& model_type_,
                int nblocks_, 
                int dilation_growth_,
                int kernel_size_,
                bool causal_)
        : name(name_)
        , model_type(model_type_)
        , nblocks(nblocks_)
        , dilation_growth(dilation_growth_)
        , kernel_size(kernel_size_)
        , causal(causal_)
    {}
    
    // Calculate receptive field based on architecture
    int calculateReceptiveField() const {
        double rf = kernel_size * dilation_growth;
        for (int layer = 1; layer < nblocks; ++layer) {
            rf += ((kernel_size - 1) * std::pow(dilation_growth, layer));
        }
        std::cout << "==== calculateReceptiveField ====" << std::endl;
        std::cout << "rf " << rf << std::endl;
        return static_cast<int>(rf);
    }

    // Calculate receptive field based on architecture
    std::string modelPath() const {return BASE_DIR + "/" + name + ".pt";}
};

