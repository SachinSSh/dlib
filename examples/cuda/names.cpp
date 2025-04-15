#include <iostream>
#include <vector>
#include <dlib/cuda/cuda_dnn.h>
#include <dlib/dnn.h>
#include <dlib/image_processing.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_io.h>
#include <dlib/opencv.h>

// Define the face recognition network architecture (ResNet)
template <template <int,template<typename>class,int,typename> class block, int N, 
          template<typename>class BN, typename SUBNET>
using residual = dlib::add_prev1<block<N,BN,1,dlib::tag1<SUBNET>>>;

template <template <int,template<typename>class,int,typename> class block, int N, 
          template<typename>class BN, typename SUBNET>
using residual_down = dlib::add_prev2<dlib::avg_pool<2,2,2,2,dlib::skip1<dlib::tag2<block<N,BN,2,dlib::tag1<SUBNET>>>>>>;

template <int N, template <typename> class BN, int stride, typename SUBNET> 
using block  = BN<dlib::con<N,3,3,1,1,dlib::relu<BN<dlib::con<N,3,3,stride,stride,SUBNET>>>>>;

template <int N, typename SUBNET> using ares      = dlib::relu<residual<block,N,dlib::affine,SUBNET>>;
template <int N, typename SUBNET> using ares_down = dlib::relu<residual_down<block,N,dlib::affine,SUBNET>>;

template <typename SUBNET> using alevel0 = ares_down<256,SUBNET>;
template <typename SUBNET> using alevel1 = ares<256,ares<256,ares_down<256,SUBNET>>>;
template <typename SUBNET> using alevel2 = ares<128,ares<128,ares_down<128,SUBNET>>>;
template <typename SUBNET> using alevel3 = ares<64,ares<64,ares<64,ares_down<64,SUBNET>>>>;
template <typename SUBNET> using alevel4 = ares<32,ares<32,ares<32,SUBNET>>>;

using anet_type = dlib::loss_metric<dlib::fc_no_bias<128,dlib::avg_pool_everything<
                            alevel0<
                            alevel1<
                            alevel2<
                            alevel3<
                            alevel4<
                            dlib::max_pool<3,3,2,2,dlib::relu<dlib::affine<dlib::con<32,7,7,2,2,
                            dlib::input_rgb_image_sized<150>
                            >>>>>>>>>>>>;

int main(int argc, char** argv) {
    try {
        // Check command line arguments
        if (argc != 3) {
            std::cout << "Usage: " << argv[0] << " <path_to_image> <path_to_dlib_face_recognition_model.dat>" << std::endl;
            return 1;
        }

        const std::string img_path = argv[1];
        const std::string model_path = argv[2];

        // Initialize CUDA device
        dlib::cuda::set_device(0);
        std::cout << "CUDA device selected: " << dlib::cuda::get_device() << std::endl;
        
        // Load the image
        dlib::matrix<dlib::rgb_pixel> img;
        dlib::load_image(img, img_path);
        
        // Create face detector and load face recognition model
        dlib::frontal_face_detector detector = dlib::get_frontal_face_detector();
        dlib::shape_predictor sp;
        anet_type net;
        
        // The shape predictor is needed to align faces
        dlib::deserialize("shape_predictor_68_face_landmarks.dat") >> sp;
        
        // Load the face recognition model
        dlib::deserialize(model_path) >> net;
        
        // Use CUDA for face detection (wherever dlib supports it)
        dlib::cuda::enable_device_sync(); // For debugging purposes
        
        // Detect faces
        std::vector<dlib::rectangle> faces = detector(img);
        std::cout << "Number of faces detected: " << faces.size() << std::endl;
        
        // Process each face
        std::vector<dlib::matrix<dlib::rgb_pixel>> face_chips;
        for (auto& face : faces) {
            // Get facial landmarks for alignment
            auto shape = sp(img, face);
            dlib::matrix<dlib::rgb_pixel> face_chip;
            // Extract and align the face
            dlib::extract_image_chip(img, dlib::get_face_chip_details(shape, 150, 0.25), face_chip);
            face_chips.push_back(std::move(face_chip));
        }
        
        // Compute face descriptors using GPU acceleration via CUDA
        std::vector<dlib::matrix<float,0,1>> face_descriptors = net(face_chips);
        
        // Display the results
        std::cout << "Face descriptors computed: " << face_descriptors.size() << std::endl;
        
        // Example: Compare first two faces if available
        if (face_descriptors.size() >= 2) {
            double distance = dlib::length(face_descriptors[0] - face_descriptors[1]);
            std::cout << "Distance between first two faces: " << distance << std::endl;
            std::cout << "Faces are " << (distance < 0.6 ? "the same person" : "different people") << std::endl;
        }
        
        // Save the results (optional)
        for (size_t i = 0; i < face_chips.size(); ++i) {
            dlib::save_jpeg(face_chips[i], "face_" + std::to_string(i) + ".jpg");
        }
        
    } catch (const std::exception& e) {
        std::cout << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
