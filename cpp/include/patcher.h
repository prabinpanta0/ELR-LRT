#ifndef PATCHER_H
#define PATCHER_H

#include <string>
#include <vector>
#include <cstdint>

// Define the patch_sequence function directly (free function, not class method)
std::vector<std::vector<uint8_t>> patch_sequence(const std::vector<uint8_t>& bytes, int k, float theta, float theta_r);

// The Patcher class - we'll implement minimal methods to match bindings
class Patcher {
public:
    Patcher();
    ~Patcher();
    
    // Add a method to access the free function
    void patch_sequence(const std::string& sequence, std::vector<int>& positions);
    std::string get_patched_sequence() const;
    void reset();
    
private:
    // Private implementation details
    std::string original_sequence;
    std::string patched_sequence;
};

#endif // PATCHER_H
