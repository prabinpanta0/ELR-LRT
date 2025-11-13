#ifndef PATCHER_H
#define PATCHER_H

#include <string>
#include <vector>
#include <cstdint>

/**
 * @brief Segment a byte sequence into variable-length patches based on entropy thresholds.
 * 
 * This function implements the Dynamic Byte Patching Module (DBPM) which analyzes
 * the entropy of byte sequences and creates patches when entropy thresholds are exceeded.
 * 
 * @param bytes Input byte sequence to be patched
 * @param k Context window size for entropy computation
 * @param theta Entropy threshold for creating new patches
 * @param theta_r Delta entropy threshold for patch segmentation
 * @return Vector of byte patches, where each patch is a vector of bytes
 */
std::vector<std::vector<uint8_t>> patch_sequence(const std::vector<uint8_t>& bytes, int k, float theta, float theta_r);

/**
 * @brief Legacy Patcher class for backward compatibility.
 * 
 * Note: This class is maintained for binary compatibility but is not actively used.
 * The free function patch_sequence() should be used instead for new code.
 */
class Patcher {
public:
    Patcher();
    ~Patcher();
    
    void patch_sequence(const std::string& sequence, std::vector<int>& positions);
    std::string get_patched_sequence() const;
    void reset();
    
private:
    std::string original_sequence;
    std::string patched_sequence;
};

#endif // PATCHER_H
