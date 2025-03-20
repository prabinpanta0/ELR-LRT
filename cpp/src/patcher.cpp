#include "patcher.h"  // Changed from relative path if needed
#include <string>
#include <vector>
#include "frequency_table.h"  // Changed from relative path if needed

std::vector<std::vector<uint8_t>> patch_sequence(const std::vector<uint8_t>& bytes, int k, float theta, float theta_r) {
    FrequencyTable freq_table;
    for (size_t i = k; i < bytes.size(); ++i) {
        std::string context(bytes.begin() + i - k, bytes.begin() + i);
        freq_table.update(context, bytes[i]);
    }

    std::vector<std::vector<uint8_t>> patches;
    std::vector<uint8_t> current_patch;
    float prev_entropy = 0;
    for (size_t i = 0; i < bytes.size(); ++i) {
        current_patch.push_back(bytes[i]);
        if (i >= k) {
            std::string context(bytes.begin() + i - k, bytes.begin() + i);
            float entropy = freq_table.compute_entropy(context);
            float delta_entropy = entropy - prev_entropy;
            if (entropy > theta || delta_entropy > theta_r) {
                patches.push_back(current_patch);
                current_patch.clear();
            }
            prev_entropy = entropy;
        }
    }
    if (!current_patch.empty()) patches.push_back(current_patch);
    return patches;
}
