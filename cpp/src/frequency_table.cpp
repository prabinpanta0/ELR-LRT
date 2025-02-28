#include "frequency_table.h"
#include <cmath>

void FrequencyTable::update(const std::string& context, uint8_t next_byte) {
    freq_table[context][next_byte]++;
}

std::array<int, 256> FrequencyTable::get_counts(const std::string& context) const {
    auto it = freq_table.find(context);
    if (it != freq_table.end()) {
        return it->second;
    } else {
        return std::array<int, 256>{0};
    }
}

float FrequencyTable::compute_entropy(const std::string& context) const {
    auto counts = get_counts(context);
    float total = 0;
    for (int count : counts) total += count;
    if (total == 0) return 0;
    float entropy = 0;
    for (int count : counts) {
        if (count > 0) {
            float p = count / total;
            entropy -= p * std::log(p);
        }
    }
    return entropy;
}
