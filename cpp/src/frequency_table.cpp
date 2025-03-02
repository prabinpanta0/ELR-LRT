#include "../include/frequency_table.h"
#include <cmath>
#include <cstdint>

FrequencyTable::FrequencyTable() : total_count(0) {}

FrequencyTable::~FrequencyTable() {}

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

void FrequencyTable::add_sequence(const std::string& sequence) {
    // Simple implementation: count k-mers of length 3
    if (sequence.length() < 3) return;
    
    for (size_t i = 0; i <= sequence.length() - 3; ++i) {
        std::string kmer = sequence.substr(i, 3);
        frequencies[kmer]++;
        total_count++;
    }
}

double FrequencyTable::get_frequency(const std::string& pattern) const {
    if (total_count == 0) return 0.0;
    
    auto it = frequencies.find(pattern);
    if (it != frequencies.end()) {
        return static_cast<double>(it->second) / total_count;
    }
    return 0.0;
}

void FrequencyTable::clear() {
    frequencies.clear();
    total_count = 0;
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
