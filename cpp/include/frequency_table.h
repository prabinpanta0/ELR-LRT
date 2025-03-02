#ifndef FREQUENCY_TABLE_H
#define FREQUENCY_TABLE_H

#include <unordered_map>
#include <string>
#include <array>
#include <cstdint>

class FrequencyTable {
public:
    FrequencyTable();
    ~FrequencyTable();
    
    // Added declarations:
    void update(const std::string& context, uint8_t next_byte);
    std::array<int, 256> get_counts(const std::string& context) const;
    float compute_entropy(const std::string& context) const;
    
    // Existing methods
    void add_sequence(const std::string& sequence);
    double get_frequency(const std::string& pattern) const;
    void clear();
    
private:
    // Private implementation details
    std::unordered_map<std::string, int> frequencies;
    int total_count;
    
    // Added data member for update and get_counts
    std::unordered_map<std::string, std::array<int, 256>> freq_table;
};

#endif // FREQUENCY_TABLE_H
