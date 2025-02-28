#ifndef FREQUENCY_TABLE_H
#define FREQUENCY_TABLE_H

#include <unordered_map>
#include <array>
#include <string>

class FrequencyTable {
private:
    std::unordered_map<std::string, std::array<int, 256>> freq_table;
public:
    void update(const std::string& context, uint8_t next_byte);
    std::array<int, 256> get_counts(const std::string& context) const;
    float compute_entropy(const std::string& context) const;
};
#endif
