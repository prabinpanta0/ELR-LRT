#ifndef FREQUENCY_TABLE_H
#define FREQUENCY_TABLE_H

#include <unordered_map>
#include <string>

class FrequencyTable {
public:
    FrequencyTable();
    ~FrequencyTable();
    
    // Add your frequency table methods here
    void add_sequence(const std::string& sequence);
    double get_frequency(const std::string& pattern) const;
    void clear();
    
private:
    // Private implementation details
    std::unordered_map<std::string, int> frequencies;
    int total_count;
};

#endif // FREQUENCY_TABLE_H
