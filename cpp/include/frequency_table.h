#ifndef FREQUENCY_TABLE_H
#define FREQUENCY_TABLE_H

#include <unordered_map>
#include <string>
#include <array>
#include <cstdint>

/**
 * @brief Frequency table for computing byte sequence entropy.
 * 
 * This class maintains frequency distributions of bytes following specific contexts,
 * which is used to compute entropy for the Dynamic Byte Patching Module (DBPM).
 */
class FrequencyTable {
public:
    FrequencyTable();
    ~FrequencyTable();
    
    /**
     * @brief Update the frequency table with a context-byte pair.
     * @param context The context string preceding the byte
     * @param next_byte The byte value that follows the context
     */
    void update(const std::string& context, uint8_t next_byte);
    
    /**
     * @brief Get the byte frequency counts for a given context.
     * @param context The context string to query
     * @return Array of 256 counts (one for each possible byte value)
     */
    std::array<int, 256> get_counts(const std::string& context) const;
    
    /**
     * @brief Compute Shannon entropy for a given context.
     * @param context The context string to analyze
     * @return Entropy value (higher values indicate more randomness)
     */
    float compute_entropy(const std::string& context) const;
    
    // Legacy methods for backward compatibility
    void add_sequence(const std::string& sequence);
    double get_frequency(const std::string& pattern) const;
    void clear();
    
private:
    std::unordered_map<std::string, int> frequencies;        ///< Pattern frequency counts
    int total_count;                                          ///< Total number of patterns
    std::unordered_map<std::string, std::array<int, 256>> freq_table;  ///< Context-based byte frequencies
};

#endif // FREQUENCY_TABLE_H
