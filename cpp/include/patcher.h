#ifndef PATCHER_H
#define PATCHER_H

#include <string>
#include <vector>

class Patcher {
public:
    Patcher();
    ~Patcher();
    
    // Add your patcher class methods here
    void patch_sequence(const std::string& sequence, std::vector<int>& positions);
    std::string get_patched_sequence() const;
    void reset();
    
private:
    // Private implementation details
    std::string original_sequence;
    std::string patched_sequence;
};

#endif // PATCHER_H
