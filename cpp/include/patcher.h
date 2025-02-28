#ifndef PATCHER_H
#define PATCHER_H

#include <vector>
#include "frequency_table.h"

std::vector<std::vector<uint8_t>> patch_sequence(const std::vector<uint8_t>& bytes, int k, float theta, float theta_r);

#endif
