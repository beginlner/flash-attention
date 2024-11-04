#pragma once

#include "cutlass/arch/barrier.h"

namespace flash {

////////////////////////////////////////////////////////////////////////////////////////////////////
// Enumerates the reserved named barriers to avoid potential conflicts

enum class NamedBarriers {
    SReady = 1,
    SoftmaxReady = 2,
    BlockTableReady = 3,
};

} // flash