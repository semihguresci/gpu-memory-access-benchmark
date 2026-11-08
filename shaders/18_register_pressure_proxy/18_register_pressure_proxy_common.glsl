layout(local_size_x = 256) in;

layout(push_constant) uniform PushConstants {
    uint element_count;
    uint rounds;
    uint temp_count;
    uint reserved;
} pc;

layout(std430, binding = 0) readonly buffer SourceBuffer {
    uint values[];
} src_buffer;

layout(std430, binding = 1) writeonly buffer DestinationBuffer {
    uint values[];
} dst_buffer;

#ifndef TEMP_COUNT
#error TEMP_COUNT must be defined before including this file
#endif

uint rotl32(uint value, uint shift) {
    shift &= 31u;
    return (value << shift) | (value >> ((32u - shift) & 31u));
}

uint transform_value(uint seed, uint index, uint temp_count, uint rounds) {
    const uint chain_count = 32u;
    uint state[TEMP_COUNT];

    if (temp_count != TEMP_COUNT) {
        return 0u;
    }

    uint result = seed ^ (index * 0x27D4EB2Du) ^ (chain_count * 0x85EBCA6Bu) ^ (rounds * 0x9E3779B9u);
    // All variants initialize/update/fold the same 32 independent chains. Only
    // the batch size, and hence the intended live-state schedule, varies.
    for (uint base = 0u; base < chain_count; base += TEMP_COUNT) {
        for (uint slot = 0u; slot < TEMP_COUNT; ++slot) {
            const uint lane = base + slot;
            uint lane_seed = seed ^ ((lane + 1u) * 0x7F4A7C15u);
            lane_seed += index * 0x85EBCA6Bu;
            lane_seed ^= rotl32(seed + ((lane + 1u) * 0x7F4A7C15u), (lane * 5u + 1u) & 31u);
            state[slot] = lane_seed;
        }
        for (uint round = 0u; round < rounds; ++round) {
            for (uint slot = 0u; slot < TEMP_COUNT; ++slot) {
                const uint lane = base + slot;
                const uint mixed = rotl32(state[slot] + (seed ^ (round * 0xC2B2AE35u)) +
                                             ((lane + 1u) * 0x7F4A7C15u),
                                         (lane + round) & 31u);
                state[slot] = (mixed ^ seed) + 0xD2511F53u + ((lane + 1u) * 0xA24BAED5u);
            }
        }
        for (uint slot = 0u; slot < TEMP_COUNT; ++slot) {
            const uint lane = base + slot;
            result ^= rotl32(state[slot] + ((lane + 1u) * 0xD2511F53u), (lane + chain_count) & 31u);
            result += state[slot] ^ ((lane + 1u) * 0xA24BAED5u);
        }
    }
    return result;
}

void main() {
    const uint index = gl_GlobalInvocationID.x;
    if (index >= pc.element_count) {
        return;
    }

    const uint seed = src_buffer.values[index];
    dst_buffer.values[index] = transform_value(seed, index, pc.temp_count, pc.rounds);
}
