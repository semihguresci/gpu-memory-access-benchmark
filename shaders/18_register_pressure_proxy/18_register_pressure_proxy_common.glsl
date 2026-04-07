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
    uint state[TEMP_COUNT];

    if (temp_count != TEMP_COUNT) {
        return 0u;
    }

    for (uint lane = 0u; lane < TEMP_COUNT; ++lane) {
        uint lane_seed = seed ^ ((lane + 1u) * 0x7F4A7C15u);
        lane_seed += index * 0x85EBCA6Bu;
        lane_seed ^= rotl32(seed + ((lane + 1u) * 0x7F4A7C15u), (lane * 5u + 1u) & 31u);
        state[lane] = lane_seed;
    }

    for (uint round = 0u; round < rounds; ++round) {
        for (uint lane = 0u; lane < TEMP_COUNT; ++lane) {
            const uint left = state[(lane + TEMP_COUNT - 1u) % TEMP_COUNT];
            const uint right = state[(lane + 1u) % TEMP_COUNT];
            const uint current = state[lane];
            const uint mixed =
                rotl32(current + (left ^ (round * 0xC2B2AE35u)) + ((lane + 1u) * 0x7F4A7C15u), (lane + round) & 31u);
            state[lane] = (mixed ^ right) + 0xD2511F53u + ((lane + 1u) * 0xA24BAED5u);
        }
    }

    uint result = seed ^ (index * 0x27D4EB2Du) ^ (TEMP_COUNT * 0x85EBCA6Bu) ^ (rounds * 0x9E3779B9u);
    for (uint lane = 0u; lane < TEMP_COUNT; ++lane) {
        result ^= rotl32(state[lane] + ((lane + 1u) * 0xD2511F53u), (lane + TEMP_COUNT) & 31u);
        result += state[lane] ^ ((lane + 1u) * 0xA24BAED5u);
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
