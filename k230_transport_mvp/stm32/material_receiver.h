#ifndef K230_MATERIAL_RECEIVER_H
#define K230_MATERIAL_RECEIVER_H
#include "protocol.h"

typedef enum {
    K230_MAT_INIT, K230_MAT_OK, K230_MAT_NOT_FOUND, K230_MAT_UNKNOWN,
    K230_MAT_UNAVAILABLE, K230_MAT_ERROR, K230_MAT_BINDING_MISMATCH,
    K230_MAT_STALE
} k230_material_status;

typedef struct {
    uint8_t color_id; /* 0 unknown; 1 red, 2 yellow, 3 blue, 4 green, 5 black, 6 cyan */
    uint8_t stable, geometry_valid, has_score;
    float u_px, v_px, semimajor_px, semiminor_px, angle_rad, score_hint;
} k230_material_object;

typedef struct {
    k230_material_status status;
    uint32_t frame_id, timestamp_ms, received_ms;
    uint16_t seq, width, height;
    uint8_t camera_id, count, complete, is_simulated, fresh, usable;
    k230_material_object objects[3];
} k230_material_snapshot;

typedef struct {
    k230_parser parser;
    k230_material_snapshot latest;
    char profile_id[65];
    char source_session[33];
    uint32_t ttl_ms, last_frame_id, last_timestamp_ms, rejected_snapshots;
    uint16_t width, height;
    uint8_t camera_id, configured, have_frame, have_snapshot;
} k230_material_receiver;

/* Profile is a SHA256 of the full capture + algorithm contract, not just width.
   All calls have ONE owner. Use a queue/critical section for an RTOS handoff.
   No HAL calls, allocation, motor output or enable/authorization are provided. */
int k230_material_init(k230_material_receiver *r, uint8_t camera_id,
                       uint16_t width, uint16_t height, const char *profile_id, const char *source_session,
                       uint32_t ttl_ms, uint32_t frame_timeout_ms);
/* 1 only for a new, bound S snapshot (negative and SIM snapshots included).
   Consume bytes promptly, with MCU local monotonic time. A/C/V cannot renew S. */
int k230_material_feed(k230_material_receiver *r, uint8_t byte, uint32_t now_ms);
/* Copies owned numeric data. 1 means fresh, complete, real, stable, known-color
   image measurements; it is NEVER a motion permission or millimetre pose. */
int k230_material_read(k230_material_receiver *r, uint32_t now_ms,
                       k230_material_snapshot *out);
/* Explicitly reset parser AND observation high-water marks, only after motors
   are independently disabled. This does not settle uncertain physical actions. */
int k230_material_reset(k230_material_receiver *r, int motors_disabled, const char *new_source_session);
#endif
