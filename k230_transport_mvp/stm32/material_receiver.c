#include "material_receiver.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

static uint32_t integer(const char *s) { return (uint32_t)strtoul(s, NULL, 10); }

static int session_valid(const char *value) {
    unsigned i;
    if (!value || strlen(value) != 32) return 0;
    for (i = 0; i < 32; ++i) if (!strchr("0123456789abcdef", value[i])) return 0;
    return 1;
}

static int stored_geometry(const k230_material_object *o, uint16_t width, uint16_t height) {
    double dx, dy;
    if (!isfinite(o->u_px) || !isfinite(o->v_px) || !isfinite(o->semimajor_px) ||
        !isfinite(o->semiminor_px) || !isfinite(o->angle_rad)) return 0;
    if (!(o->semiminor_px > 0 && o->semimajor_px >= o->semiminor_px) ||
        o->u_px < 0 || o->u_px >= width || o->v_px < 0 || o->v_px >= height ||
        o->angle_rad < 0 || o->angle_rad >= 3.14159265358979323846) return 0;
    dx = hypot(o->semimajor_px*cos(o->angle_rad), o->semiminor_px*sin(o->angle_rad));
    dy = hypot(o->semimajor_px*sin(o->angle_rad), o->semiminor_px*cos(o->angle_rad));
    return o->u_px-dx >= -0.0001 && o->v_px-dy >= -0.0001 &&
           o->u_px+dx <= width-1+0.0001 && o->v_px+dy <= height-1+0.0001;
}

static void invalidate(k230_material_receiver *r, k230_material_status status) {
    memset(&r->latest, 0, sizeof(r->latest));
    r->latest.status = status;
    r->have_snapshot = 0;
}

int k230_material_init(k230_material_receiver *r, uint8_t camera_id,
                       uint16_t width, uint16_t height, const char *profile_id, const char *source_session,
                       uint32_t ttl_ms, uint32_t frame_timeout_ms) {
    unsigned i;
    memset(r, 0, sizeof(*r));
    k230_init(&r->parser, frame_timeout_ms);
    if ((camera_id != 1 && camera_id != 2) || !width || width > 4096 || !height || height > 4096 ||
        !ttl_ms || ttl_ms > INT32_MAX || !frame_timeout_ms || frame_timeout_ms > INT32_MAX ||
        !profile_id || strlen(profile_id) != 64 || !session_valid(source_session)) return 0;
    for (i = 0; i < 64; ++i) if (!strchr("0123456789abcdef", profile_id[i])) return 0;
    memcpy(r->profile_id, profile_id, 65);
    memcpy(r->source_session, source_session, 33);
    r->camera_id = camera_id; r->width = width; r->height = height; r->ttl_ms = ttl_ms;
    r->configured = 1;
    return 1;
}

int k230_material_feed(k230_material_receiver *r, uint8_t byte, uint32_t now_ms) {
    k230_material_snapshot next;
    const k230_frame *f;
    const char *const *v;
    uint32_t frame_id, timestamp_ms, delta;
    unsigned i;
    if (!r->configured || !k230_feed(&r->parser, byte, now_ms)) return 0;
    f = &r->parser.frame;
    if (f->type != 'S') return 0;
    v = f->fields;
    if ((uint8_t)(v[4][3]-'0') != r->camera_id || integer(v[8]) != r->width ||
        integer(v[9]) != r->height || strcmp(v[7], r->profile_id) || strcmp(v[15], r->source_session)) {
        ++r->rejected_snapshots;
        invalidate(r, K230_MAT_BINDING_MISMATCH);
        return 0;
    }
    frame_id = integer(v[5]); timestamp_ms = integer(v[6]);
    delta = timestamp_ms - r->last_timestamp_ms;
    /* Frame IDs never wrap in a session. Sender ticks may wrap, within half a
       uint32 period, but must advance. Higher tx_seq alone is not fresh data. */
    if (r->have_frame && (frame_id <= r->last_frame_id || !delta || delta > INT32_MAX)) {
        ++r->rejected_snapshots;
        return 0;
    }
    r->last_frame_id = frame_id; r->last_timestamp_ms = timestamp_ms;
    r->have_frame = 1;
    memset(&next, 0, sizeof(next));
    if (!strcmp(v[3], "OK")) next.status = K230_MAT_OK;
    else if (!strcmp(v[3], "NOT_FOUND")) next.status = K230_MAT_NOT_FOUND;
    else if (!strcmp(v[3], "UNKNOWN")) next.status = K230_MAT_UNKNOWN;
    else if (!strcmp(v[3], "UNAVAILABLE")) next.status = K230_MAT_UNAVAILABLE;
    else next.status = K230_MAT_ERROR;
    next.seq = f->seq; next.frame_id = frame_id; next.timestamp_ms = timestamp_ms;
    next.received_ms = now_ms; next.camera_id = r->camera_id;
    next.width = r->width; next.height = r->height;
    next.is_simulated = (uint8_t)integer(v[11]); next.complete = (uint8_t)integer(v[12]);
    next.count = (uint8_t)integer(v[13]); next.fresh = 1;
    next.usable = next.status == K230_MAT_OK && !next.is_simulated;
    for (i = 0; i < next.count; ++i) {
        const char *const *o = v + 16u + i * 9u;
        k230_material_object *item = &next.objects[i];
        item->color_id = (uint8_t)integer(o[0]);
        item->u_px = (float)strtod(o[1], NULL); item->v_px = (float)strtod(o[2], NULL);
        item->semimajor_px = (float)strtod(o[3], NULL); item->semiminor_px = (float)strtod(o[4], NULL);
        item->angle_rad = (float)strtod(o[5], NULL); item->has_score = strcmp(o[6], "-") != 0;
        if (item->has_score) item->score_hint = (float)strtod(o[6], NULL);
        item->stable = (uint8_t)integer(o[7]); item->geometry_valid = 1;
        if (!stored_geometry(item, r->width, r->height)) {
            ++r->rejected_snapshots;
            invalidate(r, K230_MAT_UNKNOWN);
            return 0;
        }
        if (!item->stable || !item->color_id) next.usable = 0;
    }
    /* Value copy: no pointers into parser's reusable line buffer survive. This
       is one-owner coherence, not a claim of lock-free multi-task atomicity. */
    r->latest = next;
    r->have_snapshot = 1;
    return 1;
}

int k230_material_read(k230_material_receiver *r, uint32_t now_ms,
                       k230_material_snapshot *out) {
    k230_expire(&r->parser, now_ms);
    if (r->have_snapshot && now_ms - r->latest.received_ms > r->ttl_ms)
        invalidate(r, K230_MAT_STALE);
    *out = r->latest;
    return out->usable;
}

int k230_material_reset(k230_material_receiver *r, int motors_disabled, const char *new_source_session) {
    if (!motors_disabled || !r->configured || !session_valid(new_source_session) ||
        !strcmp(new_source_session, r->source_session) || !k230_reset_session(&r->parser, motors_disabled)) return 0;
    memcpy(r->source_session, new_source_session, 33);
    r->have_frame = 0;
    invalidate(r, K230_MAT_INIT);
    return 1;
}
