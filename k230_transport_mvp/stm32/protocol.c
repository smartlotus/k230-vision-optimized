#include "protocol.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

static int uint_value(const char *s, uint32_t maximum, uint32_t *out) {
    uint32_t n = 0;
    if (!*s) return 0;
    for (; *s; ++s) {
        unsigned d;
        if (*s < '0' || *s > '9') return 0;
        d = (unsigned)(*s - '0');
        if (n > maximum / 10u || (n == maximum / 10u && d > maximum % 10u)) return 0;
        n = n * 10u + d;
    }
    *out = n;
    return 1;
}
static int member(const char *s, const char *const *values, unsigned count) {
    unsigned i;
    for (i = 0; i < count; ++i) if (!strcmp(s, values[i])) return 1;
    return 0;
}
static int numeric(const char *s, double *value, int *missing) {
    const char *p;
    char *end;
    *missing = !strcmp(s, "-");
    if (*missing) return 1;
    if (!*s) return 0;
    for (p = s; *p; ++p) if (!strchr("0123456789+-.eE", *p)) return 0;
    *value = strtod(s, &end);
    return end != s && !*end && isfinite(*value) && fabs(*value) <= 10000000.0;
}
static int hex(char c);
static int escaped(const char *s) {
    /* Validate decoded UTF-8 as Python does, retaining the encoded token. */
    unsigned remaining = 0;
    uint32_t codepoint = 0, minimum = 0;
    while (*s) {
        unsigned byte = (unsigned char)*s++;
        if (byte == '%') {
            int a, b;
            if (!s[0] || !s[1]) return 0;
            a = hex(s[0]); b = hex(s[1]);
            if (a < 0 || b < 0) return 0;
            byte = (unsigned)((a<<4)|b);
            s += 2;
        }
        if (remaining) {
            if (byte < 0x80 || byte > 0xBF) return 0;
            codepoint = (codepoint<<6) | (byte & 0x3F);
            if (!--remaining && (codepoint < minimum || codepoint > 0x10FFFF ||
                                 (codepoint >= 0xD800 && codepoint <= 0xDFFF))) return 0;
        } else if (byte > 0x7F) {
            if (byte >= 0xC2 && byte <= 0xDF) {
                remaining = 1; codepoint = byte & 0x1F; minimum = 0x80;
            } else if (byte >= 0xE0 && byte <= 0xEF) {
                remaining = 2; codepoint = byte & 0x0F; minimum = 0x800;
            } else if (byte >= 0xF0 && byte <= 0xF4) {
                remaining = 3; codepoint = byte & 0x07; minimum = 0x10000;
            } else {
                return 0;
            }
        }
    }
    return remaining == 0;
}
static int material(const k230_frame *f) {
    static const char *const statuses[] = {"OK", "NOT_FOUND", "UNKNOWN", "UNAVAILABLE", "ERROR"};
    const char *const *v = f->fields;
    uint32_t n, count, width, height;
    unsigned i, j;
    int positive;
    if (f->count < 16 || strcmp(v[2], "1") || !member(v[3], statuses, 5)) return 0;
    if ((strcmp(v[4], "CSI1") && strcmp(v[4], "CSI2")) || strcmp(v[10], "IMAGE_PX")) return 0;
    if (!uint_value(v[5], UINT32_MAX, &n) || !uint_value(v[6], UINT32_MAX, &n)) return 0;
    if (strlen(v[7]) != 64) return 0;
    for (i = 0; i < 64; ++i) if (!strchr("0123456789abcdef", v[7][i])) return 0;
    if (strlen(v[15]) != 32) return 0;
    for (i = 0; i < 32; ++i) if (!strchr("0123456789abcdef", v[15][i])) return 0;
    if (!uint_value(v[8], 4096u, &width) || !width || !uint_value(v[9], 4096u, &height) || !height) return 0;
    for (i = 11; i <= 12; ++i) if (strcmp(v[i], "0") && strcmp(v[i], "1")) return 0;
    if (!uint_value(v[13], 3u, &count) || f->count != 16u + count * 9u || !escaped(v[14])) return 0;
    positive = !strcmp(v[3], "OK");
    if (positive != (count > 0)) return 0;
    if ((!strcmp(v[12], "1")) != (positive || !strcmp(v[3], "NOT_FOUND"))) return 0;
    if (!positive && strcmp(v[3], "NOT_FOUND") && !strcmp(v[14], "-")) return 0;
    for (i = 0; i < count; ++i) {
        const char *const *o = v + 16u + i * 9u;
        double values[6], u, y, a, b, angle, dx, dy;
        int missing[6];
        if (!uint_value(o[0], 6u, &n)) return 0;
        for (j = 0; j < 6; ++j) {
            if (!numeric(o[j+1], &values[j], &missing[j]) || (missing[j] && j < 5)) return 0;
        }
        if ((strcmp(o[7], "0") && strcmp(o[7], "1")) || strcmp(o[8], "1")) return 0;
        u = values[0]; y = values[1]; a = values[2]; b = values[3]; angle = values[4];
        if (!(b > 0 && a >= b && a <= (width > height ? width : height))) return 0;
        if (angle < 0 || angle >= 3.14159265358979323846 || u < 0 || u >= width || y < 0 || y >= height) return 0;
        dx = hypot(a*cos(angle), b*sin(angle)); dy = hypot(a*sin(angle), b*cos(angle));
        if (u-dx < -0.0001 || y-dy < -0.0001 || u+dx > width-1+0.0001 || y+dy > height-1+0.0001) return 0;
        if (!missing[5] && (values[5] < 0 || values[5] > 100)) return 0;
    }
    return 1;
}
static int validate(k230_frame *f) {
    static const char *const modes[] = {"IDLE", "QR", "COLOR", "RING", "LANE", "TOP", "SELFTEST"};
    static const char *const acks[] = {"ACK", "BUSY", "DONE", "ERROR", "UNKNOWN", "STOPPED", "PONG"};
    static const char *const statuses[] = {"OK", "NOT_FOUND", "UNCALIBRATED", "UNAVAILABLE", "ERROR"};
    static const char *const kinds[] = {"QR", "COLOR", "RING", "LANE", "TOP"};
    static const char *const coords[] = {"IMAGE_PX", "ROBOT_MM", "WORLD_MM", "NONE"};
    uint32_t n;
    const char *const *v = f->fields;
    unsigned i;
    if (!uint_value(v[1], 65535u, &n)) return 0;
    f->seq = (uint16_t)n;
    if (!strcmp(v[0], "C")) {
        f->type = 'C';
        if (f->count != 6) return 0;
        if (!strcmp(v[2], "MODE")) {
            if (!member(v[3], modes, 7)) return 0;
            if (strcmp(v[4], "-") && (!uint_value(v[4], 6u, &n) || n < 1u)) return 0;
            return 1;
        }
        if (strcmp(v[2], "STOP") && strcmp(v[2], "PING")) return 0;
        return !strcmp(v[3], "-") && !strcmp(v[4], "-") && !strcmp(v[5], "-");
    }
    if (!strcmp(v[0], "A")) {
        f->type = 'A';
        return f->count == 5 && uint_value(v[2], 65535u, &n) && member(v[3], acks, 7) && escaped(v[4]);
    }
    if (!strcmp(v[0], "V")) {
        double values[6];
        int missing[6];
        f->type = 'V';
        if (f->count != 24 || !member(v[2], statuses, 5) || !member(v[3], kinds, 5) || !member(v[8], coords, 4)) return 0;
        if ((!uint_value(v[5], UINT32_MAX, &n) && (strcmp(v[5], "-") || !strcmp(v[2], "OK"))) || !uint_value(v[6], UINT32_MAX, &n)) return 0;
        for (i = 0; i < 6; ++i) if (!numeric(v[9+i], &values[i], &missing[i])) return 0;
        for (i = 15; i <= 16; ++i) if (strcmp(v[i], "0") && strcmp(v[i], "1")) return 0;
        if (strcmp(v[21], "0") && strcmp(v[21], "1")) return 0;
        if (!escaped(v[20]) || !escaped(v[22]) || !escaped(v[23])) return 0;
        if (!missing[5] && (values[5] < 0 || values[5] > 100)) return 0;
        if ((strcmp(v[2], "OK") || !strcmp(v[8], "IMAGE_PX") || !strcmp(v[8], "NONE")) && (!missing[2] || !missing[3])) return 0;
        if (!strcmp(v[2], "OK") && (!strcmp(v[8], "ROBOT_MM") || !strcmp(v[8], "WORLD_MM")) && (missing[2] || missing[3])) return 0;
        return 1;
    }
    if (!strcmp(v[0], "S")) {
        f->type = 'S';
        return material(f);
    }
    return 0;
}
void k230_init(k230_parser *p, uint32_t timeout_ms) {
    memset(p, 0, sizeof(*p));
    p->timeout_ms = timeout_ms;
}
int k230_reset_session(k230_parser *p, int motors_disabled) {
    if (!motors_disabled) return 0;
    p->length = 0;
    p->has_seq = 0;
    return 1;
}
void k230_expire(k230_parser *p, uint32_t now_ms) {
    if (p->length && now_ms - p->started_ms > p->timeout_ms) {
        p->length = 0;
        ++p->rejected;
    }
}
static int hex(char c) {
    if (c >= '0' && c <= '9') return c-'0';
    if (c >= 'A' && c <= 'F') return c-'A'+10;
    return -1;
}
int k230_feed(k230_parser *p, uint8_t byte, uint32_t now_ms) {
    size_t i, start;
    unsigned checksum = 0;
    int h1, h2;
    k230_expire(p, now_ms);
    if (byte == '$') {
        p->length = 1;
        p->line[0] = '$';
        p->started_ms = now_ms;
        return 0;
    }
    if (!p->length) return 0;
    if (p->length >= K230_MAX_LINE) goto reject;
    p->line[p->length++] = (char)byte;
    if (byte != '\n') return 0;
    if (p->length < 7 || p->line[p->length-4] != '*') goto reject;
    h1 = hex(p->line[p->length-3]); h2 = hex(p->line[p->length-2]);
    if (h1 < 0 || h2 < 0) goto reject;
    for (i = 1; i < p->length-4; ++i) checksum ^= (unsigned char)p->line[i];
    if (checksum != (unsigned)((h1<<4)|h2)) goto reject;
    p->line[p->length-4] = '\0';
    p->frame.count = 0;
    start = 1;
    for (i = 1; i <= p->length-4; ++i) {
        char c = p->line[i];
        if (!c && i != p->length-4) goto reject;
        if (c == ',' || !c) {
            size_t limit = p->frame.count == 20 && p->line[1] == 'V' ? 384u : 96u;
            if (i == start || i-start > limit || p->frame.count >= K230_MAX_FIELDS) goto reject;
            p->frame.fields[p->frame.count++] = p->line+start;
            p->line[i] = '\0';
            start = i+1;
        } else if ((unsigned char)c < 32 || (unsigned char)c > 126 || c == '*' || c == '$') goto reject;
    }
    if (p->frame.count < 2 || !validate(&p->frame)) goto reject;
    if (p->has_seq && p->frame.seq <= p->last_seq) goto reject;
    p->last_seq = p->frame.seq;
    p->has_seq = 1;
    p->length = 0;
    return 1;
reject:
    p->length = 0;
    ++p->rejected;
    return 0;
}
