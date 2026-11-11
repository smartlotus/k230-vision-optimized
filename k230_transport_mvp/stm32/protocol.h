#ifndef K230_MVP_PROTOCOL_H
#define K230_MVP_PROTOCOL_H
#include <stddef.h>
#include <stdint.h>
#define K230_MAX_LINE 768u
#define K230_MAX_FIELDS 43u /* S v1 header + three nine-field objects */
typedef struct {
    char type;
    uint16_t seq;
    unsigned count;
    const char *fields[K230_MAX_FIELDS]; /* valid only until next feed */
} k230_frame;
typedef struct {
    char line[K230_MAX_LINE + 1u];
    size_t length;
    uint32_t started_ms;
    uint32_t rejected;
    uint16_t last_seq;
    unsigned has_seq;
    uint32_t timeout_ms;
    k230_frame frame;
} k230_parser;
void k230_init(k230_parser *parser, uint32_t timeout_ms);
/* Explicit new protocol session only with independently disabled motors. */
int k230_reset_session(k230_parser *parser, int motors_disabled);
void k230_expire(k230_parser *parser, uint32_t now_ms);
/* Returns 1 on a complete valid new frame, otherwise 0. No motor side effects. */
int k230_feed(k230_parser *parser, uint8_t byte, uint32_t now_ms);
#endif
