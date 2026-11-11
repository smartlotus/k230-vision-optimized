/* PC-only receiver fixture driver. It opens no UART and controls no hardware. */
#include "material_receiver.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int digit(char c) {
    if (c >= '0' && c <= '9') return c-'0';
    if (c >= 'a' && c <= 'f') return c-'a'+10;
    return -1;
}

int main(void) {
    k230_material_receiver receiver;
    k230_material_snapshot value;
    char command[4096];
    uint32_t current_ms = 0;
    unsigned i;
    if (!k230_material_init(&receiver, 2, 640, 360,
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb", 100, 50)) return 1;
    while (fgets(command, sizeof(command), stdin)) {
        char *end, *cursor;
        uint32_t now;
        unsigned accepted = 0;
        if (command[1] != ' ') return 2;
        now = (uint32_t)strtoul(command+2, &end, 10);
        if (command[0] == 'F') {
            if (*end++ != ' ') return 3;
            cursor = end;
            while (*cursor && *cursor != '\r' && *cursor != '\n') {
                int a = digit(cursor[0]), b = digit(cursor[1]);
                if (a < 0 || b < 0) return 4;
                accepted += (unsigned)k230_material_feed(&receiver, (uint8_t)((a<<4)|b), now);
                cursor += 2;
            }
        } else if (command[0] == 'R') {
            char *session = *end == ' ' ? end+1 : end;
            session[strcspn(session, "\r\n")] = '\0';
            accepted = (unsigned)k230_material_reset(&receiver, now == 1, session);
            now = current_ms;
        } else if (command[0] != 'P') return 5;
        current_ms = now;
        k230_material_read(&receiver, now, &value);
        printf("{\"accepted\":%u,\"status\":%u,\"fresh\":%u,\"usable\":%u,\"count\":%u,"
               "\"frame_id\":%lu,\"timestamp_ms\":%lu,\"received_ms\":%lu,\"rejected\":%lu,"
               "\"receiver_bytes\":%lu,\"objects\":[", accepted, (unsigned)value.status,
               (unsigned)value.fresh, (unsigned)value.usable, (unsigned)value.count,
               (unsigned long)value.frame_id, (unsigned long)value.timestamp_ms,
               (unsigned long)value.received_ms,
               (unsigned long)(receiver.rejected_snapshots+receiver.parser.rejected),
               (unsigned long)sizeof(receiver));
        for (i = 0; i < value.count; ++i) {
            const k230_material_object *o = &value.objects[i];
            printf("%s{\"color_id\":%u,\"u_px\":%.6f,\"v_px\":%.6f,\"semimajor_px\":%.6f,"
                   "\"semiminor_px\":%.6f,\"angle_rad\":%.6f,\"stable\":%u,\"geometry_valid\":%u,"
                   "\"has_score\":%u,\"score_hint\":%.6f}", i ? "," : "", (unsigned)o->color_id,
                   o->u_px, o->v_px, o->semimajor_px, o->semiminor_px, o->angle_rad,
                   (unsigned)o->stable, (unsigned)o->geometry_valid, (unsigned)o->has_score, o->score_hint);
        }
        puts("]}");
    }
    return ferror(stdin) ? 6 : 0;
}
