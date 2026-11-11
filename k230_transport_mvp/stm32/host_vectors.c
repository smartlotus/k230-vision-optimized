/* Host byte-stream runner for cross-language fixture comparison. */
#include "protocol.h"
#include <stdio.h>
#include <string.h>
static int feed_text(k230_parser *p, const char *s, uint32_t time) {
    int valid = 0;
    while (*s) valid += k230_feed(p, (uint8_t)*s++, time);
    return valid;
}
static int selftest(void) {
    k230_parser p;
    unsigned i;
    k230_init(&p, 10);
    feed_text(&p, "$C,1,", 0);
    if (feed_text(&p, "PING,-,-,-*63\n", 11)) return 1;
    if (feed_text(&p, "$C,1,PING,-,-,-*63\n", 12) != 1) return 2;
    if (feed_text(&p, "$C,1,PING,-,-,-*63\n", 13)) return 3;
    if (k230_reset_session(&p, 0)) return 4;
    if (!k230_reset_session(&p, 1)) return 5;
    k230_feed(&p, '$', 14);
    for (i = 0; i < 800; ++i) k230_feed(&p, 'x', 14);
    if (p.length > K230_MAX_LINE) return 6;
    if (feed_text(&p, "$C,1,PING,-,-,-*63\n", 15) != 1) return 7;
    k230_reset_session(&p, 1);
    feed_text(&p, "$C,1,", UINT32_MAX-5u);
    if (feed_text(&p, "PING,-,-,-*63\n", 7)) return 8;
    puts("SELFTEST_OK");
    return 0;
}
int main(int argc, char **argv) {
    k230_parser parser;
    int c;
    if (argc > 1 && !strcmp(argv[1], "--selftest")) return selftest();
    k230_init(&parser, 250);
    while ((c = getchar()) != EOF) {
        if (k230_feed(&parser, (uint8_t)c, 0))
            printf("%c,%u\n", parser.frame.type, (unsigned)parser.frame.seq);
    }
    return ferror(stdin) ? 1 : 0;
}
