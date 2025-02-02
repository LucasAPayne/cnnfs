#include "log.h"
#include "types.h"

#include <stdarg.h>
#include <stdio.h>

global const char* level_strs[] = {"TRACE", "DEBUG", "INFO", "WARN", "ERROR", "FATAL", "SUCCESS", "FAILURE"};
global const char* level_colors[] = {WHITE, BCYAN, BBLUE, BYELLOW, BRED, REDB, BGREEN, BRED};

void log_log(LogLevel level, const char* fmt, ...)
{
    va_list args;
    va_start(args, fmt);
    fprintf(stderr, "[%s%s%s]: ", level_colors[level], level_strs[level], COLOR_RESET);
    vfprintf(stderr, fmt, args);
    va_end(args);
}
