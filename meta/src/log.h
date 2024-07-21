#pragma once

#include "color.h"

// TODO(lucas): Printing to file instead of or in addition to console
// TODO(lucas): Silence logs, or set default level that silences all logs below that level

#if SIM86_DEBUG
    #define ASSERT(expression) if(!(expression)) {*(int *)0 = 0;}
#else
    #define ASSERT
#endif

typedef enum LogLevel
{
    LOG_TRACE = 0,
    LOG_DEBUG,
    LOG_INFO,
    LOG_WARN,
    LOG_ERROR,
    LOG_FATAL,
    LOG_SUCCESS,
    LOG_FAILURE
} LogLevel;

void log_log(LogLevel level, const char* fmt, ...);

#define log_trace(...) log_log(LOG_TRACE, __VA_ARGS__)
#define log_debug(...) log_log(LOG_DEBUG, __VA_ARGS__)
#define log_info(...)  log_log(LOG_INFO,  __VA_ARGS__)
#define log_warn(...)  log_log(LOG_WARN,  __VA_ARGS__)
#define log_error(...) log_log(LOG_ERROR, __VA_ARGS__)
#define log_fatal(...) log_log(LOG_FATAL, __VA_ARGS__)

#define log_success(...) log_log(LOG_SUCCESS, __VA_ARGS__)
#define log_failure(...) log_log(LOG_FAILURE, __VA_ARGS__)

#define FILE_PATH YELLOW"%s"COLOR_RESET
