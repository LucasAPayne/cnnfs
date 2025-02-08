#pragma once

#include "color.h"

// TODO(lucas): Printing to file instead of or in addition to console
// TODO(lucas): Silence logs, or set default level that silences all logs below that level

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

#define LOG_FILE_PATH YELLOW"%s"COLOR_RESET
#define LOG_FUNC_NAME CYAN"%s"COLOR_RESET
#define LOG_LINE_NUM BWHITE"%d"COLOR_RESET

// To report the file, line number, and function name where a log occurs, use these macros, e.g.,
// log_error(LOG_POS "Error number: %d", LOG_POS_ARGS, number);
#define LOG_POS "[" LOG_FILE_PATH ":" LOG_LINE_NUM "] in " LOG_FUNC_NAME ": "
#define LOG_POS_ARGS __FILE__, __LINE__, __func__

#ifdef CNNFS_DEBUG
    #define ASSERT(expr, msg, ...) if(!(expr)) {*(int *)0 = 0; log_error(LOG_POS msg, LOG_POS_ARGS, __VA_ARGS__);}
#else
    // Define as cast to void to prevent unused expression warnings.
    #define ASSERT(expr, msg, ...) (void)(expr); (void)msg
#endif
