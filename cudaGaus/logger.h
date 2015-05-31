#define TRACE_LEVEL 1
#define DEBUG_LEVEL 2
#define INFO_LEVEL  3
#define WARN_LEVEL  4
#define ERROR_LEVEL 5
#define FATAL_LEVEL 6

#define LOG(code, log_level) if (LOGGER_LEVEL <= log_level) code

#define TRACE(code) LOG(code, TRACE_LEVEL)
#define DEBUG(code) LOG(code, DEBUG_LEVEL)
#define INFO(code)  LOG(code, INFO_LEVEL)
#define WARN(code)  LOG(code, WARN_LEVEL)
#define ERROR(code) LOG(code, ERROR_LEVEL)
#define FATAL(code) LOG(code, FATAL_LEVEL)