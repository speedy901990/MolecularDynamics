#ifndef LOG_H
#define LOG_H

#include "Global.h"

class Log {
 public:
  static Log * instance();
  void toConsole(int err, string classNamestring, string functionName, int line, string notes = "-");
  void toFile(int err, string classNamestring, string functionName, int line, string notes = "-");

 private:
  Log();
  Log(Log const&);
  static Log * pInstance;
  void operator=(Log const&);
  string handleError(int err);
};

#endif /* LOG_H */
