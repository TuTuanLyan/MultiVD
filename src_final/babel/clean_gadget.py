import re

# keywords up to C11 and C++17; immutable set
keywords = frozenset(['__asm', '__builtin', '__cdecl', '__declspec', '__except', '__export', '__far16', '__far32',
                      '__fastcall', '__finally', '__import', '__inline', '__int16', '__int32', '__int64', '__int8',
                      '__leave', '__optlink', '__packed', '__pascal', '__stdcall', '__system', '__thread', '__try',
                      '__unaligned', '_asm', '_Builtin', '_Cdecl', '_declspec', '_except', '_Export', '_Far16',
                      '_Far32', '_Fastcall', '_finally', '_Import', '_inline', '_int16', '_int32', '_int64',
                      '_int8', '_leave', '_Optlink', '_Packed', '_Pascal', '_stdcall', '_System', '_try', 'alignas',
                      'alignof', 'and', 'and_eq', 'asm', 'auto', 'bitand', 'bitor', 'bool', 'break', 'case',
                      'catch', 'char', 'char16_t', 'char32_t', 'class', 'compl', 'const', 'const_cast', 'constexpr',
                      'continue', 'decltype', 'default', 'delete', 'do', 'double', 'dynamic_cast', 'else', 'enum',
                      'explicit', 'export', 'extern', 'false', 'final', 'float', 'for', 'friend', 'goto', 'if',
                      'inline', 'int', 'long', 'mutable', 'namespace', 'new', 'noexcept', 'not', 'not_eq', 'nullptr',
                      'operator', 'or', 'or_eq', 'override', 'private', 'protected', 'public', 'register',
                      'reinterpret_cast', 'return', 'short', 'signed', 'sizeof', 'static', 'static_assert',
                      'static_cast', 'struct', 'switch', 'template', 'this', 'thread_local', 'throw', 'true', 'try',
                      'typedef', 'typeid', 'typename', 'union', 'unsigned', 'using', 'virtual', 'void', 'volatile',
                      'wchar_t', 'while', 'xor', 'xor_eq', 'NULL', 'StrNCat', 'getaddrinfo', '_ui64toa', 'fclose',
                      'pthread_mutex_lock', 'gets_s', 'sleep', '_ui64tot', 'freopen_s', '_ui64tow', 'send', 'lstrcat',
                      'HMAC_Update', '__fxstat', 'StrCatBuff', '_mbscat', '_mbstok_s', '_cprintf_s',
                      'ldap_search_init_page', 'memmove_s', 'ctime_s', 'vswprintf', 'vswprintf_s', '_snwprintf',
                      '_gmtime_s', '_tccpy', '*RC6*', '_mbslwr_s', 'random', '__wcstof_internal', '_wcslwr_s',
                      '_ctime32_s', 'wcsncat*', 'MD5_Init', '_ultoa', 'snprintf', 'memset', 'syslog', '_vsnprintf_s',
                      'HeapAlloc', 'pthread_mutex_destroy', 'ChangeWindowMessageFilter', '_ultot', 'crypt_r',
                      '_strupr_s_l', 'LoadLibraryExA', '_strerror_s', 'LoadLibraryExW', 'wvsprintf', 'MoveFileEx',
                      '_strdate_s', 'SHA1', 'sprintfW', 'StrCatNW', '_scanf_s_l', 'pthread_attr_init', '_wtmpnam_s',
                      'snscanf', '_sprintf_s_l', 'dlopen', 'sprintfA', 'timed_mutex', 'OemToCharA', 'ldap_delete_ext',
                      'sethostid', 'popen', 'OemToCharW', '_gettws', 'vfork', '_wcsnset_s_l', 'sendmsg', '_mbsncat',
                      'wvnsprintfA', 'HeapFree', '_wcserror_s', 'realloc', '_snprintf*', 'wcstok', '_strncat*',
                      'StrNCpy', '_wasctime_s', 'push*', '_lfind_s', 'CC_SHA512', 'ldap_compare_ext_s', 'wcscat_s',
                      'strdup', '_chsize_s', 'sprintf_s', 'CC_MD4_Init', 'wcsncpy', '_wfreopen_s', '_wcsupr_s',
                      '_searchenv_s', 'ldap_modify_ext_s', '_wsplitpath', 'CC_SHA384_Final', 'MD2', 'RtlCopyMemory',
                      'lstrcatW', 'MD4', 'MD5', '_wcstok_s_l', '_vsnwprintf_s', 'ldap_modify_s', 'strerror',
                      '_lsearch_s', '_mbsnbcat_s', '_wsplitpath_s', 'MD4_Update', '_mbccpy_s', '_strncpy_s_l',
                      '_snprintf_s', 'CC_SHA512_Init', 'fwscanf_s', '_snwprintf_s', 'CC_SHA1', 'swprintf', 'fprintf',
                      'EVP_DigestInit_ex', 'strlen', 'SHA1_Init', 'strncat', '_getws_s', 'CC_MD4_Final', 'wnsprintfW',
                      'lcong48', 'lrand48', 'write', 'HMAC_Init', '_wfopen_s', 'wmemchr', '_tmakepath', 'wnsprintfA',
                      'lstrcpynW', 'scanf_s', '_mbsncpy_s_l', '_localtime64_s', 'fstream.open', '_wmakepath',
                      'Connection.open', '_tccat', 'valloc', 'setgroups', 'unlink', 'fstream.put', 'wsprintfA',
                      '*SHA1*', '_wsearchenv_s', 'ualstrcpyA', 'CC_MD5_Update', 'strerror_s', 'HeapCreate',
                      'ualstrcpyW', '__xstat', '_wmktemp_s', 'StrCatChainW', 'ldap_search_st', '_mbstowcs_s_l',
                      'ldap_modify_ext', '_mbsset_s', 'strncpy_s', 'move', 'execle', 'StrCat', 'xrealloc', 'wcsncpy_s',
                      '_tcsncpy*', 'execlp', 'RIPEMD160_Final', 'ldap_search_s', 'EnterCriticalSection', '_wctomb_s_l',
                      'fwrite', '_gmtime64_s', 'sscanf_s', 'wcscat', '_strupr_s', 'wcrtomb_s', 'VirtualLock',
                      'ldap_add_ext_s', '_mbscpy', '_localtime32_s', 'lstrcpy', '_wcsncpy*', 'CC_SHA1_Init', '_getts',
                      '_wfopen', '__xstat64', 'strcoll', '_fwscanf_s_l', '_mbslwr_s_l', 'RegOpenKey', 'makepath',
                      'seed48', 'CC_SHA256', 'sendto', 'execv', 'CalculateDigest', 'memchr', '_mbscpy_s', '_strtime_s',
                      'ldap_search_ext_s', '_chmod', 'flock', '__fxstat64', '_vsntprintf', 'CC_SHA256_Init', '_itoa_s',
                      '__wcserror_s', '_gcvt_s', 'fstream.write', 'sprintf', 'recursive_mutex', 'strrchr',
                      'gethostbyaddr', '_wcsupr_s_l', 'strcspn', 'MD5_Final', 'asprintf', '_wcstombs_s_l', '_tcstok',
                      'free', 'MD2_Final', 'asctime_s', '_alloca', '_wputenv_s', '_wcsset_s', '_wcslwr_s_l',
                      'SHA1_Update', 'filebuf.sputc', 'filebuf.sputn', 'SQLConnect', 'ldap_compare', 'mbstowcs_s',
                      'HMAC_Final', 'pthread_condattr_init', '_ultow_s', 'rand', 'ofstream.put', 'CC_SHA224_Final',
                      'lstrcpynA', 'bcopy', 'system', 'CreateFile*', 'wcscpy_s', '_mbsnbcpy*', 'open', '_vsnwprintf',
                      'strncpy', 'getopt_long', 'CC_SHA512_Final', '_vsprintf_s_l', 'scanf', 'mkdir', '_localtime_s',
                      '_snprintf', '_mbccpy_s_l', 'memcmp', 'final', '_ultoa_s', 'lstrcpyW', 'LoadModule',
                      '_swprintf_s_l', 'MD5_Update', '_mbsnset_s_l', '_wstrtime_s', '_strnset_s', 'lstrcpyA',
                      '_mbsnbcpy_s', 'mlock', 'IsBadHugeWritePtr', 'copy', '_mbsnbcpy_s_l', 'wnsprintf', 'wcscpy',
                      'ShellExecute', 'CC_MD4', '_ultow', '_vsnwprintf_s_l', 'lstrcpyn', 'CC_SHA1_Final', 'vsnprintf',
                      '_mbsnbset_s', '_i64tow', 'SHA256_Init', 'wvnsprintf', 'RegCreateKey', 'strtok_s', '_wctime32_s',
                      '_i64toa', 'CC_MD5_Final', 'wmemcpy', 'WinExec', 'CreateDirectory*', 'CC_SHA256_Update',
                      '_vsnprintf_s_l', 'jrand48', 'wsprintf', 'ldap_rename_ext_s', 'filebuf.open', '_wsystem',
                      'SHA256_Update', '_cwscanf_s', 'wsprintfW', '_sntscanf', '_splitpath', 'fscanf_s', 'strpbrk',
                      'wcstombs_s', 'wscanf', '_mbsnbcat_s_l', 'strcpynA', 'pthread_cond_init', 'wcsrtombs_s',
                      '_wsopen_s', 'CharToOemBuffA', 'RIPEMD160_Update', '_tscanf', 'HMAC', 'StrCCpy',
                      'Connection.connect', 'lstrcatn', '_mbstok', '_mbsncpy', 'CC_SHA384_Update', 'create_directories',
                      'pthread_mutex_unlock', 'CFile.Open', 'connect', '_vswprintf_s_l', '_snscanf_s_l', 'fputc',
                      '_wscanf_s', '_snprintf_s_l', 'strtok', '_strtok_s_l', 'lstrcatA', 'snwscanf',
                      'pthread_mutex_init', 'fputs', 'CC_SHA384_Init', '_putenv_s', 'CharToOemBuffW',
                      'pthread_mutex_trylock', '__wcstoul_internal', '_memccpy', '_snwprintf_s_l', '_strncpy*',
                      'wmemset', 'MD4_Init', '*RC4*', 'strcpyW', '_ecvt_s', 'memcpy_s', 'erand48', 'IsBadHugeReadPtr',
                      'strcpyA', 'HeapReAlloc', 'memcpy', 'ldap_rename_ext', 'fopen_s', 'srandom', '_cgetws_s',
                      '_makepath', 'SHA256_Final', 'remove', '_mbsupr_s', 'pthread_mutexattr_init',
                      '__wcstold_internal', 'StrCpy', 'ldap_delete', 'wmemmove_s', '_mkdir', 'strcat', '_cscanf_s_l',
                      'StrCAdd', 'swprintf_s', '_strnset_s_l', 'close', 'ldap_delete_ext_s', 'ldap_modrdn', 'strchr',
                      '_gmtime32_s', '_ftcscat', 'lstrcatnA', '_tcsncat', 'OemToChar', 'mutex', 'CharToOem', 'strcpy_s',
                      'lstrcatnW', '_wscanf_s_l', '__lxstat64', 'memalign', 'MD2_Init', 'StrCatBuffW', 'StrCpyN',
                      'CC_MD5', 'StrCpyA', 'StrCatBuffA', 'StrCpyW', 'tmpnam_r', '_vsnprintf', 'strcatA', 'StrCpyNW',
                      '_mbsnbset_s_l', 'EVP_DigestInit', '_stscanf', 'CC_MD2', '_tcscat', 'StrCpyNA', 'xmalloc',
                      '_tcslen', '*MD4*', 'vasprintf', 'strxfrm', 'chmod', 'ldap_add_ext', 'alloca', '_snscanf_s',
                      'IsBadWritePtr', 'swscanf_s', 'wmemcpy_s', '_itoa', '_ui64toa_s', 'EVP_DigestUpdate',
                      '__wcstol_internal', '_itow', 'StrNCatW', 'strncat_s', 'ualstrcpy', 'execvp', '_mbccat',
                      'EVP_MD_CTX_init', 'assert', 'ofstream.write', 'ldap_add', '_sscanf_s_l', 'drand48', 'CharToOemW',
                      'swscanf', '_itow_s', 'RIPEMD160_Init', 'CopyMemory', 'initstate', 'getpwuid', 'vsprintf',
                      '_fcvt_s', 'CharToOemA', 'setuid', 'malloc', 'StrCatNA', 'strcat_s', 'srand', 'getwd',
                      '_controlfp_s', 'olestrcpy', '__wcstod_internal', '_mbsnbcat', 'lstrncat', 'des_*',
                      'CC_SHA224_Init', 'set*', 'vsprintf_s', 'SHA1_Final', '_umask_s', 'gets', 'setstate',
                      'wvsprintfW', 'LoadLibraryEx', 'ofstream.open', 'calloc', '_mbstrlen', '_cgets_s', '_sopen_s',
                      'IsBadStringPtr', 'wcsncat_s', 'add*', 'nrand48', 'create_directory', 'ldap_search_ext',
                      '_i64toa_s', '_ltoa_s', '_cwscanf_s_l', 'wmemcmp', '__lxstat', 'lstrlen',
                      'pthread_condattr_destroy', '_ftcscpy', 'wcstok_s', '__xmknod', 'pthread_attr_destroy',
                      'sethostname', '_fscanf_s_l', 'StrCatN', 'RegEnumKey', '_tcsncpy', 'strcatW', 'AfxLoadLibrary',
                      'setenv', 'tmpnam', '_mbsncat_s_l', '_wstrdate_s', '_wctime64_s', '_i64tow_s', 'CC_MD4_Update',
                      'ldap_add_s', '_umask', 'CC_SHA1_Update', '_wcsset_s_l', '_mbsupr_s_l', 'strstr', '_tsplitpath',
                      'memmove', '_tcscpy', 'vsnprintf_s', 'strcmp', 'wvnsprintfW', 'tmpfile', 'ldap_modify',
                      '_mbsncat*', 'mrand48', 'sizeof', 'StrCatA', '_ltow_s', '*desencrypt*', 'StrCatW', '_mbccpy',
                      'CC_MD2_Init', 'RIPEMD160', 'ldap_search', 'CC_SHA224', 'mbsrtowcs_s', 'update', 'ldap_delete_s',
                      'getnameinfo', '*RC5*', '_wcsncat_s_l', 'DriverManager.getConnection', 'socket', '_cscanf_s',
                      'ldap_modrdn_s', '_wopen', 'CC_SHA256_Final', '_snwprintf*', 'MD2_Update', 'strcpy',
                      '_strncat_s_l', 'CC_MD5_Init', 'mbscpy', 'wmemmove', 'LoadLibraryW', '_mbslen', '*alloc',
                      '_mbsncat_s', 'LoadLibraryA', 'fopen', 'StrLen', 'delete', '_splitpath_s',
                      'CreateFileTransacted*', 'MD4_Final', '_open', 'CC_SHA384', 'wcslen', 'wcsncat', '_mktemp_s',
                      'pthread_mutexattr_destroy', '_snwscanf_s', '_strset_s', '_wcsncpy_s_l', 'CC_MD2_Final',
                      '_mbstok_s_l', 'wctomb_s', 'MySQL_Driver.connect', '_snwscanf_s_l', '*_des_*', 'LoadLibrary',
                      '_swscanf_s_l', 'ldap_compare_s', 'ldap_compare_ext', '_strlwr_s', 'GetEnvironmentVariable',
                      'cuserid', '_mbscat_s', 'strspn', '_mbsncpy_s', 'ldap_modrdn2', 'LeaveCriticalSection',
                      'CopyFile', 'getpwd', 'sscanf', 'creat', 'RegSetValue', 'ldap_modrdn2_s', 'CFile.Close',
                      '*SHA_1*', 'pthread_cond_destroy', 'CC_SHA512_Update', '*RC2*', 'StrNCatA', '_mbsnbcpy',
                      '_mbsnset_s', 'crypt', 'excel', '_vstprintf', 'xstrdup', 'wvsprintfA', 'getopt', 'mkstemp',
                      '_wcsnset_s', '_stprintf', '_sntprintf', 'tmpfile_s', 'OpenDocumentFile', '_mbsset_s_l',
                      '_strset_s_l', '_strlwr_s_l', 'ifstream.open', 'xcalloc', 'StrNCpyA', '_wctime_s',
                      'CC_SHA224_Update', '_ctime64_s', 'MoveFile', 'chown', 'StrNCpyW', 'IsBadReadPtr', '_ui64tow_s',
                      'IsBadCodePtr', 'getc', 'OracleCommand.ExecuteOracleScalar', 'AccessDataSource.Insert',
                      'IDbDataAdapter.FillSchema', 'IDbDataAdapter.Update', 'GetWindowText*', 'SendMessage',
                      'SqlCommand.ExecuteNonQuery', 'streambuf.sgetc', 'streambuf.sgetn', 'OracleCommand.ExecuteScalar',
                      'SqlDataSource.Update', '_Read_s', 'IDataAdapter.Fill', '_wgetenv', '_RecordsetPtr.Open*',
                      'AccessDataSource.Delete', 'Recordset.Open*', 'filebuf.sbumpc', 'DDX_*', 'RegGetValue',
                      'fstream.read*', 'SqlCeCommand.ExecuteResultSet', 'SqlCommand.ExecuteXmlReader', 'main',
                      'streambuf.sputbackc', 'read', 'm_lpCmdLine', 'CRichEditCtrl.Get*', 'istream.putback',
                      'SqlCeCommand.ExecuteXmlReader', 'SqlCeCommand.BeginExecuteXmlReader', 'filebuf.sgetn',
                      'OdbcDataAdapter.Update', 'filebuf.sgetc', 'SQLPutData', 'recvfrom',
                      'OleDbDataAdapter.FillSchema', 'IDataAdapter.FillSchema', 'CRichEditCtrl.GetLine',
                      'DbDataAdapter.Update', 'SqlCommand.ExecuteReader', 'istream.get', 'ReceiveFrom', '_main',
                      'fgetc', 'DbDataAdapter.FillSchema', 'kbhit', 'UpdateCommand.Execute*', 'Statement.execute',
                      'fgets', 'SelectCommand.Execute*', 'getch', 'OdbcCommand.ExecuteNonQuery', 'CDaoQueryDef.Execute',
                      'fstream.getline', 'ifstream.getline', 'SqlDataAdapter.FillSchema', 'OleDbCommand.ExecuteReader',
                      'Statement.execute*', 'SqlCeCommand.BeginExecuteNonQuery', 'OdbcCommand.ExecuteScalar',
                      'SqlCeDataAdapter.Update', 'sendmessage', 'mysqlpp.DBDriver', 'fstream.peek', 'Receive',
                      'CDaoRecordset.Open', 'OdbcDataAdapter.FillSchema', '_wgetenv_s', 'OleDbDataAdapter.Update',
                      'readsome', 'SqlCommand.BeginExecuteXmlReader', 'recv', 'ifstream.peek', '_Main', '_tmain',
                      '_Readsome_s', 'SqlCeCommand.ExecuteReader', 'OleDbCommand.ExecuteNonQuery', 'fstream.get',
                      'IDbCommand.ExecuteScalar', 'filebuf.sputbackc', 'IDataAdapter.Update', 'streambuf.sbumpc',
                      'InsertCommand.Execute*', 'RegQueryValue', 'IDbCommand.ExecuteReader', 'SqlPipe.ExecuteAndSend',
                      'Connection.Execute*', 'getdlgtext', 'ReceiveFromEx', 'SqlDataAdapter.Update', 'RegQueryValueEx',
                      'SQLExecute', 'pread', 'SqlCommand.BeginExecuteReader', 'AfxWinMain', 'getchar',
                      'istream.getline', 'SqlCeDataAdapter.Fill', 'OleDbDataReader.ExecuteReader',
                      'SqlDataSource.Insert', 'istream.peek', 'SendMessageCallback', 'ifstream.read*',
                      'SqlDataSource.Select', 'SqlCommand.ExecuteScalar', 'SqlDataAdapter.Fill',
                      'SqlCommand.BeginExecuteNonQuery', 'getche', 'SqlCeCommand.BeginExecuteReader', 'getenv',
                      'streambuf.snextc', 'Command.Execute*', '_CommandPtr.Execute*', 'SendNotifyMessage',
                      'OdbcDataAdapter.Fill', 'AccessDataSource.Update', 'fscanf', 'QSqlQuery.execBatch',
                      'DbDataAdapter.Fill', 'cin', 'DeleteCommand.Execute*', 'QSqlQuery.exec', 'PostMessage',
                      'ifstream.get', 'filebuf.snextc', 'IDbCommand.ExecuteNonQuery', 'Winmain', 'fread', 'getpass',
                      'GetDlgItemTextCCheckListBox.GetCheck', 'DISP_PROPERTY_EX', 'pread64', 'Socket.Receive*',
                      'SACommand.Execute*', 'SQLExecDirect', 'SqlCeDataAdapter.FillSchema', 'DISP_FUNCTION',
                      'OracleCommand.ExecuteNonQuery', 'CEdit.GetLine', 'OdbcCommand.ExecuteReader', 'CEdit.Get*',
                      'AccessDataSource.Select', 'OracleCommand.ExecuteReader', 'OCIStmtExecute', 'getenv_s',
                      'DB2Command.Execute*', 'OracleDataAdapter.FillSchema', 'OracleDataAdapter.Fill', 'CComboBox.Get*',
                      'SqlCeCommand.ExecuteNonQuery', 'OracleCommand.ExecuteOracleNonQuery', 'mysqlpp.Query',
                      'istream.read*', 'CListBox.GetText', 'SqlCeCommand.ExecuteScalar', 'ifstream.putback', 'readlink',
                      'CHtmlEditCtrl.GetDHtmlDocument', 'PostThreadMessage', 'CListCtrl.GetItemText',
                      'OracleDataAdapter.Update', 'OleDbCommand.ExecuteScalar', 'stdin', 'SqlDataSource.Delete',
                      'OleDbDataAdapter.Fill', 'fstream.putback', 'IDbDataAdapter.Fill', '_wspawnl', 'fwprintf',
                      'sem_wait', '_unlink', 'ldap_search_ext_sW', 'signal', 'PQclear', 'PQfinish', 'PQexec',
                      'PQresultStatus', 'ifdef', 'endif', 'bool', 'void'])
BASE_KEYWORDS = keywords

JAVA_KEYWORDS = frozenset(['abstract', 'assert', 'boolean', 'break', 'byte', 'case', 'catch', 'char', 'class', 'const',
    'continue', 'default', 'do', 'double', 'else', 'enum', 'extends', 'final', 'finally', 'float', 'for', 'goto', 'if',
    'implements', 'import', 'instanceof', 'int', 'interface', 'long', 'native', 'new', 'package', 'private',
    'protected', 'public', 'return', 'short', 'static', 'strictfp', 'super', 'switch', 'synchronized', 'this',
    'throw', 'throws', 'transient', 'try', 'void', 'volatile', 'while', 'true', 'false', 'null', 'var', 'record',
    'String', 'Object', 'Integer', 'Long', 'Boolean', 'Double', 'Float', 'Character', 'Byte', 'Short', 'Math',
    'System', 'Override', 'Exception', 'Throwable', 'RuntimeException', 'IOException', 'List', 'Map', 'Set',
    'ArrayList', 'HashMap', 'HashSet', 'StringBuilder', 'StringBuffer', 'Thread', 'Runnable', 'Class',
    'out', 'err', 'println', 'print', 'length', 'size', 'get', 'put', 'add', 'equals', 'hashCode', 'toString',
    'append', 'valueOf', 'parseInt', 'getBytes', 'charAt', 'substring', 'indexOf', 'format', 'close', 'open',
    'read', 'write', 'main', 'args'])

PYTHON_KEYWORDS = frozenset(['False', 'None', 'True', 'and', 'as', 'assert', 'async', 'await', 'break', 'class',
    'continue', 'def', 'del', 'elif', 'else', 'except', 'finally', 'for', 'from', 'global', 'if', 'import', 'in',
    'is', 'lambda', 'nonlocal', 'not', 'or', 'pass', 'raise', 'return', 'try', 'while', 'with', 'yield', 'self',
    'cls', 'print', 'len', 'range', 'str', 'int', 'float', 'bool', 'list', 'dict', 'set', 'tuple', 'open', 'type',
    'isinstance', 'super', 'object', 'Exception', 'ValueError', 'TypeError', 'KeyError', 'IndexError', 'enumerate',
    'zip', 'map', 'filter', 'sorted', 'reversed', 'min', 'max', 'sum', 'abs', 'any', 'all', 'iter', 'next',
    'getattr', 'setattr', 'hasattr', 'format', 'join', 'split', 'strip', 'append', 'extend', 'get', 'items',
    'keys', 'values', 'read', 'write', 'close', 'os', 'sys', 're', 'json', 'request', 'response', 'kwargs', 'args'])

JS_KEYWORDS = frozenset(['break', 'case', 'catch', 'class', 'const', 'continue', 'debugger', 'default', 'delete',
    'do', 'else', 'export', 'extends', 'finally', 'for', 'function', 'if', 'import', 'in', 'instanceof', 'let',
    'new', 'return', 'super', 'switch', 'this', 'throw', 'try', 'typeof', 'var', 'void', 'while', 'with', 'yield',
    'async', 'await', 'of', 'static', 'get', 'set', 'true', 'false', 'null', 'undefined', 'NaN', 'Infinity',
    'console', 'log', 'error', 'warn', 'require', 'module', 'exports', 'process', 'window', 'document', 'Object',
    'Array', 'String', 'Number', 'Boolean', 'Promise', 'JSON', 'Math', 'Date', 'Error', 'RegExp', 'Map', 'Set',
    'parse', 'stringify', 'push', 'pop', 'shift', 'slice', 'splice', 'length', 'indexOf', 'join', 'split',
    'replace', 'trim', 'toString', 'then', 'resolve', 'reject', 'res', 'req', 'next', 'callback', 'cb', 'err'])

KEYWORDS_BY_LANG = {
    'c': BASE_KEYWORDS,
    'java': BASE_KEYWORDS | JAVA_KEYWORDS,
    'python': BASE_KEYWORDS | PYTHON_KEYWORDS,
    'js': BASE_KEYWORDS | JS_KEYWORDS,
}

main_set = frozenset({'main'})
main_args = frozenset({'argc', 'argv'})


from .strip_comments import blank_comments
from . import strip_comments as _sc


# ------------------------------------------------------------------------------ literal
# Mỗi literal là một bộ (start, q, a, b, c, end, interps, closed):
#   [start, q)  tiền tố (L / u8 / R của C++, f / rb / ... của Python)
#   [q, a)      dấu mở (" ' ''' """ ` / R"delim( )
#   [a, b)      thân
#   [b, c)      dấu đóng;  [c, end) cờ của regex JS (g, i, ...)
#   interps     vùng MÃ bên trong thân cần giữ: `${...}` của template JS, `{...}` của f-string Python
#   closed      False nếu literal không đóng (chuỗi hỏng, hoặc gadget bị cắt ở Lmax giữa một chuỗi)
_PY_PREFIXES = frozenset(["r", "u", "b", "f", "br", "rb", "fr", "rf", "ur"])   # ur: Python 2
_C_PREFIXES = frozenset(["", "L", "u", "U", "u8"])


def _c_prefix_start(code, i):
    """Vị trí bắt đầu tiền tố L / u / U / u8 đứng sát trước dấu nháy (hoặc chữ R của raw string) ở i."""
    k = i
    while k > 0 and code[k - 1] in "uUL8":
        k -= 1
    if code[k:i] in _C_PREFIXES and (k == 0 or code[k - 1] not in _sc._ID):
        return k
    return i


def _skip_spans(spans):
    """Hàm hỏi 'i có nằm trong span nào không' cho danh sách span đã sắp xếp, gọi với i tăng dần."""
    st = {"k": 0}

    def end_if_inside(i):
        k = st["k"]
        while k < len(spans) and spans[k][1] <= i:
            k += 1
        st["k"] = k
        if k < len(spans) and spans[k][0] <= i:
            return spans[k][1]
        return None
    return end_if_inside


def _char_lits(code, strings, c_rules):
    """Ký tự `'x'` của C/C++/Java: quét NGOÀI các chuỗi đã nhận dạng, đúng luật nhánh ký tự của
    strip_comments (dấu nháy đơn giữa hai chữ số là dấu phân cách chữ số C++14, không phải ký tự)."""
    out, n, i = [], len(code), 0
    inside = _skip_spans(strings)
    while i < n:
        e = inside(i)
        if e is not None:
            i = e
            continue
        if code[i] != "'":
            i += 1
            continue
        if c_rules and i > 0 and code[i - 1] in _sc._DIGIT and i + 1 < n and code[i + 1] in _sc._DIGIT:
            i += 1
            continue
        j, closed = i + 1, False
        while j < n:
            if code[j] == "\\":
                j = _sc._skip_splices(code, j) if c_rules else j
                if j < n and code[j] == "\\":
                    j += 2
                continue
            if code[j] == "'":
                j += 1
                closed = True
                break
            if code[j] == "\n":
                break
            j += 1
        j = min(j, n)
        s = _c_prefix_start(code, i) if c_rules else i
        out.append((s, i, i + 1, j - 1 if closed else j, j, j, [], closed))
        i = j
    return out


def _lits_c(code):
    _, warn = _sc._spans_c(code)
    strings = sorted(warn.get("_strings", []))
    out = []
    for a0, e in strings:
        if code[a0] == "R":                          # raw string C++11: R"delim( ... )delim"
            s = _c_prefix_start(code, a0)
            out.append((s, a0 + 1, a0 + 2, e - 1, e, e, [], True))
            continue
        closed = e - a0 >= 2 and code[e - 1] == '"'
        out.append((_c_prefix_start(code, a0), a0, a0 + 1, e - 1 if closed else e, e, e, [], closed))
    return out + _char_lits(code, strings, c_rules=True)


def _lits_java(code):
    _, warn = _sc._spans_java(code)
    strings = sorted(warn.get("_strings", []))
    out = []
    for a0, e in strings:
        dq = 3 if code.startswith('"""', a0) else 1
        closed = e - a0 >= 2 * dq and code[e - dq:e] == '"' * dq
        out.append((a0, a0, a0 + dq, e - dq if closed else e, e, e, [], closed))
    return out + _char_lits(code, strings, c_rules=False)


def _js_scan_lits(code, i, n, out, in_expr):
    """Quét mã JS từ i, ghi literal vào out. Đệ quy vào `${...}` của template, nên nhận được cả chuỗi,
    template lồng và REGEX nằm trong phần nội suy (bộ quét của strip_comments không nhận regex ở đó:
    `${s.replace(/"/g, "&quot;")}` bị đọc thành chuỗi `"/g, "`). in_expr: dừng ở `}` cân bằng, trả vị trí nó."""
    depth = 0
    while i < n:
        c = code[i]
        if c in "'\"":
            j, closed = i + 1, False
            while j < n:
                if code[j] == "\\":
                    j += 2
                    continue
                if code[j] == c:
                    j += 1
                    closed = True
                    break
                if code[j] == "\n":
                    break
                j += 1
            j = min(j, n)
            out.append((i, i, i + 1, j - 1 if closed else j, j, j, [], closed))
            i = j
            continue
        if c == "`":
            j, closed, interps = i + 1, False, []
            while j < n:
                if code[j] == "\\":
                    j += 2
                    continue
                if code[j] == "`":
                    j += 1
                    closed = True
                    break
                if code[j] == "$" and j + 1 < n and code[j + 1] == "{":
                    e = _js_scan_lits(code, j + 2, n, out, True)
                    interps.append((j, min(e + 1, n)))     # giữ cả `${` và `}`
                    j = e + 1
                    continue
                j += 1
            j = min(j, n)
            out.append((i, i, i + 1, j - 1 if closed else j, j, j, interps, closed))
            i = j
            continue
        if c == "/" and i + 1 < n and code[i + 1] in "/*":   # comment (thường đã xoá trắng từ trước)
            if code[i + 1] == "/":
                e = code.find("\n", i)
                i = n if e < 0 else e
            else:
                e = code.find("*/", i + 2)
                i = n if e < 0 else e + 2
            continue
        if c == "/" and _sc._js_regex_allowed(code, i, []):
            j, incls, ok = i + 1, False, False
            while j < n:
                ch = code[j]
                if ch == "\\":
                    j += 2
                    continue
                if ch == "\n":
                    break                           # regex không bắc qua dòng -> là phép chia
                if ch == "[":
                    incls = True
                elif ch == "]":
                    incls = False
                elif ch == "/" and not incls:
                    ok = True
                    break
                j += 1
            if ok:
                end = j + 1
                while end < n and code[end] in "dgimsuvy":
                    end += 1
                out.append((i, i, i + 1, j, j + 1, end, [], True))
                i = end
                continue
        if in_expr:
            if c == "{":
                depth += 1
            elif c == "}":
                if depth == 0:
                    return i
                depth -= 1
        i += 1
    return n


def _lits_js(code):
    out = []
    _js_scan_lits(code, 0, len(code), out, False)
    return out


def _py_string_at(code, start, q, n):
    """Chuỗi Python bắt đầu (tiền tố) ở start, dấu nháy ở q. Luật tới Python 3.11: tìm nháy đóng như chuỗi
    thường (bỏ qua ngoặc nhọn), `\\` luôn nuốt ký tự sau (kể cả raw). Không đóng: chuỗi một nháy dừng ở cuối
    dòng; ba nháy kéo tới HẾT gadget (gadget bị cắt ở Lmax giữa một chuỗi)."""
    ch = code[q]
    dq = 3 if code.startswith(ch * 3, q) else 1
    a, j = q + dq, q + dq
    while j < n:
        if code[j] == "\\":
            j += 2
            continue
        if code.startswith(ch * dq, j):
            return a, j, j + dq, True
        if dq == 1 and code[j] == "\n":
            return a, j, j, False
        j += 1
    return a, n, n, False


def _fstring_expr(code, i, b, out, nested):
    """Biểu thức f-string từ i (ngay sau `{`) tới `}` khớp. Ghi vùng GIỮ vào out, gồm cả `{` và `}` để hai
    biểu thức liền nhau không dính thành "kiểu + tên" trong regex rx_var; phần sau `!r` / `:` là định dạng
    (chữ, xoá trắng), trừ `{...}` lồng trong đó. Trả vị trí sau `}`."""
    depth, m = 0, i
    while m < b:
        c = code[m]
        if c in "'\"":                              # chuỗi lồng (khác kiểu nháy với chuỗi ngoài)
            s = m
            while s > i and code[s - 1].isalnum():
                s -= 1
            if code[s:m].lower() not in _PY_PREFIXES or (s > i and code[s - 1] == "_"):
                s = m
            a2, b2, e2, cl2 = _py_string_at(code, s, m, b)
            nested.append((s, m, a2, b2, e2, e2, [], cl2))
            m = e2
            continue
        if c in "([{":
            depth += 1
        elif c in ")]}":
            if depth == 0:
                out.append((i - 1, m + 1))
                return m + 1
            depth -= 1
        elif depth == 0 and (c == ":" or (c == "!" and m + 2 < b and code[m + 1] in "rsa" and code[m + 2] in ":}")):
            out.append((i - 1, m))
            m += 2 if c == "!" else 0
            while m < b:
                if code[m] == "{":
                    m = _fstring_expr(code, m + 1, b, out, nested)
                    continue
                if code[m] == "}":
                    out.append((m, m + 1))
                    return m + 1
                m += 1
            return m
        m += 1
    out.append((i - 1, m))
    return m


def _lits_python(code):
    out, n, i = [], len(code), 0
    while i < n:
        c = code[i]
        if c == "#":                                # comment (thường đã xoá trắng từ trước)
            e = code.find("\n", i)
            i = n if e < 0 else e
            continue
        if c.isalnum() or c == "_":
            j = i
            while j < n and (code[j].isalnum() or code[j] == "_"):
                j += 1
            if j < n and code[j] in "'\"" and code[i:j].lower() in _PY_PREFIXES:
                q = j
            else:
                i = j
                continue
        elif c in "'\"":
            q = i
        else:
            i += 1
            continue
        a, b, e, closed = _py_string_at(code, q, q, n)
        interps, nested = [], []
        if "f" in code[i:q].lower():
            k = a
            while k < b:
                if code[k] == "\\":
                    if code.startswith("N{", k + 1):
                        z = code.find("}", k, b)
                        k = b if z < 0 else z + 1
                    else:
                        k += 2
                    continue
                if code[k] == "{":
                    if k + 1 < b and code[k + 1] == "{":
                        k += 2
                        continue
                    k = _fstring_expr(code, k + 1, b, interps, nested)
                    continue
                k += 1
        out.append((i, q, a, b, e, e, interps, closed))
        out.extend(nested)
        i = e
    return out


_LITS = {"c": _lits_c, "java": _lits_java, "js": _lits_js, "python": _lits_python}


def literal_spans(code, lang="c"):
    """Mọi literal (chuỗi, ký tự, raw string, text block, template JS, regex JS, chuỗi Python mọi tiền tố)."""
    return _LITS[_sc._ALIAS.get(lang, lang)](code)


def blank_string_literals(code, lang="c"):
    """24/09 (fix_gadget): xoá trắng NỘI DUNG mọi literal trên CẢ HÀM theo vị trí, GIỮ số dòng.

    Vì sao: clean_gadget xét TỪNG DÒNG bằng hai regex `"..."` rồi `'...'`, nên:
      - dòng GIỮA một chuỗi nhiều dòng (SQL ba nháy, template JS, text block Java, chuỗi nối dòng C) bị đọc
        như mã -> từ trong chuỗi thành VARk/FUNk, trùng tên biến thật thì sinh cạnh def-use/co-use GIẢ;
      - dấu nháy LỒNG ghép cặp sai trên một dòng: `'<a href="' + url + '">'` (JS/Python), `'"'` (C/Java),
        `/"/g` (regex JS) -> nuốt mất biến thật `url` hoặc biến chữ trong chuỗi thành biến;
      - tiền tố chuỗi f / b / r / u (Python), L / u8 (C) còn lại sau `""` thành một VAR dùng chung, nối
        mọi dòng có chuỗi cùng tiền tố với nhau;
      - regex JS `/^[a-z]+$/gi`: chữ trong regex và cờ `g`/`i` thành VAR;
      - f-string MỘT dòng `f"... {user.name}"`: regex xoá cả `{user.name}` -> mất biến bị nội suy (điểm tiêm
        của CWE-089), trong khi f-string nhiều dòng thì giữ;
      - chuỗi chứa `/* */` làm cả dòng bị giữ nguyên văn (không đổi tên gì);
      - gadget bị cắt ở Lmax giữa một chuỗi ba nháy: tokenize báo lỗi, chuỗi đó bị bỏ qua.
    Xử lý: literal MỘT dòng không nội suy -> xoá trắng tiền tố và thân, GIỮ hai dấu nháy (regex từng dòng
    bên dưới vẫn ra đúng `""` / `''` như cũ); literal NHIỀU dòng, không đóng, hoặc có nội suy -> xoá trắng
    cả dấu nháy, giữ `\\n` và giữ nguyên phần mã trong `${...}` / `{...}`; regex JS -> xoá thân và cờ.
    Chỉ ảnh hưởng bản dùng để DỰNG CẠNH; văn bản đưa vào CodeBERT không đổi (trừ khi --norm_text 1)."""
    lang = _sc._ALIAS.get(lang, lang)
    if lang not in _LITS:
        return code
    try:
        lits = _LITS[lang](code)
    except Exception:
        return code
    if not lits:
        return code
    chars = list(code)
    for start, q, a, b, c, end, interps, closed in lits:
        if interps or not closed or "\n" in code[start:end]:
            keep = set()
            for x, y in interps:
                keep.update(range(x, y))
            for k in range(start, end):
                if chars[k] != "\n" and k not in keep:
                    chars[k] = " "
        else:
            for k in range(start, q):
                chars[k] = " "
            for k in range(a, b):
                chars[k] = " "
            for k in range(c, end):
                chars[k] = " "
    return "".join(chars)


def clean_gadget(gadget, lang="c"):
    keywords = KEYWORDS_BY_LANG.get(lang, BASE_KEYWORDS)
    fun_symbols = {}
    var_symbols = {}

    fun_count = 1
    var_count = 1

    rx_comment = re.compile(r'/\*.*?\*/')

    rx_fun = re.compile(r'\b([_A-Za-z]\w*)\b(?=\s*\()')
    rx_var = re.compile(r'\b([_A-Za-z]\w*)\b(?:(?=\s*\w+\()|(?!\s*\w+))(?!\s*\()')

    cleaned_gadget = []

    # 23/09 (v4): xoá trắng comment trên CẢ HÀM theo vị trí (giữ số dòng) rồi mới xét từng dòng.
    # Xét từng dòng riêng lẻ thì không thấy được comment khối / docstring nhiều dòng, và đọc nhầm
    # `\//` của regex JS thành comment. Dữ liệu đã sạch comment thì bước này không đổi gì.
    try:
        blank = blank_comments("\n".join(gadget), lang)
    except ValueError:
        blank = "\n".join(gadget)
    # 24/09 (fix_gadget): xoá trắng nội dung chuỗi nhiều dòng / template JS trên cả hàm (xem blank_string_literals).
    blank = blank_string_literals(blank, lang).split("\n")
    if len(blank) != len(gadget):
        blank = list(gadget)

    for line, bline in zip(gadget, blank):
        if rx_comment.search(bline) is not None:
            # A `/*...*/` that survived comment blanking sits inside a (single-line) string literal. Keep the
            # line verbatim instead of dropping it: dropping shifts every later index, so the
            # data graph built here stops lining up with the control graph built on raw lines.
            cleaned_gadget.append(line)
        else:
            # 23/09: chuoi/ky tu hieu dau nhay thoat (\" \') — ban cu ".*?" cat giua "http:\"//..." roi doi ten URL thanh VARk
            nostrlit_line = re.sub(r'"(?:\\.|[^"\\])*"', '""', bline)

            nocharlit_line = re.sub(r"'(?:\\.|[^'\\])*'", "''", nostrlit_line)

            ascii_line = re.sub(r'[^\x00-\x7f]', r'', nocharlit_line)

            # Comment đã được xoá trắng ở trên (theo vị trí, trên cả hàm) — không xoá lại theo từng dòng:
            # xoá theo dòng từng làm mất định danh ở dòng `#define`/`#if` của C và ở regex JS.
            comment_code = ascii_line

            user_fun = rx_fun.findall(comment_code)
            user_var = rx_var.findall(comment_code)

            for fun_name in user_fun:
                if len({fun_name}.difference(main_set)) != 0 and len({fun_name}.difference(keywords)) != 0:
                    if fun_name not in fun_symbols.keys():
                        fun_symbols[fun_name] = 'FUN' + str(fun_count)
                        fun_count += 1
                    ascii_line = re.sub(r'\b(' + fun_name + r')\b(?=\s*\()', fun_symbols[fun_name], ascii_line)

            for var_name in user_var:
                if len({var_name}.difference(keywords)) != 0 and len({var_name}.difference(main_args)) != 0:
                    if var_name not in var_symbols.keys():
                        var_symbols[var_name] = 'VAR' + str(var_count)
                        var_count += 1
                    ascii_line = re.sub(r'\b(' + var_name + r')\b(?:(?=\s*\w+\()|(?!\s*\w+))(?!\s*\()', \
                                        var_symbols[var_name], ascii_line)

            cleaned_gadget.append(ascii_line)

    return cleaned_gadget


def replace_multiple_whitespace(lst):
    result = []
    for item in lst:
        if not item.strip():
            continue
        cleaned_item = re.sub(r'\s{2,}', ' ', item)
        result.append(cleaned_item)
    return result

