# Retrieving a Software Package
```
%> wget http://framework.zend.com/releases/ZendFramework-1.10.3/ZendFramework-1.10.3-minimal.tar.gz
```

# Monitoring Server Processes
```
%> top
...
ID   COMMAND      %CPU TIME     #TH   #WQ  #PORT MEM    PURG   CMPRS  PGRP PPID STATE    BOOSTS          %CPU_ME %CPU_OTHRS UID  FAULTS
1701  top          2.7  00:00.42 1/1   0    20    2872K+ 0B     0B     1701 604  running  *0[1]           0.00000 0.00000    0    3547+
1675  com.apple.We 0.8  00:02.27 12    2    188-  41M-   6816K  0B     1675 1    sleeping *0[1061]        0.00000 0.00000    501  22651+
1651  fsnotifier   0.0  00:00.02 3     1    30    292K   0B     404K   1644 1644 sleeping *0[1]           0.00000 0.00000    501  1144
1650  syncdefaults 0.0  00:00.56 4     2    124   8652K  0B     880K   1650 1    sleeping  0[2]           0.00000 0.00000    501  6658
1649  CVMCompiler  0.0  00:00.47 2     2    25    11M    0B     1572K  1649 1    sleeping *0[1]           0.00000 0.00000    501  4136
1645  ocspd        0.0  00:00.03 2     1    32    708K   0B     616K   1645 1    sleeping *0[1]           0.00000 0.00000    0    1502
1644  pycharm      3.9  02:01.61 52    2    344   509M   6372K  29M    1644 1    sleeping *0[50]          0.00000 0.00000    501  239705
1601  mdworker     0.0  00:00.12 3     1    56    116K   0B     4308K  1601 1    sleeping *0[1]           0.00000 0.00000    501  4105

%> ps aux
...
USER              PID  %CPU %MEM      VSZ    RSS   TT  STAT STARTED      TIME COMMAND
_windowserver     173   4.1  0.7  4877596  60536   ??  Ss   10:04AM  14:08.75 /System/Library/PrivateFrameworks/SkyLight.framework/Resources/WindowServer -daemon
hongong          1644   2.2  8.2  7245168 688464   ??  S    12:16PM   7:39.61 /Applications/PyCharm CE.app/Contents/MacOS/pycharm
hongong           589   1.2  0.5  2661812  44496   ??  S    10:09AM   0:43.55 /Applications/Utilities/Terminal.app/Contents/MacOS/Terminal
hongong           701   0.8  0.5  2976980  41488   ??  S    10:19AM   4:25.12 /Applications/Sublime Text.app/Contents/MacOS/Sublime Text

kill -<level> <pid>	Kill a process	$ kill -15 24601
pkill -<level> -f <name>	Kill matching processes	$ pkill -15 -f spring
```

# Reviewing Log Files
```
%> tail /var/log/apache/error.log
%> tail -n 100 /var/log/apache/error.log | more
%> tail -f /var/log/apache/error.log
%> cat /var/log/apache/error.log
%> less /var/log/apache/error.log
```

# Copying Files with scp
```
%> scp id_rsa.pub webuser@192.168.1.1:/home/webuser/.ssh/id_rsa.pub
```

# Backing Up Your Web Directory
```
# backup
%> tar cpzf archive.backup.042710.tgz /var/mywebsite
# restore
%> tar xvpfz archive.backup.042710.tgz -C /var/www/
```

# Viewing Your Command History
```
%> history
...
12  sudo ./configure && make
13  find . | grep config.log
14  less ./config.log 
15  mongod
16  asadmin start-domain --debug
```

# Creating Directory Trees
```
%> mkdir -p webapp/application/controllers
```

# Creating Command Aliases
You can add them to an account configuration file such as .bashrc.
```
%> alias dir='ls -al'
%> dir
...
drwxr-xr-x@  6 hongong  staff      204 Oct 20 12:00 .
drwxr-xr-x@ 13 hongong  staff      442 Sep  5 21:51 ..
-rw-r--r--@  1 hongong  staff     6148 Sep 26 15:29 .DS_Store
drwxr-xr-x   3 hongong  staff      102 Sep  5 19:01 server
-rw-r--r--   1 hongong  staff   158814 Oct 20 12:20 server.log
```

# Editing the line
```
echo <string>	Print string to screen
man <command>	Display manual page for command
⌃C	Get out of trouble
⌃A	Move to beginning of line
⌃E	Move to end of line
⌃U	Delete to beginning of line
⌃K	Delete to ending of line
⌃W	Delete word before cursor
```

# Manipulating files
```
>	                Redirect output to filename
>>	                Append output to filename
diff <f1> <f2>	    Diff files 1 & 2
```

# Wordcount and pipes
```
wc server.log
1131   10946  167679 server.log
1131 lines, 10946 words, 167679 bytes 

head server.log | wc
10     104    1438
```

# Less is more
```
up & down arrow keys	Move up or down one line	
spacebar	            Move forward one page	
⌃F	                    Move forward one page	
⌃B	                    Move back one page	
G	                    Move to end of file	
1G	                    Move to beginning of file
/<string>	            Search file for string
n	                    Move to next search result
N	                    Move to previous search result
q	                    Quit less
-N                      View line number
```

# Grepping
```
grep <string> <file>	Find string in file
grep -i <string> <file>	Find case-insensitively

```