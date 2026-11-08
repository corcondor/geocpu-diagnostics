"""Bounded math Python worker. Guarded process; not a security sandbox."""
import ast
import builtins
import contextlib
import importlib
import io
import json
import os
from pathlib import Path
import sys

ROOTS={'math','cmath','decimal','fractions','statistics','itertools','functools','operator','collections','sympy','numpy','pandas'}
MODULES={}
IMPORT_ERRORS={}
for root in sorted(ROOTS):
    try:MODULES[root]=importlib.import_module(root)
    except Exception as exc:IMPORT_ERRORS[root]=type(exc).__name__+': '+str(exc)

REAL_IMPORT=builtins.__import__
LIBRARY_ROOTS=[]
for name,module in MODULES.items():
    fn=getattr(module,'__file__',None)
    if fn:LIBRARY_ROOTS.append(Path(fn).resolve().parent)
LIBRARY_ROOTS.extend([Path(sys.base_prefix).resolve(),Path(sys.prefix).resolve()])

def audit(event,args):
    if event.startswith(('socket.','subprocess.','ctypes.')) or event in {'os.system','os.exec','os.posix_spawn','os.spawn','os.fork','os.kill','os.remove','os.rmdir','os.rename','os.mkdir','os.chdir','os.chmod','os.link','os.symlink','os.truncate'}:
        raise PermissionError('Operation disabled in math worker: '+event)
    if event in {'os.listdir','os.scandir'}:
        file=args[0]
        if not isinstance(file,(str,bytes,os.PathLike)):raise PermissionError('Filesystem enumeration disabled')
        path=Path(os.fsdecode(file)).resolve()
        if not any(path==root or root in path.parents for root in LIBRARY_ROOTS):raise PermissionError('Filesystem enumeration outside math libraries disabled')
    if event=='open':
        file,mode,flags=args
        if not isinstance(file,(str,bytes,os.PathLike)):raise PermissionError('File descriptor access disabled')
        path=Path(os.fsdecode(file)).resolve()
        write_flags=os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND
        if flags & write_flags:raise PermissionError('Filesystem writes disabled')
        if not any(path==root or root in path.parents for root in LIBRARY_ROOTS):
            raise PermissionError('Filesystem reads outside installed math libraries disabled')

sys.addaudithook(audit)

DENIED_NAMES={'eval','exec','compile','open','input','help','dir','vars','globals','locals','getattr','setattr','delattr','breakpoint','memoryview'}
DENIED_ATTRS={'save','savez','savez_compressed','load','loads','dump','fromfile','tofile','memmap','ctypes','ctypeslib','system','popen','fork','spawn','get_handle','urlopen','request','write','writelines','open','codegen','autowrap','ufuncify','os','sys','builtins','subprocess','socket','environ','attrgetter','methodcaller'}

def validate(code):
    tree=ast.parse(code,filename='<tir-code>',mode='exec')
    for node in ast.walk(tree):
        if isinstance(node,(ast.Global,ast.Nonlocal,ast.ClassDef)):
            raise PermissionError('Global/nonlocal/class declarations disabled')
        if isinstance(node,ast.Name) and (node.id.startswith('__') or node.id in DENIED_NAMES):
            raise PermissionError('Name disabled: '+node.id)
        if isinstance(node,ast.Attribute):
            if node.attr.startswith('_') or node.attr in DENIED_ATTRS or node.attr.startswith(('read_','to_csv','to_excel','to_json','to_pickle','to_parquet','to_hdf','to_sql','to_feather','to_orc')):
                raise PermissionError('Attribute disabled: '+node.attr)
            if isinstance(node.ctx,(ast.Store,ast.Del)):
                raise PermissionError('Attribute mutation disabled')
        if isinstance(node,ast.Import):
            for name in node.names:
                if name.name.split('.')[0] not in ROOTS:raise PermissionError('Import disabled: '+name.name)
        if isinstance(node,ast.ImportFrom):
            if node.level or not node.module or node.module.split('.')[0] not in ROOTS:
                raise PermissionError('Import disabled: '+str(node.module))
            if any(a.name=='*' or a.name.startswith('_') for a in node.names):raise PermissionError('Wildcard/private import disabled')
    return tree

def safe_import(name,globals=None,locals=None,fromlist=(),level=0):
    if level or name.split('.')[0] not in ROOTS:raise PermissionError('Import disabled: '+name)
    if name.split('.')[0] not in MODULES:raise ImportError(IMPORT_ERRORS.get(name,'Unavailable math library'))
    return REAL_IMPORT(name,globals,locals,fromlist,level)

SAFE_BUILTINS={name:getattr(builtins,name) for name in (
 'abs','all','any','bin','bool','bytes','bytearray','chr','complex','dict','divmod','enumerate','filter','float','format','frozenset','hash','hex','int','isinstance','issubclass','iter','len','list','map','max','min','next','oct','ord','pow','print','range','repr','reversed','round','set','slice','sorted','str','sum','tuple','zip','Exception','ValueError','TypeError','ZeroDivisionError','RuntimeError','AssertionError','NameError','IndexError','KeyError','StopIteration')}
SAFE_BUILTINS['__import__']=safe_import

SEARCH_ENABLED=False
SEARCH_SEQUENCE=0

def offline_search(query):
    global SEARCH_SEQUENCE
    if not SEARCH_ENABLED:raise PermissionError('Offline corpus search is disabled')
    if not isinstance(query,str) or not query.strip() or len(query)>512:raise ValueError('Search query must be 1..512 characters')
    SEARCH_SEQUENCE+=1
    sys.__stdout__.write(json.dumps({'op':'search_request','id':SEARCH_SEQUENCE,'query':query},ensure_ascii=False)+'\n');sys.__stdout__.flush()
    response=json.loads(sys.stdin.readline())
    if response.get('op')!='search_response' or response.get('id')!=SEARCH_SEQUENCE:raise RuntimeError('Search response protocol mismatch')
    if response.get('error'):raise RuntimeError(response['error'])
    result=response.get('result')
    if not isinstance(result,str):raise TypeError('Search response must be text')
    return result

def fresh_namespace():return {'__builtins__':SAFE_BUILTINS.copy(),'__name__':'tir_math','search':offline_search}

class OutputLimitExceeded(RuntimeError):pass
class CappedText(io.TextIOBase):
    def __init__(self,cap):self.cap=cap;self.parts=[];self.size=0;self.truncated=False
    def write(self,text):
        text=str(text);room=max(0,self.cap-self.size)
        if room:self.parts.append(text[:room]);self.size+=min(len(text),room)
        if len(text)>room:self.truncated=True;raise OutputLimitExceeded('Captured output exceeded '+str(self.cap)+' characters')
        return len(text)
    def flush(self):pass
    def getvalue(self):return ''.join(self.parts)

def main():
    global SEARCH_ENABLED
    namespace=fresh_namespace()
    print(json.dumps({'ready':True,'libraries':sorted(MODULES),'import_errors':IMPORT_ERRORS,'security_sandbox':False}),flush=True)
    for line in sys.stdin:
        request=json.loads(line);op=request.get('op')
        if op=='close':break
        if op=='reset':
            namespace=fresh_namespace();print(json.dumps({'reset':True}),flush=True);continue
        if op!='execute':print(json.dumps({'success':False,'error':'Unknown operation'}),flush=True);continue
        SEARCH_ENABLED=request.get('search_enabled') is True
        code=request['code'];cap=int(request.get('max_output_chars',8192));out=CappedText(cap);err=CappedText(cap)
        result={'success':False,'stdout':'','stderr':'','error':None,'exception_type':None,'timeout':False,'truncated':False,'worker_alive':True}
        try:
            if not isinstance(code,str) or len(code)>int(request.get('max_code_chars',32768)):raise ValueError('Code length limit')
            tree=validate(code)
            with contextlib.redirect_stdout(out),contextlib.redirect_stderr(err):
                exec(compile(tree,'<tir-code>','exec'),namespace,namespace)
            result['success']=True
        except BaseException as exc:
            result['exception_type']=type(exc).__name__;result['error']=type(exc).__name__+': '+str(exc)
        result['stdout']=out.getvalue();result['stderr']=err.getvalue();result['truncated']=out.truncated or err.truncated
        print(json.dumps(result,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
