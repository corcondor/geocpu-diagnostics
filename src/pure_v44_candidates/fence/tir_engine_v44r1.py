"""Standard Python TIR: generated code -> actual observation -> continued generation.

No ALU controller; no model hidden-state coordinates are interpreted as token IDs.
The worker is process/time/output bounded, not an adversarial security sandbox.
"""
from dataclasses import dataclass
import json
import os
from pathlib import Path
import queue
import re
import subprocess
import sys
import tempfile
import threading
import time

class StrictVocabularyError(ValueError):pass

class PythonExecutor:
    def __init__(self,timeout_seconds=5.,max_output_chars=8192,max_code_chars=32768,startup_timeout_seconds=60.,search_callback=None,max_search_calls=8):
        self.timeout_seconds=float(timeout_seconds);self.max_output_chars=int(max_output_chars);self.max_code_chars=int(max_code_chars)
        self.temp=tempfile.TemporaryDirectory(prefix='geocpu_tir_');self.messages=queue.Queue();self.stderr_tail='';self.closed=False
        self.search_callback=search_callback;self.max_search_calls=int(max_search_calls);self.search_trace=[]
        worker=Path(__file__).with_name('executor_worker_v44r1.py')
        env={k:v for k,v in os.environ.items() if k.upper() in {'PATH','SYSTEMROOT','WINDIR','SYSTEMDRIVE','TEMP','TMP','HOME','USERPROFILE','LANG','LC_ALL'}}
        env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONIOENCODING='utf-8',PYTHONDONTWRITEBYTECODE='1')
        kwargs={'creationflags':subprocess.CREATE_NO_WINDOW} if os.name=='nt' else {}
        self.process=subprocess.Popen([sys.executable,'-u',str(worker)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
            cwd=self.temp.name,text=True,encoding='utf-8',bufsize=1,env=env,**kwargs)
        def read_stdout():
            try:
                for line in self.process.stdout:
                    try:self.messages.put(json.loads(line))
                    except json.JSONDecodeError:self.messages.put({'protocol_error':line[:1000]})
            finally:self.messages.put({'worker_eof':True})
        def read_stderr():
            for line in self.process.stderr:self.stderr_tail=(self.stderr_tail+line)[-4096:]
        self.stdout_thread=threading.Thread(target=read_stdout,daemon=True);self.stdout_thread.start()
        self.stderr_thread=threading.Thread(target=read_stderr,daemon=True);self.stderr_thread.start()
        try:self.ready=self.messages.get(timeout=startup_timeout_seconds)
        except queue.Empty:
            self.close();raise TimeoutError('Python worker startup timeout')
        if not self.ready.get('ready'):
            detail=self.stderr_tail;self.close();raise RuntimeError('Python worker failed: '+str(self.ready)+' '+detail)
    def _rpc(self,request,timeout):
        if self.closed or self.process.poll() is not None:raise RuntimeError('Python worker is not alive')
        self.process.stdin.write(json.dumps(request,ensure_ascii=False)+'\n');self.process.stdin.flush()
        deadline=time.monotonic()+timeout
        while True:
            remaining=deadline-time.monotonic()
            if remaining<=0:raise queue.Empty
            response=self.messages.get(timeout=remaining)
            if 'protocol_error' in response or 'worker_eof' in response:raise RuntimeError('Python worker protocol failure: '+str(response)+' '+self.stderr_tail)
            if response.get('op')!='search_request':return response
            reply={'op':'search_response','id':response.get('id'),'result':None,'error':None}
            try:
                query=response.get('query')
                if self.search_callback is None:raise PermissionError('Offline corpus search is disabled')
                if not isinstance(query,str) or not query.strip() or len(query)>512:raise ValueError('Search query must be 1..512 characters')
                if len(self.search_trace)>=self.max_search_calls:raise RuntimeError('Search call limit reached')
                text=self.search_callback(query)
                if not isinstance(text,str):raise TypeError('Offline corpus search must return text')
                reply['result']=text[:2048]
                self.search_trace.append({'query':query,'result':reply['result'],'source':'fixed_v43_offline_corpus'})
            except Exception as exc:reply['error']=type(exc).__name__+': '+str(exc)
            self.process.stdin.write(json.dumps(reply,ensure_ascii=False)+'\n');self.process.stdin.flush()
    def execute(self,code):
        if not isinstance(code,str):raise TypeError('code must be text')
        before=len(self.search_trace)
        try:
            result=self._rpc({'op':'execute','code':code,'max_output_chars':self.max_output_chars,'max_code_chars':self.max_code_chars,'search_enabled':self.search_callback is not None},self.timeout_seconds)
            result['search_trace']=self.search_trace[before:]
            return result
        except queue.Empty:
            self.close();return {'success':False,'stdout':'','stderr':'','error':'TimeoutError: Python execution deadline exceeded',
                'exception_type':'TimeoutError','timeout':True,'truncated':False,'worker_alive':False,'search_trace':self.search_trace[before:]}
        except (RuntimeError,BrokenPipeError) as exc:
            self.close();return {'success':False,'stdout':'','stderr':self.stderr_tail,'error':str(exc),'exception_type':'WorkerError','timeout':False,'truncated':False,'worker_alive':False,'search_trace':self.search_trace[before:]}
    def reset(self):
        result=self._rpc({'op':'reset'},self.timeout_seconds);self.search_trace.clear();return result
    def close(self):
        if self.closed:return
        self.closed=True
        if self.process.poll() is None:
            try:self.process.kill()
            except OSError:pass
        try:self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:pass
        for f in (self.process.stdin,self.process.stdout,self.process.stderr):
            try:f.close()
            except Exception:pass
        self.temp.cleanup()
    def __enter__(self):return self
    def __exit__(self,*args):self.close()

class StrictCharacterTokenizer:
    def __init__(self,vocab,inv_vocab=None,eos_id=3):
        self.vocab=vocab;self.inv={int(k):v for k,v in inv_vocab.items()} if inv_vocab is not None else {int(v):k for k,v in vocab.items()};self.eos_id=int(eos_id)
    def encode(self,text):
        missing=sorted({c for c in text if c not in self.vocab})
        if missing:raise StrictVocabularyError('Characters absent from vocabulary: '+repr(missing[:12]))
        return [int(self.vocab[c]) for c in text]
    def decode_one(self,token):
        text=self.inv.get(int(token))
        if not isinstance(text,str) or len(text)!=1:raise StrictVocabularyError('Non-character or unknown generated token: '+repr(token))
        return text

class TorchModelAdapter:
    """Adapter for IntegratedModel prefill/step_state; uses state.last_logits only."""
    def __init__(self,model,arm='learned',repetition_penalty=1.0,vocab=None):
        import torch
        self.torch=torch;self.model=model;self.arm=arm;self.device=next(model.parameters()).device
        self.max_context_tokens=int(model.config.max_positions)
        self.repetition_penalty=float(repetition_penalty);self.history=[]
        if self.repetition_penalty<=0:raise ValueError('Repetition penalty must be positive')
        self.layout_tokens={vocab[c] for c in ' \n\t|' if c in vocab} if vocab is not None else set()
    def prefill(self,ids):
        self.history=[]
        with self.torch.no_grad():return self.model.prefill(self.torch.tensor([ids],dtype=self.torch.long,device=self.device),arm=self.arm)
    def next_token(self,state):
        logits=state.last_logits
        if logits is None or logits.numel()==0:raise RuntimeError('Missing vocabulary logits')
        scores=logits.reshape(-1,logits.shape[-1])[-1]
        if self.repetition_penalty!=1.0 and self.history:
            scores=scores.float().clone()
            for token in set(self.history[-16:]):
                if token in self.layout_tokens:continue
                value=float(scores[token]);scores[token]=value/self.repetition_penalty if value>0 else value*self.repetition_penalty
            last=self.history[-1];quad=len(self.history)>=4 and len(set(self.history[-4:]))==1
            if last not in self.layout_tokens or quad:
                value=float(scores[last]);scores[last]=value/self.repetition_penalty if value>0 else value*self.repetition_penalty
        return int(self.torch.argmax(scores).item())
    def advance(self,state,token):
        self.history.append(int(token))
        with self.torch.no_grad():return self.model.step_state(self.torch.tensor([[int(token)]],dtype=self.torch.long,device=self.device),state,compute_logits=True)

@dataclass(frozen=True)
class TIRLimits:
    max_new_tokens:int=2048
    max_context_tokens:int=8192
    max_tool_calls:int=8
    max_wall_seconds:float=180.
    per_tool_seconds:float=5.
    max_output_chars:int=8192
    max_code_chars:int=32768
    max_sampled_tokens:int|None=None
    max_search_calls:int=8

PYTHON_BLOCK=re.compile(r'(?ms)^```python[ \t]*\r?\n(.*?)^```[ \t]*(?:\r?\n|$)')
FORGED_OUTPUT=re.compile(r'(?m)^```output[ \t]*(?:\r?\n|$)')

def observation_text(result):
    content=result.get('stdout','')
    if result.get('stderr'):content+='\nSTDERR:\n'+result['stderr']
    if not result.get('success'):content+='\n'+str(result.get('error') or 'ExecutionError')
    if not content:content='[no stdout]'
    max_run=max([len(m.group()) for m in re.finditer(r'`+',content)]+[0]);fence='`'*max(3,max_run+1)
    return '\n'+fence+'output\n'+content+('' if content.endswith('\n') else '\n')+fence+'\n'

def generate_with_python_tir(adapter,prompt,vocab,inv_vocab=None,*,limits=None,executor=None,eos_id=3,search_callback=None):
    limits=limits or TIRLimits();tokenizer=StrictCharacterTokenizer(vocab,inv_vocab,eos_id)
    prompt_ids=tokenizer.encode(prompt)
    native_cap=getattr(adapter,'max_context_tokens',limits.max_context_tokens)
    cap=min(int(native_cap),int(limits.max_context_tokens))
    if not prompt_ids:raise ValueError('Empty prompt')
    if len(prompt_ids)>cap:raise ValueError('Prompt exceeds native context capacity')
    if limits.max_new_tokens<0 or limits.max_tool_calls<0:raise ValueError('Negative generation/tool budget')
    started=time.monotonic();state=adapter.prefill(prompt_ids);sampled=[];injected=[];trace=[];model_text='';text='';scan=0
    stop='new_token_limit';error=None;ended=False;owned=executor is None
    tool=executor
    try:
        while True:
            if time.monotonic()-started>=limits.max_wall_seconds:stop='wall_time_limit';break
            if len(prompt_ids)+len(sampled)+len(injected)>=cap:stop='context_limit';break
            if len(sampled)+len(injected)>=limits.max_new_tokens:stop='new_token_limit';break
            if limits.max_sampled_tokens is not None and len(sampled)>=limits.max_sampled_tokens:stop='sampled_token_limit';break
            token=int(adapter.next_token(state))
            if token!=eos_id:
                try:char=tokenizer.decode_one(token)
                except StrictVocabularyError as exc:stop='unknown_token';error=str(exc);break
            sampled.append(token)
            state=adapter.advance(state,token)  # Every sampled token enters the state exactly once.
            if token==eos_id:ended=True;stop='eos';break
            model_text+=char;text+=char
            if FORGED_OUTPUT.search(model_text[scan:]):stop='model_generated_observation';error='Model attempted to write a tool observation';break
            match=PYTHON_BLOCK.search(model_text,scan)
            if not match:continue
            if len(trace)>=limits.max_tool_calls:stop='tool_call_limit';break
            code=match.group(1)
            if len(code)>limits.max_code_chars:stop='code_length_limit';break
            remaining=max(0.,limits.max_wall_seconds-(time.monotonic()-started))
            if remaining<=0:stop='wall_time_limit';break
            if tool is None:
                try:tool=PythonExecutor(limits.per_tool_seconds,limits.max_output_chars,limits.max_code_chars,startup_timeout_seconds=min(60.,remaining),search_callback=search_callback,max_search_calls=limits.max_search_calls)
                except (TimeoutError,RuntimeError) as exc:stop='tool_worker_startup_error';error=str(exc);break
            remaining=max(0.,limits.max_wall_seconds-(time.monotonic()-started))
            if remaining<=0:stop='wall_time_limit';break
            previous_timeout=tool.timeout_seconds;tool.timeout_seconds=min(previous_timeout,remaining)
            try:result=tool.execute(code)
            finally:tool.timeout_seconds=previous_timeout
            item={'index':len(trace),'code':code,'result':result,'observation':None,'observation_complete':False,'injected_token_ids':[]};trace.append(item)
            obs=observation_text(result)
            try:obs_ids=tokenizer.encode(obs)
            except StrictVocabularyError as exc:stop='tool_output_unencodable';error=str(exc);break
            if len(prompt_ids)+len(sampled)+len(injected)+len(obs_ids)>cap:stop='context_limit';break
            if len(sampled)+len(injected)+len(obs_ids)>limits.max_new_tokens:stop='new_token_limit';break
            partial=[]
            for itok in obs_ids:
                if time.monotonic()-started>=limits.max_wall_seconds:stop='wall_time_limit';break
                state=adapter.advance(state,itok);injected.append(itok);partial.append(itok)
            injected_text=''.join(tokenizer.decode_one(t) for t in partial)
            text+=injected_text;item['observation']=injected_text;item['injected_token_ids']=partial
            if len(partial)!=len(obs_ids):break
            item['observation_complete']=True
            scan=match.end()
            if not result.get('worker_alive',True):stop='tool_worker_terminated';break
    finally:
        if owned and tool is not None:tool.close()
    return {'output_text':text,'full_text':prompt+text,'model_generated_text':model_text,'generated_tokens':sampled,
        'injected_tokens':injected,'tool_trace':trace,'tool_invocations':len(trace),
        'successful_tool_invocations':sum(t['result']['success'] for t in trace),'ended_with_eos':ended,
        'stop_reason':stop,'error':error,'sampled_token_count':len(sampled),'injected_token_count':len(injected),
        'total_new_token_count':len(sampled)+len(injected),'total_context_token_count':len(prompt_ids)+len(sampled)+len(injected),
        'wall_seconds':time.monotonic()-started,'security_sandbox':False,
        'search_trace':[call for item in trace for call in item['result'].get('search_trace',[])]}
