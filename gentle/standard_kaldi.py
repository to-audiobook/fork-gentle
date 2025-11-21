import subprocess
import os
import logging

from .util.paths import get_binary

EXECUTABLE_PATH = get_binary("ext/k3")
logger = logging.getLogger(__name__)

STDERR = subprocess.DEVNULL

class Kaldi:
    def __init__(self, nnet_dir=None, hclg_path=None, proto_langdir=None):
        cmd = [EXECUTABLE_PATH]
        
        if nnet_dir is not None:
            cmd.append(nnet_dir)
            cmd.append(hclg_path)

        if not os.path.exists(hclg_path):
            logger.error('hclg_path does not exist: %s', hclg_path)
        self._p = subprocess.Popen(cmd,
                                   stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                   stderr=STDERR)
        self.finished = False

    def _cmd(self, c):
        self._p.stdin.write(("%s\n" % (c)).encode())
        self._p.stdin.flush()

    def push_chunk(self, buf):
        # Wait until we're ready
        self._cmd("push-chunk")
        
        cnt = int(len(buf)/2)

        # we will write the whole buffer to stdin, but k3 will read only 
        # {cnt * 2}. If {len(buf)} is not divisible by 2 then stdin will be left
        # with "garbage" and when k3 tries to read what it thinks is the next
        # command, bad things are likely to happen
        if((cnt * 2) != len(buf)):
            logging.info(f'Kaldi.push_chunk() buffer size not multiple of 2!!! len(buf): {len(buf)}');

        self._cmd(str(cnt))
        self._p.stdin.write(buf) #arr.tostring())
        self._p.stdin.flush()
        status = self._p.stdout.readline().strip().decode()
        return status == 'ok'

    def get_final(self):
        self._cmd("get-final")
        words = []
        while True:
            line = self._p.stdout.readline().decode()
            if line.startswith("done"):
                break
            parts = line.split(' / ')
            if line.startswith('word'):
                wd = {}
                wd['word'] = parts[0].split(': ')[1]
                wd['start'] = float(parts[1].split(': ')[1])
                wd['duration'] = float(parts[2].split(': ')[1])
                wd['phones'] = []
                words.append(wd)
            elif line.startswith('phone'):
                ph = {}
                ph['phone'] = parts[0].split(': ')[1]
                ph['duration'] = float(parts[1].split(': ')[1])
                words[-1]['phones'].append(ph)

        self._reset()
        return words

    def _reset(self):
        self._cmd("reset")

    def stop(self):
        if not self.finished:
            self.finished = True
            self._cmd("stop")
            # read everything from stdout before waiting for the child process
            # to terminate, just in case the child process was waiting for the
            # buffer to have enough space to write stuff. Otherwise we might
            # deadlock ourselves
            while(True):
                chunk = self._p.stdout.read(4096);
                if(len(chunk) <= 0):
                    break;
            # the above does not seem to be enough. We are still deadlocking
            # sometimes, so let's just kill the process
            self._p.wait(timeout=2);
            self._p.stdin.close()
            self._p.stdout.close()            

    def __del__(self):
        self.stop()

if __name__=='__main__':
    import numm3
    import sys

    infile = sys.argv[1]
    
    k = Kaldi()

    buf = numm3.sound2np(infile, nchannels=1, R=8000)
    print('loaded_buf', len(buf))
    
    idx=0
    while idx < len(buf):
        k.push_chunk(buf[idx:idx+160000].tostring())
        print(k.get_final())
        idx += 160000
