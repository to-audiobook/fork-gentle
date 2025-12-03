import subprocess
import os
import logging
import tempfile;

from .util.paths import get_binary

EXECUTABLE_PATH = get_binary("ext/k3")
logger = logging.getLogger(__name__)

STDERR = subprocess.DEVNULL

class Kaldi:
    def __init__(self, nnet_dir=None, hclg_path=None, proto_langdir=None):
        self.nnet_dir = nnet_dir;
        self.hclg_path = hclg_path;
        
        if not os.path.exists(hclg_path):
            logger.error('hclg_path does not exist: %s', hclg_path)

    def push_chunk(self, buf):
        cnt = int(len(buf)/2)

        # we will write the whole buffer to stdin, but k3 will read only 
        # {cnt * 2}. If {len(buf)} is not divisible by 2 then stdin will be left
        # with "garbage" and when k3 tries to read what it thinks is the next
        # command, bad things are likely to happen
        if((cnt * 2) != len(buf)):
            raise Exception(f'Kaldi.push_chunk() buffer size not multiple of 2!!! len(buf): {len(buf)}');

        # create temporary files we will use to send data and read the results
        # from the k3 application
        with tempfile.NamedTemporaryFile(mode='w', delete=True) as chunkFile, \
            tempfile.NamedTemporaryFile(mode='rw', delete=True) as resultFile:
            chunkFile.write(str(cnt));
            chunkFile.write(buf);
            chunkFile.flush();

            cmd = [
                EXECUTABLE_PATH,
                    self.nnet_dir,
                    self.hclg_path,
                    chunkFile.name,
                    resultFile.name
            ];

            subprocess.run(cmd, check=True);

            self._words = [];
            resultFile.seek(0, 0);
            while True:
                line = resultFile.readline().decode()
                if line.startswith("done"):
                    break
                parts = line.split(' / ')
                if line.startswith('word'):
                    wd = {}
                    wd['word'] = parts[0].split(': ')[1]
                    wd['start'] = float(parts[1].split(': ')[1])
                    wd['duration'] = float(parts[2].split(': ')[1])
                    wd['phones'] = []
                    self._words.append(wd)
                elif line.startswith('phone'):
                    ph = {}
                    ph['phone'] = parts[0].split(': ')[1]
                    ph['duration'] = float(parts[1].split(': ')[1])
                    self._words[-1]['phones'].append(ph)
        
        return True;


    def get_final(self):        
        return self._words;

    def _reset(self):
        pass;

    def stop(self):
        pass;

    def __del__(self):
        pass;

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
