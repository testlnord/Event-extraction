import os
import subprocess
import logging as log

def run_server(corenlp_path, port=9000, timeout=None, java_mx_mem='4g'):
    # TODO check for path existence
    corenlp_server = "edu.stanford.nlp.pipeline.StanfordCoreNLPServer"
    command = ['java', '-mx'+java_mx_mem, '-cp', corenlp_path + '/*', corenlp_server]
    if port:
        command += ['-port', str(port)]
    if timeout:
        # in milliseconds
        command += ['-timeout', str(timeout)]

    serv_completed = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    # TODO maybe some postprocessing and error handling
    return_code = serv_completed.returncode
    log.info(return_code)


if __name__ == "__main__":
    log.basicConfig(format='%(levelname)s:%(message)s', level=log.INFO)
    path = '/home/hades/projects/corenlp'
    run_server(path)
