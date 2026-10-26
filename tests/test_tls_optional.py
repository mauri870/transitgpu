"""Transit Least Squares is an optional dependency of transitgpu (the extra `tls`): a plain `import transitgpu` must not import it, and `import transitgpu.tls` must give TLS's interface with the backends of the GPU. Each in a fresh interpreter, because an import in this process would hide the other."""
import subprocess
import sys


def run(code):
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)


def test_a_plain_import_of_transitgpu_does_not_import_transit_least_squares():
    found = run("import sys, transitgpu; print('transitleastsquares' in sys.modules); print('transitgpu.tls' in sys.modules)")
    assert found.returncode == 0, found.stderr
    assert found.stdout.split() == ["False", "False"], found.stdout


def test_importing_transitgpu_tls_gives_the_interface_of_tls_with_the_gpu_backends_registered():
    found = run("import transitgpu.tls as tls; from transitleastsquares import transitleastsquares as base; print(issubclass(tls.transitleastsquares, base)); print(sorted(b for b in tls.available_backends() if b.startswith('gpu')))")
    assert found.returncode == 0, found.stderr
    assert found.stdout.split("\n")[:2] == ["True", "['gpu', 'gpu-binned', 'gpu-pl']"], found.stdout
