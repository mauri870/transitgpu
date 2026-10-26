"""Refinement of one peak of the hybrid search (transitgpu.hybrid): TLS's `power()` on a window of periods around it."""


def refine_peak(t, y, dy, period, window, backend="gpu", tls_options=None):
    """Run TLS from period * (1 - window) to period * (1 + window) and return (results, None), or (None, reason).

    The reason is the type and message of TLS's error (no period grid in the window, too few points). `backend` None
    leaves TLS's default; any other value is passed as `power(backend=...)` with the backends of transitgpu.tls
    registered.
    """
    try:
        options = {"show_progress_bar": False, "verbose": False, **(tls_options or {})}
        if backend is None:
            from transitleastsquares import transitleastsquares
        else:
            from transitgpu.tls import transitleastsquares

            options["backend"] = backend
        return transitleastsquares(t, y, dy).power(period_min=period * (1 - window), period_max=period * (1 + window), **options), None
    except Exception as error:  # a window that TLS cannot search: the peak keeps the box values and records why
        return None, f"{type(error).__name__}: {error}"
