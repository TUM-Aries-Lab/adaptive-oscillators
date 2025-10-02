FROM python:3.12-slim

ARG PKG=adaptive_oscillator
ARG VER=latest
ENV PKG=${PKG} VER=${VER}

RUN python -m pip install --upgrade pip && \
    if [ "$VER" = "latest" ]; then \
        pip install "$PKG"; \
    else \
        pip install "$PKG==$VER"; \
    fi

# Add a little smoke-test script
RUN printf '%s\n' \
  "import importlib, os, sys" \
  "m = importlib.import_module(os.environ['PKG'])" \
  "print('✅ import ok:', getattr(m, '__version__', 'unknown'), 'on', sys.version)" \
  > /smoke.py

CMD ["python", "/smoke.py"]
