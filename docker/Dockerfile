FROM edarace-agent-base:ubuntu24

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_NO_INDEX=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONHASHSEED=0 \
    PCBPARSE_APP_ROOT=/submission/app \
    LD_LIBRARY_PATH=/usr/local/lib/python3.12/dist-packages/nvidia/cuda_nvrtc/lib:/usr/local/lib/python3.12/dist-packages/nvidia/cuda_runtime/lib:/usr/local/lib/python3.12/dist-packages/nvidia/cublas/lib:/usr/local/lib/python3.12/dist-packages/nvidia/cudnn/lib:/usr/local/lib/python3.12/dist-packages/nvidia/cufft/lib:/usr/local/lib/python3.12/dist-packages/nvidia/curand/lib:/usr/local/lib/python3.12/dist-packages/nvidia/cusolver/lib:/usr/local/lib/python3.12/dist-packages/nvidia/cusparse/lib:/usr/local/lib/python3.12/dist-packages/nvidia/nccl/lib:/usr/local/lib/python3.12/dist-packages/nvidia/nvjitlink/lib:/usr/local/lib/python3.12/dist-packages/nvidia/nvtx/lib:/usr/local/lib/python3.12/dist-packages/nvidia/cuda_cupti/lib

WORKDIR /submission/app

COPY requirements.edarace.txt /submission/app/
COPY wheels /submission/wheels
COPY docker/verify_wheelhouse.py /submission/verify_wheelhouse.py
COPY configs /submission/app/configs
COPY models/models.lock.json /submission/app/models/models.lock.json
COPY models/releases/v8-portable /submission/app/models/releases/v8-portable
COPY docker/agent_system /submission/agent_system
COPY docker/agent_system /submission/run.sh

RUN python3 /submission/verify_wheelhouse.py /submission/wheels \
    && python3 -m pip install --break-system-packages --no-cache-dir \
        --no-index --find-links=/submission/wheels --only-binary=:all: --no-deps \
        -r /submission/app/requirements.edarace.txt \
    && python3 -m pip install --break-system-packages --no-cache-dir --no-index --no-deps \
        /submission/wheels/pcb_schematic_parser-0.1.0-py3-none-any.whl \
    && chmod 0555 /submission/agent_system /submission/run.sh

# BuildKit has no GPU device. Verify imports/serialization plus every model and
# lineage byte here; execute the four runtime adapters after image creation.
RUN python3 -m pcbparse.cli healthcheck \
        --config /submission/app/configs/pipeline.edarace.yaml \
        --models-lock /submission/app/models/models.lock.json \
    && python3 -c "from pcbparse.perception.production_provider import composite_provider_config_from_pipeline, required_model_tasks_for_config; from pcbparse.config import load_config; from pcbparse.service.readiness import readiness_report; config=load_config('/submission/app/configs/pipeline.edarace.yaml'); tasks=required_model_tasks_for_config(composite_provider_config_from_pipeline(config)); report=readiness_report(provider_name='production', models_lock='/submission/app/models/models.lock.json', required_tasks=tasks, require_lineage=True); assert report['ready'], report"

WORKDIR /submission
ENTRYPOINT ["/submission/run.sh"]
