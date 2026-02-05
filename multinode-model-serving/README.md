# Multi-node Model Serving in PCAI

| Owner                 | Name              | Email                              |
| ----------------------|-------------------|------------------------------------|
| Use Case Owner        | Geun-Tak, Roh     | geun-tak.roh@hpe.com               |

## Abstract
Currently, Multi-node Model serving is not officially supported by AI Essentials with Kuberay, MLIS/Kserve
So This demo demonstrates a **Custom Work-around for Multi-node Model serving in PCAI** with **LeaderWorkerSet + vLLM/SGLang**. 


## Steps
As PCAI already includes most of the components ( network operator, High bandwidth network ) for Multi-node model serving, Steps are quite simple.
- Import LeaderWorkerSet in PCAI
- Deploy Model via LWS with Model Serving Frameworks. ( e.g. vllm, SGLang...etc )

### Step 1 — Import LeaderWorkerSet in PCAI
LeaderWorkerSet ([LWS](https://github.com/kubernetes-sigs/lws)) is an API for deploying a group of pods as a unit of replication. It aims to address common deployment patterns of AI/ML inference workloads, especially multi-host inference workloads where the LLM will be sharded and run across multiple devices on multiple nodes.
1. Get LWS Helm chart ( [link](https://github.com/ai-solution-eng/frameworks/tree/main/LeaderWorkerSet) )
2. Import frameworks
    - Log in to the HPE AI Essentials web interface.
    - Click the Tools & Frameworks icon on the left navigation bar.
    - Click + Import Framework.
        - **Framework Name**: of your choice, for example **LeaderWorkerSet**.
        - **Description**: of your choice, for example **github link** or **any descriptions**.
        - **Category**: of your choice, for example Data Science.
        - **Framework Icon**: Click Select File and select the icon you want to use, e.g. the logo file in the [repo](https://github.com/ai-solution-eng/frameworks/tree/main/LeaderWorkerSet).
        - **Helm Chart**: Choose the packaged .tgz chart file in the [repo](https://github.com/ai-solution-eng/frameworks/tree/main/LeaderWorkerSet).
        - **Namespace**: where you deploy the framework, for example **lws**
3. Create Role and Binding to the service account
    - Helm chart will install LWS's Custom Resource Definition.
    - Admin should bind proper role with permission to enable the user handling LeaderWorkerSet custom resource in jupyter notebook.
```bash
$ kubectl create role lws-manager \
  --verb=create,update,patch,delete,get,list,watch \
  --resource=leaderworkersets \
  -n [user's namespace]

$ kubectl create rolebinding lws-manager-binding \
  --role=lws-manager \
  --serviceaccount=[user's namespace]:default-editor \
  -n [user's namespace]
```
    

### Step 2 — Deploy Model via LWS with Model Serving Frameworks.
Following examples are tested in PCAI Gen 1 medium system ( with 8 * L40s ). So The configuration can be differ in another systems like Gen2 systems ( with 4 x 400Gb NIC ). 
- Please update the Huggingface Token environment variable in the manifests.
- For Debugging purpose, we highly recommend to set **NCCL_DEBUG=Info**. Once the deployment is successfully done, Please remove the debug related environment variable.

#### SGLang / 2 RDMA Devices / llama3.3-70b-instruct / 4 Tensor Parallelism x 2 Pipeline Parallelism ( Total 8 GPUs, 4 GPUs from each 2 nodes )
```yaml
apiVersion: leaderworkerset.x-k8s.io/v1
kind: LeaderWorkerSet
metadata:
  name: llama-70b-2rdma-4tp-2pp-pvc
spec:
  replicas: 1
  leaderWorkerTemplate:
    size: 2
    restartPolicy: RecreateGroupOnPodRestart
    leaderTemplate:
      metadata:
        annotations:
          k8s.v1.cni.cncf.io/networks: nvidia-network-operator/nv-ipam-macvlannetwork-a, nvidia-network-operator/nv-ipam-macvlannetwork-b
          sidecar.istio.io/inject: "false"
        labels:
          role: llama-70b-2rdma-4tp-2pp-pvc
      spec:
        containers:
          - name: sglang-leader-ib
            image: lmsysorg/sglang:v0.5.6.post2
            securityContext:
              capabilities:
                add: ["IPC_LOCK"]            
            env:
              - name: HF_TOKEN
                value: '<your HF Token>'
              - name: HF_HOME
                value: '/models'
              - name: NCCL_DEBUG
                value: "INFO"
              - name: NCCL_IB_ADDR_FAMILY
                value: "AF_INET6"     
            command:
              - python3
              - -m
              - sglang.launch_server
              - --model-path
              - "meta-llama/Llama-3.3-70B-Instruct"
              - --tp
              - "4" # Size of Tensor Parallelism
              - --pipeline-parallel-size
              - "2"
              - --dist-init-addr
              - $(LWS_LEADER_ADDRESS):31001
              - --nnodes
              - $(LWS_GROUP_SIZE)
              - --node-rank
              - $(LWS_WORKER_INDEX)
              - --trust-remote-code
              - --host
              - "0.0.0.0"
              - --port
              - "31000"
              - --max-running-requests
              - '500'              
            resources:
              limits:
                nvidia.com/gpu: "4"
                rdma/rdma_shared_device_a: 1
                rdma/rdma_shared_device_b: 1
            ports:
              - containerPort: 31000
            readinessProbe:
              tcpSocket:
                port: 31000
              initialDelaySeconds: 15
              periodSeconds: 30
              failureThreshold: 300
            volumeMounts:
              - mountPath: /dev/shm
                name: dshm
              - name: model-storage
                mountPath: /models                
        volumes:
          - name: dshm
            emptyDir:
              medium: Memory
          - name: model-storage
            persistentVolumeClaim:
              claimName: models-pvc              
    workerTemplate:
      metadata:
        annotations:
          k8s.v1.cni.cncf.io/networks: nvidia-network-operator/nv-ipam-macvlannetwork-a, nvidia-network-operator/nv-ipam-macvlannetwork-b
          sidecar.istio.io/inject: "false"
      spec:
        containers:
          - name: llama-worker-2rdma
            image: lmsysorg/sglang:v0.5.6.post2
            securityContext:
              capabilities:
                add: ["IPC_LOCK"]            
            env:
              - name: HF_TOKEN
                value: '<your HF Token>'
              - name: HF_HOME
                value: '/models'            
              - name: NCCL_DEBUG
                value: "INFO"
              - name: NCCL_IB_ADDR_FAMILY
                value: "AF_INET6"
            command:
              - python3
              - -m
              - sglang.launch_server
              - --model-path
              - "meta-llama/Llama-3.3-70B-Instruct"              
              - --tp
              - "4" # Size of Tensor Parallelism
              - --pipeline-parallel-size
              - "2"
              - --dist-init-addr
              - $(LWS_LEADER_ADDRESS):31001
              - --nnodes
              - $(LWS_GROUP_SIZE)
              - --node-rank
              - $(LWS_WORKER_INDEX)
              - --trust-remote-code
              - --log-level
              - 'debug'
              - --max-running-requests
              - '500'
            resources:
              limits:
                nvidia.com/gpu: "4"
                rdma/rdma_shared_device_a: 1
                rdma/rdma_shared_device_b: 1                
            volumeMounts:
              - mountPath: /dev/shm
                name: dshm
              - name: model-storage
                mountPath: /models                
        volumes:
          - name: dshm
            emptyDir:
              medium: Memory
          - name: model-storage
            persistentVolumeClaim:
              claimName: models-pvc              
---
apiVersion: v1
kind: Service
metadata:
  name: llama-leader-2rdma-4tp-2pp-pvc
spec:
  selector:
    leaderworkerset.sigs.k8s.io/name: llama-70b-2rdma-4tp-2pp-pvc
    role: llama-70b-2rdma-4tp-2pp-pvc
  ports:
    - protocol: TCP
      port: 31000
      targetPort: 31000
```

#### vLLM / 2 RDMA Devices / llama3.3-70b-instruct / 4 Tensor Parallelism x 2 Pipeline Parallelism ( Total 8 GPUs, 4 GPUs from each 2 nodes )
```yaml
apiVersion: leaderworkerset.x-k8s.io/v1
kind: LeaderWorkerSet
metadata:
  name: vllm-llama-70b-2rdma-pvc
spec:
  replicas: 1
  leaderWorkerTemplate:
    size: 2
    restartPolicy: RecreateGroupOnPodRestart
    leaderTemplate:
      metadata:
        annotations:
          k8s.v1.cni.cncf.io/networks: nvidia-network-operator/nv-ipam-macvlannetwork-a, nvidia-network-operator/nv-ipam-macvlannetwork-b
          sidecar.istio.io/inject: "false"
        labels:
          role: vllm-llama-70b-2rdma-pvc
      spec:
        containers:
          - name: vllm-leader-ib
            image: vllm/vllm-openai:v0.10.2
            securityContext:
              capabilities:
                add: ["IPC_LOCK"]            
            env:
              - name: HF_TOKEN
                value: '<your HF token>'
              - name: HF_HOME
                value: '/models'
              - name: NCCL_DEBUG
                value: "INFO"
              - name: NCCL_IB_ADDR_FAMILY
                value: "AF_INET6"   
            command:
              - sh
              - -c
              - "bash /vllm-workspace/examples/online_serving/multi-node-serving.sh leader --ray_cluster_size=$(LWS_GROUP_SIZE); 
                 python3 -m vllm.entrypoints.openai.api_server --port 8080 --model meta-llama/Llama-3.3-70B-Instruct --tensor-parallel-size 4 --pipeline_parallel_size 2"
            resources:
              limits:
                nvidia.com/gpu: "4"
                rdma/rdma_shared_device_a: 1
                rdma/rdma_shared_device_b: 1
            ports:
              - containerPort: 8080
            readinessProbe:
              tcpSocket:
                port: 8080
              initialDelaySeconds: 15
              periodSeconds: 10
            volumeMounts:
              - mountPath: /dev/shm
                name: dshm
              - name: model-storage
                mountPath: /models                
        volumes:
          - name: dshm
            emptyDir:
              medium: Memory
          - name: model-storage
            persistentVolumeClaim:
              claimName: models-pvc              
    workerTemplate:
      metadata:
        annotations:
          k8s.v1.cni.cncf.io/networks: nvidia-network-operator/nv-ipam-macvlannetwork-a, nvidia-network-operator/nv-ipam-macvlannetwork-b
          sidecar.istio.io/inject: "false"
      spec:
        containers:
          - name: vllm-worker-ib
            image: vllm/vllm-openai:v0.10.2
            securityContext:
              capabilities:
                add: ["IPC_LOCK"]            
            env:
              - name: HF_TOKEN
                value: '<your HF token>'
              - name: HF_HOME
                value: '/models'
              - name: NCCL_DEBUG
                value: "INFO"
              - name: NCCL_IB_ADDR_FAMILY
                value: "AF_INET6"
            command:
              - sh
              - -c
              - "bash /vllm-workspace/examples/online_serving/multi-node-serving.sh worker --ray_address=$(LWS_LEADER_ADDRESS)"
            resources:
              limits:
                nvidia.com/gpu: "4"
                rdma/rdma_shared_device_a: 1
                rdma/rdma_shared_device_b: 1                
            volumeMounts:
              - mountPath: /dev/shm
                name: dshm
              - name: model-storage
                mountPath: /models                
        volumes:
          - name: dshm
            emptyDir:
              medium: Memory
          - name: model-storage
            persistentVolumeClaim:
              claimName: models-pvc              
---
apiVersion: v1
kind: Service
metadata:
  name: vllm-leader-ib
spec:
  ports:
    - name: http
      port: 8080
      protocol: TCP
      targetPort: 8080
  selector:
    leaderworkerset.sigs.k8s.io/name: vllm-llama-70b-2rdma-pvc
    role: vllm-llama-70b-2rdma-pvc
  type: ClusterIP
```


## NOTEs
### ⚙️ Enabling GPUDirect RDMA
GPUDirect RDMA is essential for the Performance of Multi-node workloads include Multi-node Model Serving. Following options are mandatory for GPUDirect RDMA in PCAI. if it’s not defined, deployment will face the issues or Deployment will not use GPUDirect RDMA.
- **IPC_LOCK** : capability for locking memory pages and prevent swapping to disk
- **NCCL_IB_ADDR_FAMILY** : for enabling IPv6 in RDMA
- **k8s.v1.cni.cncf.io/networks annotation** : for IPv6 address allocation
- **Shared memory ( /dev/shm )** : for IPC ( interprocess communication )
- **RDMA in resource spec** : for allocate RDMA Devices

Please check the details in the [HPE AI Essential's manual](https://support.hpe.com/hpesc/public/docDisplay?docId=a00aie18hen_us&page=pcai/Operators/gpu-rdma-config.html)

---

### 🚫 Do Not Configure the name of K8s service for LWS leader Pod alinging with LWS's name
Once LeaderWorkerSet custom resource is deployed, it will automatically create k8s service with LWS custom resource's name, and it’s leveraged for inter-pod communication. So if the name is duplicated, then the pods will not be able to resolve each leader/worker pod’s name. 
```bash
$ k get lws
NAME                          READY   DESIRED   UP-TO-DATE   AGE
llama-70b-2rdma-4tp-2pp-pvc   1       1         1            2m

$ k get endpoints | grep "4tp-2pp"
llama-70b-2rdma-4tp-2pp-pvc      10.224.10.80,10.224.28.177   2m7s          <-- K8s service for Leader/Worker Pod
llama-leader-2rdma-4tp-2pp-pvc   10.224.28.177:31000          7h22m         <-- K8s service for Inference Engine in Leader pod

$ k get pods -o wide |grep -i llama
llama-70b-2rdma-4tp-2pp-pvc-0             1/1     Running   0          2m22s   10.224.28.177   scs04.staging.discover.hpepcai.com   <none>           <none>
llama-70b-2rdma-4tp-2pp-pvc-0-1           1/1     Running   0          2m22s   10.224.10.80    scs05.staging.discover.hpepcai.com   <none>           <none>
```
#### NCCL Logs
- As we can see in the NCCL logs, NCCL will point out [Pod name].[service name].[namespace] for initilization.
- So if we override this service, NCCL Initilization will be failed by Name resolution failure
```bash
$ k logs llama-70b-2rdma-4tp-2pp-pvc-0 | grep distributed_init_method
[2025-12-24 16:00:43 PP0 TP0] world_size=8 rank=0 local_rank=0 distributed_init_method=tcp://llama-70b-2rdma-4tp-2pp-pvc-0.llama-70b-2rdma-4tp-2pp-pvc.project-user-geun-tak-roh:31001 backend=nccl
[2025-12-24 16:00:46 PP0 TP1] world_size=8 rank=1 local_rank=1 distributed_init_method=tcp://llama-70b-2rdma-4tp-2pp-pvc-0.llama-70b-2rdma-4tp-2pp-pvc.project-user-geun-tak-roh:31001 backend=nccl
[2025-12-24 16:00:50 PP0 TP2] world_size=8 rank=2 local_rank=2 distributed_init_method=tcp://llama-70b-2rdma-4tp-2pp-pvc-0.llama-70b-2rdma-4tp-2pp-pvc.project-user-geun-tak-roh:31001 backend=nccl
[2025-12-24 16:00:53 PP0 TP3] world_size=8 rank=3 local_rank=3 distributed_init_method=tcp://llama-70b-2rdma-4tp-2pp-pvc-0.llama-70b-2rdma-4tp-2pp-pvc.project-user-geun-tak-roh:31001 backend=nccl
```
----
### 🔑 Power-of-2 GPU allocation and utilize whole GPUs in each nodes.
For multi-node model serving, We recommend allocating 2 ^ n ( power of 2 , 2,4,8… ) number of GPUs and utilize whole GPUs in each nodes. For example, even though the model’s size is fit into 6 GPUs, Using the 8 GPUs from 2 nodes. 

This simplifies:
- HW Topology consideration
- Model architecture constraints (e.g., attention heads must be divisible by tensor parallel degree - [link](https://github.com/vllm-project/vllm/issues/4232) )
---
### ⚠️ Avoid multi-node model serving with Partial GPU Utilization
**Because it would not consider HW topology and it could cause:**
- Significant performance degradation by Poor inter-node communication bandwidth.
- Potential service instability and unexpected errors.
- Hang in NCCL initialization phase

If the model fits within a single node like below example, Please deploy it using in single node with tensor parallelism. Multi node model serving with partial GPU utilization should be avoided even though total # of available GPUs are enough across the cluster.
**Example**: Llama 3.3-70B fits in 4× L40s
- ✅ Deploy on 1 node with 4 GPUs
- ❌ Don't split across 2 nodes (2 GPUs each) even those those GPUs are idle

>AI Essentials 1.10 introduced Topology awareness for scheduling. it likely resolve this restriction but not tested.
---
### ⬇️ Pre-download the model files in PVC
Most of the model serving frameworks download the model artifacts, when the service is up and running. To reduce start-up time and deployment effort, we highly recommend to use K8s Job to download the model into Persistent Volumes. ( In PCAI, **model-pvc** is created in every user's namespace and following K8s job uses **model-pvc** )
- Download the model under ./hub directory. then we can use HF_HOME environment variable to specify local path.
- To launch K8s Job in user’s namespace in PCAI, Please disable Istio sidecar injection via annotation
```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: download-hf-model
spec:
  backoffLimit: 3
  template:
    metadata:
      labels:
        app: hf-model-downloader
      annotations:
        sidecar.istio.io/inject: "false"        
    spec:
      restartPolicy: OnFailure
      containers:
      - name: model-downloader
        image: python:3.11-slim
        command: ["/bin/bash", "-c"]
        args:
          - |
            set -e
            echo "Installing dependencies..."
            pip install --no-cache-dir huggingface_hub
            echo "Starting model download..."
            python3 << 'EOF'
            from huggingface_hub import snapshot_download
            import os
            # Configuration
            model_id = os.getenv("MODEL_ID", "bert-base-uncased")
            cache_dir = os.getenv("CACHE_DIR", "/models/hub")
            hf_token = os.getenv("HF_TOKEN", None)
            print(f"Downloading model: {model_id}")
            print(f"Target directory: {cache_dir}")
            # Create cache directory if it doesn't exist
            if not os.path.exists(cache_dir):
                print(f"Creating cache directory: {cache_dir}")
                os.makedirs(cache_dir, exist_ok=True)
            else:
                print("Cache directory already exists")
            try:
                # Download the model
                local_path = snapshot_download(
                    repo_id=model_id,
                    cache_dir=cache_dir,
                    token=hf_token,
                    resume_download=True
                )
                print(f"\n✓ Model download completed successfully!")
                print(f"✓ Model saved to: {local_path}")
                # List downloaded files
                print("\nDownloaded files:")
                total_size = 0
                file_count = 0
                for root, dirs, files in os.walk(cache_dir):
                    for file in files:
                        filepath = os.path.join(root, file)
                        size = os.path.getsize(filepath)
                        total_size += size
                        file_count += 1
                        # Show relative path for readability
                        rel_path = os.path.relpath(filepath, cache_dir)
                        print(f"  {rel_path} ({size:,} bytes)")
                print(f"\nTotal: {file_count} files, {total_size:,} bytes ({total_size / (1024**3):.2f} GB)")
            except Exception as e:
                print(f"\n✗ Error during download: {e}")
                exit(1)
            EOF
            echo "Job completed successfully!"
        env:
        - name: MODEL_ID
          value: "meta-llama/Llama-3.3-70B-Instruct"  # Change to your desired model
        - name: CACHE_DIR
          value: "/models/hub"
        - name: HF_TOKEN
          value: "<your HF Token>"
        volumeMounts:
        - name: model-storage
          mountPath: /models
        resources:
          requests:
            memory: "2Gi"
            cpu: "1"
          limits:
            memory: "4Gi"
            cpu: "2"
      volumes:
      - name: model-storage
        persistentVolumeClaim:
          claimName: models-pvc
```
---
## References
- [HPE AI Essentials Manual - GPUDirect RDMA Config](https://support.hpe.com/hpesc/public/docDisplay?docId=a00aie18hen_us&page=pcai/Operators/gpu-rdma-config.html)
- [LeaderWorkerSet – Concepts](https://lws.sigs.k8s.io/docs/concepts/)
- [LeaderWorkerSet – Examples](https://lws.sigs.k8s.io/docs/examples/)
- [SGLang – Multi-Node Deployment in K8s](https://docs.sglang.io/references/multi_node_deployment/deploy_on_k8s.html)
- [vLLM – LWS example](https://docs.vllm.ai/en/stable/deployment/frameworks/lws/?h=lws)
- [Parallelism and Scaling - vLLM](https://docs.vllm.ai/en/stable/serving/parallelism_scaling/)
- [Parallelisms — NVIDIA NeMo Framework User Guide](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/features/parallelisms.html#model-parallelism)
- [Environment Variables — NCCL 2.28.9 documentation](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html)