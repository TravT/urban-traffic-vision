job "vision-worker" {
  datacenters = ["dc1"]
  type        = "service"

  # Host constraint: Dell Latitude 7390 Master Node
  constraint {
    attribute = "${node.unique.name}"
    value     = "homelab"
  }

  group "vision" {
    count = 1

    network {
      mode = "host"
      port "http" {
        static = 9099
      }
    }

    # Dynamic Traefik v3 Reverse Proxy Ingress (vision.home.arpa)
    service {
      name     = "vision-worker"
      provider = "nomad"
      port     = "http"
      tags = [
        "traefik.enable=true",
        "traefik.http.routers.vision.rule=Host(`vision.home.arpa`)",
        "traefik.http.routers.vision.entrypoints=web",
        "traefik.http.routers.vision-tls.rule=Host(`vision.home.arpa`)",
        "traefik.http.routers.vision-tls.entrypoints=websecure",
        "traefik.http.routers.vision-tls.tls=true",
        "traefik.http.services.vision-worker.loadbalancer.server.port=9099"
      ]

      check {
        type     = "http"
        path     = "/health"
        interval = "15s"
        timeout  = "3s"
      }
    }

    # Task 1: Urban Vision Appliance (Server, Analytics, QA & Calibration Studio)
    task "vision-appliance" {
      driver = "docker"

      config {
        image        = "homelab/urban-traffic-vision:local"
        force_pull   = false
        network_mode = "host"
        args         = ["9099"]
        volumes = [
          "/home/tlima/Enterprise_Hub/data/media/merged/vision:/data/media/merged/vision",
          "/home/tlima/Enterprise_Hub/dev/urban-traffic-vision/config/zones.json:/app/config/zones.json",
          "/home/tlima/Enterprise_Hub/dev/urban-traffic-vision/config/settings.yaml:/app/config/settings.yaml:ro",
          "/home/tlima/Enterprise_Hub/dev/urban-traffic-vision/src:/app/src:ro"
        ]
      }

      env {
        PYTHONUNBUFFERED = "1"
        VISION_VAL_DIR   = "/data/media/merged/vision/validation"
      }

      resources {
        cpu    = 500
        memory = 512
      }
    }

    # Task 2: Sandboxed YOLOv8 Inference Engine (AVX2 CPU, nice -n 19, ionice -c 3)
    task "inference-engine" {
      driver = "docker"

      config {
        image        = "ultralytics/ultralytics:latest-cpu@sha256:28faf8dac89befba9fa208d8c4c584f755682fbce1f29c4d9284a426df0c9966"
        network_mode = "host"
        volumes = [
          "/home/tlima/Enterprise_Hub/data/media/merged/vision:/vision",
          "/home/tlima/Enterprise_Hub/scripts/vision_pipeline/inference_engine.py:/app/inference_engine.py:ro"
        ]
        command = "sh"
        args = [
          "-c",
          "nice -n 19 ionice -c 3 python3 /app/inference_engine.py --model /vision/yolov8s.pt --imgsz 640 --conf 0.45"
        ]
      }

      resources {
        cpu        = 1000 # 1 core CFS quota
        memory     = 600  # 600 MB limit
        memory_max = 768  # 768 MB hard cap
      }

      env {
        PYTHONUNBUFFERED = "1"
      }
    }
  }
}
